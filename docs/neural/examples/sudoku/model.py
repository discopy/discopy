# -*- coding: utf-8 -*-

"""
The sudoku models: a diagram, an interpretation, a solver.

Everything generic -- the compilation, the message passing, the cells, the
execution strategies -- is :mod:`discopy.neural`'s.  What is sudoku's is in
this file and nowhere else:

1. the **roles** a wire can carry (:data:`MESSAGE`, :data:`STATE`, ...),
   atomic types naming *what a wire carries* rather than how wide it is;
2. the **signatures** of the two generators, i.e. how many ports each has
   and which of them are one orbit under a permutation -- the members of a
   constraint unit are, since a sudoku constraint does not care which of
   its members is which;
3. the **combinatorics** of the grid, :func:`memberships` and
   :func:`peers_of`, which draw the two diagrams;
4. an **encoder** and a **readout**, the two ends of a fill-in-the-blanks
   task.

Three models come out of those four artefacts, differing only in which
diagram they read, at which widths, and with which solver:

* **A, :func:`goi`** -- the bipartite cell/unit factor graph: 81 shared
  cells, 27 shared units, 405 wires; a mean-pooled ``GRUCell`` site and a
  summed Deep-Sets relation; :class:`~discopy.neural.Iterate` with a loss on
  every round of one differentiated run.
* **B, :func:`rrn`** -- the peer clique of :cite:t:`PalmEtAl18`: 81 shared
  cells, 972 wires carrying full hidden states; a summed pairwise message
  into an ``LSTMCell`` site whose two states share one traced loop; same
  solver.
* **C, :func:`trm`** -- model A's diagram plus one traced answer loop per
  cell, run by :class:`~discopy.neural.Recursion`, the segmented recursion
  of :cite:t:`JolicoeurMartineau25`; :func:`act` adds the halt head.

Models A and C share *one* diagram: A sends the answer role to ``Dim(0)``,
which erases its ports and the wires on them, and C sends it to
``Dim(y_dim)``.

Note
----
The order in which the modules below are constructed is load-bearing: every
constructor draws from the global generator, so permuting them permutes
every weight in the model under a given seed.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from discopy.frobenius import Ty
from discopy.neural import (
    ACT, Dim, HaltHead, Iterate, MapNN, Mode, Orbit, Recursion, Refresh,
    Relation, Signature, Site, Sym, from_incidence, from_relation)

from config import N, WIDTHS, Widths

# --- 1. the roles ----------------------------------------------------------

#: A cell-to-unit wire of the bipartite factor graph.
MESSAGE = Ty("message")

#: A cell-to-cell wire of the pairwise clique.
PEER = Ty("peer")

#: A cell's recurrent state, on a traced loop.
STATE = Ty("state")

#: The two states of an ``LSTMCell``, two named roles on *one* traced loop
#: rather than two halves of one wide port.
HIDDEN, MEMORY = Ty("hidden"), Ty("memory")

#: The clue embedding, on a traced loop.
CLUE = Ty("clue")

#: The current guess of a recursion model, on a traced loop.  Sending it to
#: ``Dim(0)`` erases the loop, which is how one diagram serves the model
#: with an answer and the model without.
ANSWER = Ty("answer")


# --- 2. the signatures -----------------------------------------------------

def cell(degree: int = 3) -> Signature:
    """
    A cell of the factor graph: one message port per unit it belongs to,
    plus a state, a clue and an answer loop.

    Parameters:
        degree : The number of units a cell belongs to.

    Example
    -------
    >>> print(cell().cod)
    message @ message @ message @ state @ state @ clue @ clue @ answer @ \
answer
    """
    return Signature((
        Orbit(MESSAGE, degree, Sym.PERM), Orbit(STATE, traced=True),
        Orbit(CLUE, traced=True), Orbit(ANSWER, traced=True)))


def unit(size: int = N) -> Signature:
    """
    A constraint unit: one port per member, permutation-symmetric.

    Parameters:
        size : The number of cells in a unit.
    """
    return Signature((Orbit(MESSAGE, size, Sym.PERM), ))


def peer_cell(peers: int = 20) -> Signature:
    """
    A cell of the clique: one port per peer, plus one traced loop carrying
    both states of an ``LSTMCell`` and one carrying the clue.

    Parameters:
        peers : The number of peers of a cell.

    Example
    -------
    >>> print(peer_cell(2).cod)
    peer @ peer @ hidden @ memory @ hidden @ memory @ clue @ clue
    >>> peer_cell(2).loops()
    ((2, 4), (3, 5), (6, 7))
    """
    return Signature((
        Orbit(PEER, peers, Sym.PERM), Orbit(HIDDEN @ MEMORY, traced=True),
        Orbit(CLUE, traced=True)))


def factor_ob(widths: Widths, answer_dim: int = 0) -> dict:
    """ The ``Dim`` each role of the factor graph carries. """
    return {MESSAGE: Dim(widths.dim), STATE: Dim(widths.state_dim),
            CLUE: Dim(widths.dim), ANSWER: Dim(answer_dim)}


def clique_ob(widths: Widths) -> dict:
    """
    The ``Dim`` each role of the clique carries: a peer wire carries a full
    hidden state, and so does each of the two states on the loop.
    """
    return {PEER: Dim(widths.state_dim), HIDDEN: Dim(widths.state_dim),
            MEMORY: Dim(widths.state_dim), CLUE: Dim(widths.dim)}


def widths_of(ob) -> dict:
    """ The same mapping with plain integer widths, as a cell reads it. """
    return {role: sum(dim.inside) for role, dim in ob.items()}


# --- 3. the combinatorics, and the two diagrams ----------------------------

@lru_cache(maxsize=None)
def positional_ids(n: int = N):
    """ The row, column and block index of each of the ``n * n`` cells. """
    index = np.arange(n * n)
    root = int(round(n ** 0.5))
    row, col = index // n, index % n
    return row, col, (row // root) * root + (col // root)


@lru_cache(maxsize=None)
def peers_of(n: int = N) -> tuple[tuple[int, ...], ...]:
    """ The cells each cell must differ from: its row, column and block. """
    row, col, block = positional_ids(n)
    return tuple(tuple(
        other for other in range(n * n)
        if other != site and (
            row[other] == row[site] or col[other] == col[site]
            or block[other] == block[site]))
        for site in range(n * n))


@lru_cache(maxsize=None)
def memberships(n: int = N) -> tuple[tuple[int, ...], ...]:
    """
    The three units each cell belongs to -- its row, its column and its
    block, numbered in that order.
    """
    row, col, block = positional_ids(n)
    return tuple(
        (int(row[i]), n + int(col[i]), 2 * n + int(block[i]))
        for i in range(n * n))


@lru_cache(maxsize=None)
def factor_graph(n: int = N):
    """
    The bipartite factor graph of models A and C: every cell is wired to
    the three units it belongs to.

    Interned, so that :meth:`~discopy.neural.MapNN.compile` builds it once
    per process.

    Example
    -------
    >>> grid = factor_graph()
    >>> len(grid.boxes), grid.n_ports // 2
    (108, 486)
    """
    return from_incidence(memberships(n), cell(3), unit(n))


@lru_cache(maxsize=None)
def clique(n: int = N):
    """
    The pairwise peer clique of model B: every cell is wired to each of its
    peers.

    Example
    -------
    >>> grid = clique()
    >>> len(grid.boxes), grid.n_ports // 2
    (81, 1053)
    """
    peers = peers_of(n)
    return from_relation(peers, peer_cell(len(peers[0])))


# --- 4. the two ends of a fill-in-the-blanks task --------------------------

class Encoder(torch.nn.Embedding):
    """
    The clue embedding: one vector per digit, plus one for a blank.

    Example
    -------
    >>> Encoder(9, 4)(torch.zeros(2, 81, dtype=torch.long)).shape
    torch.Size([2, 81, 4])
    """
    def __init__(self, n_classes: int, dim: int):
        super().__init__(n_classes + 1, dim)


class Decoder(torch.nn.Linear):
    """
    The readout: class logits from whatever the solver decodes, plus the
    fill-in-the-blanks rule that turns them into a solution.

    Example
    -------
    >>> logits = torch.zeros(1, 3, 9)
    >>> logits[0, 0, 4] = 1.0
    >>> Decoder.decode(logits, torch.tensor([[0, 7, 0]]))
    tensor([[5, 7, 1]])
    """
    def __init__(self, dim: int, n_classes: int):
        super().__init__(dim, n_classes)

    @staticmethod
    def decode(logits, clues):
        """
        Argmax classes with the clues written back over the predictions: a
        given is never something to predict.
        """
        return torch.where(clues > 0, clues, logits.argmax(-1) + 1)


# --- the model ------------------------------------------------------------

class Solver(torch.nn.Module):
    """
    A sudoku solver: an encoder, a :class:`~discopy.neural.MapNN` and a
    readout.

    The three are an ordinary torch module, so training is an ordinary
    training loop; what the class adds over calling ``MapNN`` directly is
    the two ends of the task and the names of the port families they write
    to and read from.

    Parameters:
        diagram : The wiring to interpret, :func:`factor_graph` or
                  :func:`clique`.
        embedding : The clue encoder.
        interpretation : The :class:`~discopy.neural.MapNN`.
        readout : The class readout.
        answer : The ``(generator name, role)`` the readout decodes.
        initial : The learned initial answer of a recursion model, or
                  ``None``.
        n_classes : The number of digits.
    """
    def __init__(self, diagram, embedding, interpretation, readout,
                 answer: tuple, initial=None, n_classes: int = N):
        super().__init__()
        self.diagram = diagram
        self.embedding = embedding
        self.map = interpretation
        self.readout = readout
        self.y0 = initial
        self.answer = answer
        self.n = n_classes
        self.n_cells = self.map.sites(diagram, ("cell", CLUE))

    # --- what the solver needs to know about the map -----------------------

    @property
    def interaction(self):
        """ The compiled diagram, built once and cached by the model. """
        return self.map.compile(self.diagram)

    @property
    def segmented(self) -> bool:
        """ Whether the solver is supervised once per detached segment. """
        return isinstance(self.map.solver, Recursion)

    @property
    def n_sup(self) -> int:
        """ The supervision steps of a segmented run. """
        return getattr(self.map.solver, "steps", 1)

    @property
    def rounds(self) -> int:
        """ The rounds of message passing per cycle. """
        return self.map.solver.rounds

    @property
    def n_wires(self) -> int:
        """ The number of wires of the compiled diagram. """
        return self.interaction.n_wires

    # --- running it --------------------------------------------------------

    def initial(self, clues):
        """
        The initial flat state: the embedded clues on every cell's clue
        loop, the learned initial answer on its answer loop.

        Parameters:
            clues : The puzzles, of shape ``(rows, 81)``.
        """
        values = {("cell", CLUE): self.embedding(clues)}
        if self.y0 is not None:
            values[self.answer] = self.y0.expand(
                len(clues), self.n_cells, len(self.y0))
        return self.map.initial(self.diagram, values)

    def logits(self, state):
        """ Class logits of shape ``(rows, 81, 9)`` read off a state. """
        return self.readout(self.map.read(self.diagram, state, self.answer))

    def step(self, state, cycles: int = None, grad: bool = True):
        """
        One supervision step of the segmented recursion, and the logits it
        ends on -- the unit a training loop takes an optimizer step on.
        """
        state = Recursion.step(
            self.map.solver, self.interaction, state, cycles, grad)
        return state, self.logits(state)

    def act_step(self, state, cycles: int = None, grad: bool = True):
        """ One supervision step of :class:`~discopy.neural.ACT`. """
        state, answer, halt = self.map.solver.step(
            self.interaction, state, cycles, grad)
        return state, self.readout(answer), halt

    def halt_logit(self, halt):
        """ The problem-level halt logit. """
        return self.map.solver.halt.logit(halt)

    def halt_loss(self, halt, logits, targets):
        """ The halt head's loss against the correctness of ``logits``. """
        with torch.no_grad():
            correct = logits.argmax(-1) == targets
        return self.map.solver.halt.loss(halt, correct)

    def forward(self, clues, deep: bool = False, **overrides):
        """
        The class logits of a whole run, at every supervised checkpoint
        when ``deep``, at the last one otherwise.

        Parameters:
            clues : The puzzles, of shape ``(rows, 81)``.
            deep : Whether to return every checkpoint.
            overrides : Test-time compute, ``rounds`` for the single-run
                        models and ``steps`` for the recursion.
        """
        states = self.map(self.diagram, self.initial(clues), deep=deep,
                          **overrides)
        return [self.logits(state) for state in states] if deep \
            else self.logits(states)


# --- the three models ------------------------------------------------------

def _site(widths: Widths, answer_dim: int, resumable: bool = False) -> Site:
    """ The shared cell of the factor-graph models. """
    return Site(
        cell(3), widths_of(factor_ob(widths, answer_dim)),
        {STATE: Mode.STATE, CLUE: Mode.INPUT, ANSWER: Mode.CARRY},
        hidden=widths.hidden, depth=2, pool="mean", recurrent="gru",
        emit=True, resumable=resumable)


def _relation(widths: Widths) -> Relation:
    """ The shared constraint unit of the factor-graph models. """
    return Relation(unit(N), {MESSAGE: widths.dim}, hidden=widths.hidden)


def goi(widths: Widths = None, rounds: int = 16, n: int = N) -> Solver:
    """
    Model A: the geometry-of-interaction factor graph, supervised on every
    round of one differentiated run.

    Parameters:
        widths : The widths, ``WIDTHS["goi"]`` by default.
        rounds : The default number of message-passing rounds.
        n : The size of the grid.
    """
    widths = widths or WIDTHS["goi"]
    embedding = Encoder(n, widths.dim)
    site, relation = _site(widths, 0), _relation(widths)
    readout = Decoder(widths.state_dim, n)
    return Solver(
        factor_graph(n), embedding,
        MapNN(factor_ob(widths), {"cell": site, "unit": relation},
              solver=Iterate(rounds=rounds, inject=True)),
        readout, answer=("cell", STATE), n_classes=n)


def rrn(widths: Widths = None, rounds: int = 16, n: int = N) -> Solver:
    """
    Model B: the recurrent relational network on the peer clique.

    Parameters:
        widths : The widths, ``WIDTHS["rrn"]`` by default.
        rounds : The default number of message-passing rounds.
        n : The size of the grid.
    """
    widths = widths or WIDTHS["rrn"]
    peers = len(peers_of(n)[0])
    embedding = Encoder(n, widths.dim)
    site = Site(
        peer_cell(peers), widths_of(clique_ob(widths)),
        {HIDDEN: Mode.STATE, MEMORY: Mode.STATE, CLUE: Mode.INPUT},
        hidden=widths.hidden, depth=3, pool="sum", recurrent="lstm",
        emit=False)
    readout = Decoder(widths.state_dim, n)
    return Solver(
        clique(n), embedding,
        MapNN(clique_ob(widths), {"cell": site},
              solver=Iterate(rounds=rounds, inject=True)),
        readout, answer=("cell", HIDDEN), n_classes=n)


def _recursion(widths: Widths, rounds: int, cycles: int, steps: int, n: int,
               halt: str = None, halt_detach: bool = False) -> Solver:
    """ Model C, with or without a halt head. """
    embedding = Encoder(n, widths.dim)
    site = _site(widths, widths.y_dim, resumable=True)
    relation = _relation(widths)
    refresh = Refresh(
        torch.nn.GRUCell(widths.state_dim, widths.y_dim),
        torch.nn.LayerNorm(widths.y_dim),
        source=("cell", STATE), target=("cell", ANSWER))
    readout = Decoder(widths.y_dim, n)
    initial = torch.nn.Parameter(torch.zeros(widths.y_dim))
    # the halt head is built last, so that a model with one draws exactly
    # the random numbers a model without draws, and no others.
    solver = Recursion(rounds, cycles, steps, refresh=refresh) \
        if halt is None else ACT(
            rounds, cycles, steps, refresh=refresh,
            halt=HaltHead(widths.y_dim, halt), detach=halt_detach)
    return Solver(
        factor_graph(n), embedding,
        MapNN(factor_ob(widths, widths.y_dim),
              {"cell": site, "unit": relation}, solver=solver),
        readout, answer=("cell", ANSWER), initial=initial, n_classes=n)


def trm(widths: Widths = None, rounds: int = 6, cycles: int = 3,
        steps: int = 8, n: int = N) -> Solver:
    """
    Model C: the segmented recursion on model A's diagram.

    Parameters:
        widths : The widths, ``WIDTHS["trm"]`` by default.
        rounds : The rounds per cycle, ``n`` in the paper.
        cycles : The cycles per supervision step, ``T``.
        steps : The default number of supervision steps.
        n : The size of the grid.
    """
    return _recursion(widths or WIDTHS["trm"], rounds, cycles, steps, n)


def act(widths: Widths = None, rounds: int = 6, cycles: int = 3,
        steps: int = 8, n: int = N, halt_detach: bool = False,
        halt_head: str = "mean") -> Solver:
    """
    Model C with the halt head, i.e. adaptive computation time.

    Built with the same seed it has bitwise the same weights as
    :func:`trm`: the halt head is initialised to constants and built last.

    Parameters:
        widths : The widths, ``WIDTHS["trm"]`` by default.
        rounds, cycles, steps : The recursion shape.
        n : The size of the grid.
        halt_detach : Whether the head reads a detached answer.
        halt_head : ``"mean"`` or ``"softmin"``.
    """
    return _recursion(widths or WIDTHS["trm"], rounds, cycles, steps, n,
                      halt=halt_head, halt_detach=halt_detach)


BUILDERS = {"goi": goi, "rrn": rrn, "trm": trm, "act": act}


def count_parameters(module) -> int:
    """ The number of trainable parameters of a module. """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build(name: str, budget=None, widths: Widths = None, **kwargs) -> Solver:
    """
    One model by name, with the depths taken from a budget.

    Parameters:
        name : ``"goi"``, ``"rrn"``, ``"trm"`` or ``"act"``.
        budget : The :class:`~config.Budget` giving the depths.
        widths : Widths overriding :data:`config.WIDTHS`.
    """
    widths = widths or WIDTHS[name]
    if budget is not None:
        kwargs.setdefault("rounds", budget.rounds)
        if name in ("trm", "act"):
            kwargs.setdefault("cycles", budget.cycles)
            kwargs.setdefault("steps", budget.steps)
    return BUILDERS[name](widths=widths, **kwargs)


#: How the keys of a checkpoint trained before the refactor map to the keys
#: of a model built here.  The weights never moved -- same shapes, same
#: values, same order -- but the modules did: the generators now live under
#: the ``MapNN`` that shares them and the answer refresh under the solver
#: that runs it.  The clique cell used to be an ``RRNCell`` whose pairwise
#: message function was called ``message``; it is a
#: :class:`~discopy.neural.Site` whose encoder is called ``encode``, because
#: that is what the same layer is called in every other site.
RENAMES = (
    ("cell.", "map.ar.cell."),
    ("factor.", "map.ar.unit."),
    ("answer_norm.", "map.solver.refresh.norm."),
    ("answer.", "map.solver.refresh.module."),
    ("q_head.", "map.solver.halt."),
)


def rename(key: str) -> str:
    """
    The name a pre-refactor checkpoint key has now.

    Example
    -------
    >>> rename("cell.message.0.weight")
    'map.ar.cell.encode.0.weight'
    >>> rename("answer_norm.weight"), rename("readout.bias"), rename("y0")
    ('map.solver.refresh.norm.weight', 'readout.bias', 'y0')
    """
    key = key.replace(".message.", ".encode.")
    for old, new in RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def translate(state_dict: dict) -> dict:
    """ A whole pre-refactor ``state_dict``, renamed. """
    return {rename(key): value for key, value in state_dict.items()}


def load_checkpoint(module, path, strict: bool = True):
    """
    Load a checkpoint into a model, translating pre-refactor keys.

    Parameters:
        module : The model to load into.
        path : A bare ``state_dict`` or the study's
               ``{"state_dict": ..., ...}`` record.
        strict : Whether to require every key to match, as it should.

    Returns:
        Whatever ``torch.load`` read, so that the caller can still see the
        history and metadata beside the weights.
    """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    weights = stored["state_dict"] if isinstance(stored, dict) \
        and "state_dict" in stored else stored
    module.load_state_dict(translate(weights), strict=strict)
    return stored


def match_widths(target: int, tolerance: float = 0.1) -> dict:
    """
    The parameter count of the three models at the configured widths,
    together with whether they all fall within ``tolerance`` of ``target``.

    Example
    -------
    >>> from config import PARAM_TARGET
    >>> report = match_widths(PARAM_TARGET)
    >>> report["matched"]
    True
    """
    counts = {name: count_parameters(build(name))
              for name in ("goi", "rrn", "trm")}
    return {"counts": counts, "target": target, "matched": all(
        abs(count - target) <= tolerance * target
        for count in counts.values())}
