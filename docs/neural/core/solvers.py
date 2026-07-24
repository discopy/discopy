# -*- coding: utf-8 -*-

"""
The solver family: three architectures over a constraint-satisfaction
skeleton, generic in the task.

All three are closed combinatorial maps in :mod:`discopy.neural` and share
an input embedding, a linear readout and the decode rule of
:mod:`core.train`; they differ in the shape of the skeleton they
interpret, the width carried by a wire, the update cell, and -- for the
recursion solver -- the *evaluation strategy* that composes macro-steps of
message passing.

* :class:`FactorGraphSolver` : a bipartite variable/unit factor graph, a
  shared GRU cell wired to shared Deep-Sets unit boxes, supervised at
  every round.
* :class:`CliqueSolver` : the pairwise peer clique of :cite:t:`PalmEtAl18`,
  full hidden states on the wires, pairwise messages sum-pooled into an
  ``LSTMCell``, supervised at every round.
* :class:`RecursionSolver` : the factor graph plus a traced answer loop,
  run by the segmented outer loop of :cite:t:`JolicoeurMartineau25`.

A task instantiates a solver by handing it the abstract skeleton built
from its combinatorics -- see ``sudoku/models.py`` for the worked example.
Every module asserts the shape of what it reads and what it emits, since a
mis-sliced port would otherwise train quietly into a wrong model.
"""

from __future__ import annotations

import torch

from core.cmaps import Router
from core.functors import build, clique_functor, factor_functor
from core.study import Widths


def count_parameters(module) -> int:
    """ The number of trainable parameters of a module. """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --- the boxes ------------------------------------------------------------

class FactorBox(torch.nn.Module):
    """
    The shared unit box of the factor-graph solvers: a permutation-
    equivariant Deep-Sets relation over the members of a constraint unit.
    It embeds each incoming message, *sums* the embeddings into an
    order-invariant summary of the unit, and answers each member with that
    summary alongside its own message.

    Parameters:
        dim : The width of a member message.
        hidden : The width of the hidden layers.
    """
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dim = dim
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.ReLU())
        self.rho = torch.nn.Sequential(
            torch.nn.Linear(dim + hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, dim))

    def forward(self, x):
        members = x.shape[-1] // self.dim
        assert x.shape[-1] == members * self.dim, "ragged unit box input"
        message = x.reshape(-1, members, self.dim)
        pooled = self.phi(message).sum(1, keepdim=True).expand(
            -1, members, -1)
        out = self.rho(torch.cat([message, pooled], -1))
        assert out.shape == message.shape, "unit box changed the port widths"
        return out.reshape(-1, members * self.dim)


class GoICell(torch.nn.Module):
    """
    The shared cell box of the factor-graph solvers.

    It reads its ports as ``[m_1 .. m_P | h, h' | c, c' (| y, y')]``: ``P``
    unit messages, a state loop, a clue loop and, for a recursion solver,
    an answer loop. Each round it encodes every incoming message against
    its own state, *mean*-pools the encodings, runs a ``GRUCell`` from the
    pool, the clue and (when present) the answer, normalises the result,
    and broadcasts a fresh belief to its units.

    On the clue loop it emits zeros when ``resumable`` is false, so that
    the clue must be re-injected every round by ``inject=True``; it emits
    the clue back when ``resumable`` is true, so that the run carries its
    own clues and can be stopped and restarted with ``inject=False``. The
    answer loop is passed through unchanged: the cell reads ``y`` but
    never writes it, the outer loop does.

    Parameters:
        dim : The width of a message and of a clue.
        state_dim : The width of the state.
        hidden : The width of the hidden layers.
        answer_dim : The width of the answer loop, ``0`` for none.
        resumable : Whether to re-emit the clue rather than zeros.
    """
    def __init__(self, dim: int, state_dim: int, hidden: int,
                 answer_dim: int = 0, resumable: bool = False):
        super().__init__()
        self.dim, self.state_dim = dim, state_dim
        self.answer_dim, self.resumable = answer_dim, resumable
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(state_dim + dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden))
        self.update = torch.nn.GRUCell(hidden + dim + answer_dim, state_dim)
        self.norm = torch.nn.LayerNorm(state_dim)
        self.emit = torch.nn.Linear(state_dim, dim)

    def forward(self, x):
        dim, state_dim, answer_dim = self.dim, self.state_dim, self.answer_dim
        width = x.shape[-1] - 2 * state_dim - 2 * dim - 2 * answer_dim
        n_message = width // dim
        assert width == n_message * dim and n_message > 0, \
            f"cannot read {x.shape[-1]} as a cell of width {dim}"
        cursor = n_message * dim
        message = x[:, :cursor].reshape(-1, n_message, dim)
        state = x[:, cursor:cursor + state_dim]
        cursor += 2 * state_dim
        clue = x[:, cursor:cursor + dim]
        cursor += 2 * dim
        answer = x[:, cursor:cursor + answer_dim]

        pooled = self.encode(torch.cat([
            state.unsqueeze(1).expand(-1, n_message, -1), message], -1)
        ).mean(1)
        state = self.norm(self.update(
            torch.cat([pooled, clue, answer], -1), state))
        assert state.shape[-1] == state_dim, "the state changed width"
        belief = self.emit(state).unsqueeze(1).expand(
            -1, n_message, -1).reshape(-1, n_message * dim)
        blank = clue if self.resumable else torch.zeros_like(clue)
        out = torch.cat(
            [belief, state, state, blank, blank]
            + ([answer, answer] if answer_dim else []), -1)
        assert out.shape == x.shape, "the cell changed its port widths"
        return out


class RRNCell(torch.nn.Module):
    """
    The shared cell box of the clique solver, faithful to
    :cite:t:`PalmEtAl18`.

    A peer wire carries a full hidden state, so the message a cell receives
    from a peer *is* that peer's ``h``. The cell forms the pairwise message
    ``f([h_own, h_peer])`` for each of its peers and **sums** them, then
    updates its node state with an ``LSTMCell`` reading ``[pooled, clue]``.
    Computing ``f`` at the receiver rather than at the sender is
    mathematically the same edge function, since the receiver holds both
    endpoint states; broadcasting ``h`` and evaluating ``f`` on arrival
    just lets one shared box play both roles.

    The state loop carries ``[h, c]`` concatenated, since an ``LSTMCell``
    keeps two states; only ``h`` goes out on the peer wires.

    Parameters:
        dim : The width of a clue.
        state_dim : The width of ``h``, hence of a peer wire.
        hidden : The width of the message function.
    """
    def __init__(self, dim: int, state_dim: int, hidden: int):
        super().__init__()
        self.dim, self.state_dim = dim, state_dim
        self.message = torch.nn.Sequential(
            torch.nn.Linear(2 * state_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden))
        self.update = torch.nn.LSTMCell(hidden + dim, state_dim)
        self.norm = torch.nn.LayerNorm(state_dim)

    def forward(self, x):
        dim, state_dim = self.dim, self.state_dim
        width = x.shape[-1] - 4 * state_dim - 2 * dim
        n_peers = width // state_dim
        assert width == n_peers * state_dim and n_peers > 0, \
            f"cannot read {x.shape[-1]} as a clique cell"
        cursor = n_peers * state_dim
        peers = x[:, :cursor].reshape(-1, n_peers, state_dim)
        hidden = x[:, cursor:cursor + state_dim]
        memory = x[:, cursor + state_dim:cursor + 2 * state_dim]
        cursor += 4 * state_dim
        clue = x[:, cursor:cursor + dim]

        pairs = torch.cat([
            hidden.unsqueeze(1).expand(-1, n_peers, -1), peers], -1)
        pooled = self.message(pairs).sum(1)
        hidden, memory = self.update(
            torch.cat([pooled, clue], -1), (hidden, memory))
        hidden = self.norm(hidden)
        assert hidden.shape[-1] == state_dim, "the state changed width"
        state = torch.cat([hidden, memory], -1)
        broadcast = hidden.unsqueeze(1).expand(-1, n_peers, -1).reshape(
            -1, n_peers * state_dim)
        blank = torch.zeros_like(clue)
        out = torch.cat([broadcast, state, state, blank, blank], -1)
        assert out.shape == x.shape, "the cell changed its port widths"
        return out


# --- the solvers ----------------------------------------------------------

class Solver(torch.nn.Module):
    """
    What the solvers share: an input embedding, a linear readout, the map
    and its layout, and the ports that clues and states live on.

    Parameters:
        widths : The widths of this solver.
        n_classes : The number of classes a variable can take.
    """
    #: Whether training reads a list of per-round logits or drives the
    #: outer loop itself (the recursion solver).
    outer_loop = False

    def __init__(self, widths: Widths, n_classes: int):
        super().__init__()
        self.widths, self.n = widths, n_classes
        self.embedding = torch.nn.Embedding(n_classes + 1, widths.dim)

    def _finish(self, cmap, layout):
        """ Cache the map, its router and the port families we read. """
        self.grid, self.layout = cmap, layout
        self.n_cells = layout.n_cells
        self.cells = cmap.as_network().module
        self.router = Router(cmap)
        ports = [cmap.box_ports(cell) for cell in range(self.n_cells)]
        self.clue_ports = tuple(
            port for cell in ports for port in
            (cell[layout.clue[0]], cell[layout.clue[1]]))
        self.state_ports = tuple(cell[layout.state[0]] for cell in ports)
        self.answer_ports = tuple(
            port for cell in ports for port in
            (cell[layout.answer[0]], cell[layout.answer[1]])
        ) if layout.answer else ()

    def compile_cells(self, **kwargs):
        """
        Compile the round step of the map with ``torch.compile``; see
        :meth:`discopy.neural.CMap.compile`. Message passing on these maps
        is launch-bound (many small kernels per round), so this is a
        several-fold wall-clock speedup on a GPU at identical numerics up
        to rounding error. Named ``compile_cells`` so as not to shadow
        ``torch.nn.Module.compile``.
        """
        self.grid.compile(**kwargs)
        return self

    def initial(self, clues):
        """
        The initial per-port messages: the clue embedding on both ends of
        every clue loop, zeros everywhere else.

        Parameters:
            clues : The puzzles, of shape ``(batch, n_cells)``.
        """
        assert clues.shape[1] == self.n_cells, "wrong number of cells"
        embedded = self.embedding(clues)
        assert embedded.shape == (
            len(clues), self.n_cells, self.widths.dim), "bad embedding shape"
        flat = torch.zeros(len(clues), self.router.total,
                           dtype=embedded.dtype, device=embedded.device)
        return self.router.write(
            flat, self.clue_ports,
            embedded.repeat_interleave(2, dim=1))

    def readout_from(self, states):
        """ Class logits of shape ``(batch, n_cells, n)`` from states. """
        logits = self.readout(states)
        assert logits.shape[1:] == (self.n_cells, self.n), \
            "bad logits shape"
        return logits

    @property
    def n_wires(self) -> int:
        """ The number of wires of the map. """
        return self.grid.n_ports // 2


class FactorGraphSolver(Solver):
    """
    The geometry-of-interaction factor graph.

    Clues enter as initial messages on the traced clue loops and are
    re-injected every round (``inject=True``); a shared linear head reads
    every cell's state. With ``deep=True`` the forward pass returns the
    logits of *every* round, which is how deep supervision is applied.

    Parameters:
        abstract : The factor-graph skeleton to interpret, from
                   :func:`core.skeletons.factor_graph`.
        widths : The widths of this solver.
        rounds : The default number of message-passing rounds.
        n_classes : The number of classes a variable can take.
    """
    def __init__(self, abstract, widths: Widths, rounds: int,
                 n_classes: int):
        super().__init__(widths, n_classes)
        self.rounds = rounds
        self.cell = GoICell(widths.dim, widths.state_dim, widths.hidden)
        self.factor = FactorBox(widths.dim, widths.hidden)
        self.readout = torch.nn.Linear(widths.state_dim, n_classes)
        functor = factor_functor(
            abstract, self.cell, self.factor, widths.dim, widths.state_dim)
        self._finish(*build(functor, abstract))

    def forward(self, clues, deep: bool = False, rounds: int = None):
        init = self.initial(clues)
        emitted = self.cells(init=init, n_rounds=rounds or self.rounds,
                             inject=True, return_rounds=deep,
                             return_flat=True)

        def head(flat):
            states = self.router.read(flat, self.state_ports)
            return self.readout_from(states)
        return [head(step) for step in emitted] if deep else head(emitted)


class CliqueSolver(Solver):
    """
    The recurrent relational network of :cite:t:`PalmEtAl18`.

    Same map-level semantics as the factor-graph solver -- clues injected
    on a traced loop, a shared readout, a loss on every round -- but the
    wiring is the pairwise peer clique and the wires carry hidden states.

    Parameters:
        abstract : The clique skeleton to interpret, from
                   :func:`core.skeletons.clique`.
        widths : The widths of this solver.
        rounds : The default number of message-passing rounds.
        n_classes : The number of classes a variable can take.
    """
    def __init__(self, abstract, widths: Widths, rounds: int,
                 n_classes: int):
        super().__init__(widths, n_classes)
        self.rounds = rounds
        self.cell = RRNCell(widths.dim, widths.state_dim, widths.hidden)
        self.readout = torch.nn.Linear(widths.state_dim, n_classes)
        functor = clique_functor(
            abstract, self.cell, widths.state_dim, widths.dim)
        self._finish(*build(functor, abstract))

    def forward(self, clues, deep: bool = False, rounds: int = None):
        init = self.initial(clues)
        emitted = self.cells(init=init, n_rounds=rounds or self.rounds,
                             inject=True, return_rounds=deep,
                             return_flat=True)

        def head(flat):
            states = self.router.read(flat, self.state_ports)
            return self.readout_from(states[..., :self.widths.state_dim])
        return [head(step) for step in emitted] if deep else head(emitted)


class RecursionSolver(Solver):
    """
    The factor-graph map with one extra traced loop, run by the segmented
    outer loop of :cite:t:`JolicoeurMartineau25`.

    The map is the factor-graph solver's, plus an answer loop of width
    ``y_dim`` on every cell which the cell reads but passes through
    unchanged, and a cell that re-emits its clue so a run carries its own
    clues. Message passing is then resumable: :meth:`cycle` runs ``n``
    rounds with ``inject=False`` and reads back the flat incoming messages
    (``return_flat=True``, the same tensor :func:`core.cmaps.route` would
    rebuild), and the answer ``y`` is refreshed from the latent state
    ``z`` by a ``GRUCell`` before the next macro-step. One supervision
    step is ``T`` such cycles, the first ``T - 1`` without gradients.

    Parameters:
        abstract : The factor-graph skeleton to interpret.
        widths : The widths of this solver.
        rounds : The rounds per cycle, ``n``.
        cycles : The cycles per supervision step, ``T``.
        n_sup : The default number of supervision steps.
        n_classes : The number of classes a variable can take.
    """
    outer_loop = True

    def __init__(self, abstract, widths: Widths, rounds: int, cycles: int,
                 n_sup: int, n_classes: int):
        super().__init__(widths, n_classes)
        self.rounds, self.cycles, self.n_sup = rounds, cycles, n_sup
        self.cell = GoICell(widths.dim, widths.state_dim, widths.hidden,
                            answer_dim=widths.y_dim, resumable=True)
        self.factor = FactorBox(widths.dim, widths.hidden)
        self.answer = torch.nn.GRUCell(widths.state_dim, widths.y_dim)
        self.answer_norm = torch.nn.LayerNorm(widths.y_dim)
        self.readout = torch.nn.Linear(widths.y_dim, n_classes)
        self.y0 = torch.nn.Parameter(torch.zeros(widths.y_dim))
        functor = factor_functor(
            abstract, self.cell, self.factor, widths.dim, widths.state_dim,
            answer_dim=widths.y_dim)
        self._finish(*build(functor, abstract))

    def initial(self, clues):
        """ The shared initial messages plus the learned answer ``y_0``. """
        flat = super().initial(clues)
        y = self.y0.expand(len(clues), self.n_cells, self.widths.y_dim)
        return self.router.write(
            flat, self.answer_ports, y.repeat_interleave(2, dim=1))

    def cycle(self, state, rounds: int = None):
        """
        One macro-step: ``rounds`` rounds of resumable message passing,
        then one refresh of the answer from the latent state.

        Parameters:
            state : The flat incoming messages, of width ``router.total``.
            rounds : The rounds of this cycle, ``self.rounds`` by default.
        """
        assert state.shape[-1] == self.router.total, "bad state width"
        state = self.cells(init=state, n_rounds=rounds or self.rounds,
                           inject=False, return_flat=True)
        latent = self.router.read(state, self.state_ports)
        answer = self.router.read(state, self.answer_ports)[:, ::2]
        assert latent.shape[1:] == (self.n_cells, self.widths.state_dim)
        assert answer.shape[1:] == (self.n_cells, self.widths.y_dim)
        updated = self.answer_norm(self.answer(
            latent.reshape(-1, self.widths.state_dim),
            answer.reshape(-1, self.widths.y_dim))
        ).reshape(len(state), self.n_cells, self.widths.y_dim)
        return self.router.write(
            state, self.answer_ports, updated.repeat_interleave(2, dim=1))

    def step(self, state, cycles: int = None, grad: bool = True):
        """
        One supervision step: ``T - 1`` cycles without gradients and one
        with, returning the new state and the logits read off the answer.

        Parameters:
            state : The flat incoming messages.
            cycles : The cycles of this step, ``self.cycles`` by default.
            grad : Whether the last cycle is differentiated.
        """
        cycles = self.cycles if cycles is None else cycles
        with torch.no_grad():
            for _ in range(cycles - 1):
                state = self.cycle(state)
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            state = self.cycle(state)
            answer = self.router.read(state, self.answer_ports)[:, ::2]
            logits = self.readout_from(answer)
        return state, logits

    def forward(self, clues, deep: bool = False, n_sup: int = None,
                cycles: int = None):
        """
        Run the whole outer loop without gradients, e.g. to evaluate.

        Training does not call this: it interleaves :meth:`step` with a
        backward pass and an optimizer step, detaching the state in
        between.
        """
        n_sup = n_sup or self.n_sup
        state, every = self.initial(clues), []
        for _ in range(n_sup):
            state, logits = self.step(state, cycles=cycles, grad=False)
            state = state.detach()
            every.append(logits)
        return every if deep else every[-1]
