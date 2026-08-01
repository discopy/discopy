# -*- coding: utf-8 -*-

"""
The three sudoku solvers, as configurations of one engine.

Nothing computational lives here.  A model is a skeleton, an assignment of
widths to roles, a handful of modules built in a fixed order, and a
:class:`~discopy.neural.engine.Schedule`; the wiring, the cells, the
message passing and the supervision are all
:mod:`discopy.neural`'s.  What this module records is *which* skeleton each
model interprets, at which widths, with which schedule.

* **A, :func:`goi`** -- the bipartite cell/unit factor graph: 81 shared
  cells, 27 shared units, 405 wires; a mean-pooled ``GRUCell`` site and a
  summed Deep-Sets relation; a loss on every round of one differentiated
  run.
* **B, :func:`rrn`** -- the peer clique of :cite:t:`PalmEtAl18`: 81 shared
  cells, 972 wires carrying full hidden states; a summed pairwise message
  into an ``LSTMCell`` site whose two states share one traced loop; same
  supervision.
* **C, :func:`trm`** -- model A's map plus an answer loop of width
  ``y_dim``, run by the segmented recursion of
  :cite:t:`JolicoeurMartineau25`: gradients only through the last cycle of
  each detached supervision step.

Note
----
The ``parts`` mapping is ordered and the order is load-bearing: every
module constructor draws from the global generator, so the embedding, the
cells, the answer refresh, the readout and the initial answer must be
built in exactly this order for a seed to mean the same model.
"""

from __future__ import annotations

import torch

from discopy.neural.cells import Mode, Relation, Site
from discopy.neural.engine import ACTEngine, Engine, HaltHead, RecursionEngine
from discopy.neural.engine import Schedule
from discopy.neural.functor import Interpretation
from sudoku import signature as roles
from sudoku import skeleton
from sudoku.config import N, WIDTHS, Widths
from sudoku.heads import Decoder, Encoder

#: Model A and B supervise every round of one differentiated run.
DEEP = dict(cycles=1, steps=1, inject=True, supervise="round")

#: Model C supervises every detached segment and carries its own clues.
SEGMENTED = dict(inject=False, supervise="step")


def count_parameters(module) -> int:
    """ The number of trainable parameters of a module. """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def goi(widths: Widths = None, rounds: int = 16, n: int = N) -> Engine:
    """
    Model A: the geometry-of-interaction factor graph.

    Parameters:
        widths : The widths of this model, ``WIDTHS["goi"]`` by default.
        rounds : The default number of message-passing rounds.
        n : The size of the grid.
    """
    widths = widths or WIDTHS["goi"]
    grid = skeleton.factor_graph(n)
    return Engine(
        grid, roles.factor_widths(widths),
        parts={
            "embedding": lambda: Encoder(n, widths.dim),
            "cell": lambda: _site(widths, answer_dim=0),
            "factor": lambda: _relation(widths),
            "readout": lambda: Decoder(widths.state_dim, n)},
        sites={"cell": "cell", "unit": "factor"},
        schedule=Schedule(rounds=rounds, **DEEP),
        clue=("cell", roles.CLUE), state=("cell", roles.STATE),
        n_classes=n)


def rrn(widths: Widths = None, rounds: int = 16, n: int = N) -> Engine:
    """
    Model B: the recurrent relational network on the peer clique.

    Parameters:
        widths : The widths of this model, ``WIDTHS["rrn"]`` by default.
        rounds : The default number of message-passing rounds.
        n : The size of the grid.
    """
    widths = widths or WIDTHS["rrn"]
    grid = skeleton.clique(n)
    peers = len(skeleton.peers_of(n)[0])
    return Engine(
        grid, roles.clique_widths(widths),
        parts={
            "embedding": lambda: Encoder(n, widths.dim),
            "cell": lambda: Site(
                roles.peer_cell(peers),
                Interpretation(roles.clique_widths(widths)).widths,
                {roles.HIDDEN: Mode.STATE, roles.MEMORY: Mode.STATE,
                 roles.CLUE: Mode.INPUT},
                hidden=widths.hidden, depth=3, pool="sum",
                recurrent="lstm", emit=False),
            "readout": lambda: Decoder(widths.state_dim, n)},
        sites={"cell": "cell"},
        schedule=Schedule(rounds=rounds, **DEEP),
        clue=("cell", roles.CLUE), state=("cell", roles.HIDDEN),
        n_classes=n)


def trm(widths: Widths = None, rounds: int = 6, cycles: int = 3,
        n_sup: int = 8, n: int = N) -> RecursionEngine:
    """
    Model C: the TRM recursion on model A's map.

    Parameters:
        widths : The widths of this model, ``WIDTHS["trm"]`` by default.
        rounds : The rounds per cycle, ``n``.
        cycles : The cycles per supervision step, ``T``.
        n_sup : The default number of supervision steps.
        n : The size of the grid.
    """
    return _recursion(RecursionEngine, widths or WIDTHS["trm"], rounds,
                      cycles, n_sup, n)


def act(widths: Widths = None, rounds: int = 6, cycles: int = 3,
        n_sup: int = 8, n: int = N, halt_detach: bool = False,
        halt_head: str = "mean") -> ACTEngine:
    """
    Model C with the halt head, i.e. adaptive computation time.

    Built with the same seed it has bitwise the same weights as
    :func:`trm`: the halt head is initialised to constants, and it is
    built last, so it draws exactly the numbers the plain recursion draws
    and no others.

    Parameters:
        widths : The widths of this model, ``WIDTHS["trm"]`` by default.
        rounds : The rounds per cycle, ``n``.
        cycles : The cycles per supervision step, ``T``.
        n_sup : The maximum number of supervision steps.
        n : The size of the grid.
        halt_detach : Whether the head reads a detached answer.
        halt_head : ``"mean"`` or ``"softmin"``; see
                    :class:`discopy.neural.engine.HaltHead`.
    """
    widths = widths or WIDTHS["trm"]
    return _recursion(
        ACTEngine, widths, rounds, cycles, n_sup, n,
        halt=lambda: HaltHead(widths.y_dim, halt_head),
        halt_detach=halt_detach)


BUILDERS = {"goi": goi, "rrn": rrn, "trm": trm, "act": act}


def build(name: str, budget=None, widths: Widths = None, **kwargs) -> Engine:
    """
    One solver by name, with the depths taken from a budget.

    Parameters:
        name : ``"goi"``, ``"rrn"``, ``"trm"`` or ``"act"``.
        budget : The :class:`core.study.Budget` giving the depths.
        widths : Widths overriding :data:`sudoku.config.WIDTHS`.
    """
    widths = widths or WIDTHS["trm" if name == "act" else name]
    if budget is not None:
        recursive = name in ("trm", "act")
        kwargs.setdefault(
            "rounds", budget.trm_n if recursive else budget.rounds)
        if recursive:
            kwargs.setdefault("cycles", budget.trm_T)
            kwargs.setdefault("n_sup", budget.trm_n_sup)
    return BUILDERS[name](widths=widths, **kwargs)


def match_widths(target: int, tolerance: float = 0.1) -> dict:
    """
    Report the parameter count of the three models at the configured
    widths, together with whether they all fall within ``tolerance`` of
    ``target``.

    Parameters:
        target : The parameter count the three models should match.
        tolerance : The relative tolerance, ``0.1`` for the 10% of the
                    fairness protocol.
    """
    counts = {name: count_parameters(build(name))
              for name in ("goi", "rrn", "trm")}
    return {"counts": counts, "target": target, "matched": all(
        abs(count - target) <= tolerance * target
        for count in counts.values())}


# --- the shared pieces of the two factor-graph models ----------------------

def _site(widths: Widths, answer_dim: int, resumable: bool = False) -> Site:
    """ The shared cell of the factor-graph models. """
    return Site(
        roles.cell(3),
        Interpretation(roles.factor_widths(widths, answer_dim)).widths,
        {roles.STATE: Mode.STATE, roles.CLUE: Mode.INPUT,
         roles.ANSWER: Mode.CARRY},
        hidden=widths.hidden, depth=2, pool="mean", recurrent="gru",
        emit=True, resumable=resumable)


def _relation(widths: Widths) -> Relation:
    """ The shared constraint unit of the factor-graph models. """
    return Relation(roles.unit(N), {roles.MESSAGE: widths.dim},
                    hidden=widths.hidden)


def _recursion(factory, widths: Widths, rounds: int, cycles: int,
               n_sup: int, n: int, **kwargs):
    """
    Model C, with or without a halt head: the same parts in the same
    order, and whatever ``kwargs`` the engine adds after them.
    """
    grid = skeleton.factor_graph(n)
    parts = {
        "embedding": lambda: Encoder(n, widths.dim),
        "cell": lambda: _site(widths, widths.y_dim, resumable=True),
        "factor": lambda: _relation(widths),
        "answer": lambda: torch.nn.GRUCell(widths.state_dim, widths.y_dim),
        "answer_norm": lambda: torch.nn.LayerNorm(widths.y_dim),
        "readout": lambda: Decoder(widths.y_dim, n),
        "y0": lambda: torch.nn.Parameter(torch.zeros(widths.y_dim))}
    return factory(
        grid, roles.factor_widths(widths, widths.y_dim),
        parts=parts, sites={"cell": "cell", "unit": "factor"},
        schedule=Schedule(rounds=rounds, cycles=cycles, steps=n_sup,
                          **SEGMENTED),
        clue=("cell", roles.CLUE), state=("cell", roles.STATE),
        n_classes=n, answer=("cell", roles.ANSWER), **kwargs)
