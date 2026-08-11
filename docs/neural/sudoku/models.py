# -*- coding: utf-8 -*-

"""
The three sudoku solvers, as instantiations of the family templates.

Nothing computational lives here, and no longer any *shape* either: the
parts of each solver, their order -- which is load-bearing, every
constructor draws from the global generator -- and their schedules are
:mod:`core.solvers`'s.  What this module records is only what is
sudoku's: which skeleton each model interprets, at which widths, with
which cell hyperparameters, reading which roles.

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
  :cite:t:`JolicoeurMartineau25`; :func:`act` adds the halt head.
"""

from __future__ import annotations

from discopy.neural.cells import Mode, Relation, Site
from discopy.neural.engine import ACTEngine, Engine, HaltHead, RecursionEngine
from discopy.neural.functor import Interpretation
from core.solvers import count_parameters  # noqa: F401 -- re-exported
from core import solvers
from sudoku import signature as roles
from sudoku import skeleton
from sudoku.config import N, WIDTHS, Widths


def goi(widths: Widths = None, rounds: int = 16, n: int = N) -> Engine:
    """
    Model A: the geometry-of-interaction factor graph.

    Parameters:
        widths : The widths of this model, ``WIDTHS["goi"]`` by default.
        rounds : The default number of message-passing rounds.
        n : The size of the grid.
    """
    widths = widths or WIDTHS["goi"]
    return solvers.single_run(
        skeleton.factor_graph(n), roles.factor_widths(widths),
        cell=lambda: _site(widths, answer_dim=0),
        relation=lambda: _relation(widths),
        n_classes=n, dim=widths.dim, state_dim=widths.state_dim,
        rounds=rounds,
        inputs=("cell", roles.CLUE), state=("cell", roles.STATE))


def rrn(widths: Widths = None, rounds: int = 16, n: int = N) -> Engine:
    """
    Model B: the recurrent relational network on the peer clique.

    Parameters:
        widths : The widths of this model, ``WIDTHS["rrn"]`` by default.
        rounds : The default number of message-passing rounds.
        n : The size of the grid.
    """
    widths = widths or WIDTHS["rrn"]
    peers = len(skeleton.peers_of(n)[0])
    return solvers.single_run(
        skeleton.clique(n), roles.clique_widths(widths),
        cell=lambda: Site(
            roles.peer_cell(peers),
            Interpretation(roles.clique_widths(widths)).widths,
            {roles.HIDDEN: Mode.STATE, roles.MEMORY: Mode.STATE,
             roles.CLUE: Mode.INPUT},
            hidden=widths.hidden, depth=3, pool="sum",
            recurrent="lstm", emit=False),
        n_classes=n, dim=widths.dim, state_dim=widths.state_dim,
        rounds=rounds,
        inputs=("cell", roles.CLUE), state=("cell", roles.HIDDEN))


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
    return _recursion(widths or WIDTHS["trm"], rounds, cycles, n_sup, n)


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
        widths, rounds, cycles, n_sup, n,
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


def _recursion(widths: Widths, rounds: int, cycles: int, n_sup: int,
               n: int, **kwargs):
    """ Model C, with or without a halt head, on the family template. """
    return solvers.recursion(
        skeleton.factor_graph(n), roles.factor_widths(widths, widths.y_dim),
        cell=lambda: _site(widths, widths.y_dim, resumable=True),
        relation=lambda: _relation(widths),
        n_classes=n, dim=widths.dim, state_dim=widths.state_dim,
        y_dim=widths.y_dim, rounds=rounds, cycles=cycles, n_sup=n_sup,
        inputs=("cell", roles.CLUE), state=("cell", roles.STATE),
        answer=("cell", roles.ANSWER), **kwargs)
