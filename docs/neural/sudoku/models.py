# -*- coding: utf-8 -*-

"""
The three sudoku solvers: the :mod:`core.solvers` family bound to the
sudoku skeletons and the matched widths of the study.

* **A, :class:`GoISolver`** : :class:`core.solvers.FactorGraphSolver` on
  the bipartite cell/unit factor graph.
* **B, :class:`RRNSolver`** : :class:`core.solvers.CliqueSolver` on the
  peer clique of :cite:t:`PalmEtAl18`.
* **C, :class:`TRMSolver`** : :class:`core.solvers.RecursionSolver` on
  model A's factor graph, with the traced answer loop kept.

Nothing computational lives here: the cells, the interpretation and the
forward passes are the family's.  This module only says *which* skeleton
each model interprets and at *which* widths.
"""

from __future__ import annotations

from core.solvers import (  # noqa: F401 -- cells re-exported for notebooks
    CliqueSolver, FactorBox, FactorGraphSolver, GoICell, RecursionSolver,
    RRNCell, Solver, count_parameters)
from sudoku import skeleton
from sudoku.config import N, WIDTHS, Widths


class GoISolver(FactorGraphSolver):
    """
    Model A: the geometry-of-interaction baseline on the sudoku factor
    graph -- 81 shared cells, 27 shared units, 405 wires.

    Parameters:
        widths : The widths of this model, ``WIDTHS["goi"]`` by default.
        rounds : The default number of message-passing rounds.
        n : The size of the grid.
    """
    def __init__(self, widths: Widths = None, rounds: int = 16, n: int = N):
        super().__init__(skeleton.factor_graph(n),
                         widths or WIDTHS["goi"], rounds, n_classes=n)


class RRNSolver(CliqueSolver):
    """
    Model B: the recurrent relational network on the sudoku peer clique --
    81 shared cells, 972 wires carrying full hidden states.

    Parameters:
        widths : The widths of this model, ``WIDTHS["rrn"]`` by default.
        rounds : The default number of message-passing rounds.
        n : The size of the grid.
    """
    def __init__(self, widths: Widths = None, rounds: int = 16, n: int = N):
        super().__init__(skeleton.clique(n),
                         widths or WIDTHS["rrn"], rounds, n_classes=n)


class TRMSolver(RecursionSolver):
    """
    Model C: the TRM recursion on model A's map -- the same factor graph
    plus an answer loop of width ``y_dim`` on every cell.

    Parameters:
        widths : The widths of this model, ``WIDTHS["trm"]`` by default.
        rounds : The rounds per cycle, ``n``.
        cycles : The cycles per supervision step, ``T``.
        n_sup : The default number of supervision steps.
        n : The size of the grid.
    """
    def __init__(self, widths: Widths = None, rounds: int = 6,
                 cycles: int = 3, n_sup: int = 8, n: int = N):
        super().__init__(skeleton.factor_graph(n),
                         widths or WIDTHS["trm"], rounds, cycles, n_sup,
                         n_classes=n)


BUILDERS = {"goi": GoISolver, "rrn": RRNSolver, "trm": TRMSolver}


def build(name: str, budget=None, widths: Widths = None, **kwargs) -> Solver:
    """
    One solver by name, with the rounds taken from a budget.

    Parameters:
        name : ``"goi"``, ``"rrn"`` or ``"trm"``.
        budget : The :class:`core.study.Budget` giving the depths.
        widths : Widths overriding :data:`sudoku.config.WIDTHS`.
    """
    widths = widths or WIDTHS[name]
    if budget is not None:
        kwargs.setdefault("rounds", budget.trm_n if name == "trm"
                          else budget.rounds)
        if name == "trm":
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
    counts = {name: count_parameters(build(name)) for name in BUILDERS}
    return {"counts": counts, "target": target, "matched": all(
        abs(count - target) <= tolerance * target
        for count in counts.values())}
