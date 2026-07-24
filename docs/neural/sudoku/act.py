# -*- coding: utf-8 -*-

"""
Adaptive computation time for model C: :mod:`core.act` bound to the sudoku
factor graph.  The halt head, the slot-refill trainer and early-stopping
inference are the family's; this module only fixes the skeleton and the
default widths.
"""

from __future__ import annotations

from core.act import (  # noqa: F401 -- trainer and inference re-exported
    ACTTrainer, PuzzleStream, evaluate_act)
from core import act as _act
from sudoku import skeleton
from sudoku.config import N, WIDTHS, Widths


class ACTSolver(_act.ACTSolver):
    """
    Model C with the halt head of :class:`core.act.ACTSolver`, on the
    sudoku factor graph.  Built with the same seed it has bitwise the same
    weights as a plain :class:`sudoku.models.TRMSolver`.

    Parameters:
        widths : The widths of the model, ``WIDTHS["trm"]`` by default.
        rounds : The rounds per cycle, ``n``.
        cycles : The cycles per supervision step, ``T``.
        n_sup : The maximum number of supervision steps.
        n : The size of the grid.
        halt_detach : See :class:`core.act.ACTSolver`.
        halt_head : ``"mean"`` or ``"softmin"``, see
                    :class:`core.act.ACTSolver`.
    """
    def __init__(self, widths: Widths = None, rounds: int = 6,
                 cycles: int = 3, n_sup: int = 8, n: int = N,
                 halt_detach: bool = False, halt_head: str = "mean"):
        super().__init__(skeleton.factor_graph(n),
                         widths or WIDTHS["trm"], rounds, cycles, n_sup,
                         n_classes=n, halt_detach=halt_detach,
                         halt_head=halt_head)
