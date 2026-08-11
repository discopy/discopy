# -*- coding: utf-8 -*-

"""
Adaptive computation time for model C -- an import path, not a module.

Everything that used to be here dissolved upward: the halt head, the
slot-refill trainer and early stopping are :mod:`discopy.neural.engine`'s
and apply to any skeleton; the decode-bound evaluations are
:mod:`core.act`'s and apply to any fill-in-the-blanks task.  What is
sudoku's is only the model itself, :func:`sudoku.models.act`, re-exported
here alongside the rest so that existing callers keep resolving.
"""

from __future__ import annotations

from core.act import (  # noqa: F401 -- re-exported
    ACTEngine, ACTTrainer, ExampleStream, HaltHead, PuzzleStream,
    evaluate_act, evaluate_selected)
from sudoku.models import act  # noqa: F401 -- the builder, re-exported

__all__ = ["ACTEngine", "ACTTrainer", "ExampleStream", "HaltHead",
           "PuzzleStream", "act", "evaluate_act", "evaluate_selected"]
