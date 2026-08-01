# -*- coding: utf-8 -*-

"""
Adaptive computation time for model C.

The halt head, the slot-refill trainer and early-stopping inference are
:mod:`discopy.neural.engine`'s and apply to any skeleton; this module only
binds the sudoku decode rule to :func:`evaluate_act` so that a caller does
not have to pass it.  The model itself is :func:`sudoku.models.act`.
"""

from __future__ import annotations

from discopy.neural.engine import (  # noqa: F401 -- re-exported
    ACTEngine, ACTTrainer, HaltHead, PuzzleStream)
from discopy.neural import engine as _engine
from sudoku.heads import Decoder
from sudoku.models import act  # noqa: F401 -- the builder, re-exported


def evaluate_act(model, split, max_sup: int = None, batch_size: int = 2000,
                 threshold: float = 0.0) -> dict:
    """
    Inference with the paper's early stopping, on sudoku; see
    :func:`discopy.neural.engine.evaluate_act`.

    Parameters:
        model : The trained model.
        split : The split to evaluate on.
        max_sup : The cap on supervision steps.
        batch_size : The evaluation batch size.
        threshold : The margin the halt logit must clear.
    """
    return _engine.evaluate_act(
        model, split, Decoder.decode, max_sup=max_sup,
        batch_size=batch_size, threshold=threshold)
