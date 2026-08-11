# -*- coding: utf-8 -*-

"""
Adaptive computation time, bound to the family's decode rule.

The halt head, the slot-refill trainer and the two early-stopping
evaluations are :mod:`discopy.neural.engine`'s and apply to any skeleton;
this module only binds the fill-in-the-blanks decode rule of
:class:`core.heads.Decoder` so that a caller does not have to pass it.
A task whose decode rule differs binds its own, the same three lines.
"""

from __future__ import annotations

from discopy.neural.engine import (  # noqa: F401 -- re-exported
    ACTEngine, ACTTrainer, HaltHead, PuzzleStream)
from discopy.neural import engine as _engine
from core.heads import Decoder


def evaluate_act(model, split, max_sup: int = None, batch_size: int = 2000,
                 threshold: float = 0.0) -> dict:
    """
    Inference with the paper's early stopping, under the
    fill-in-the-blanks decode rule; see
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


def evaluate_selected(model, split, rollouts: int = 4, sigma: float = 0.1,
                      max_sup: int = None, batch_size: int = 2000,
                      threshold: float = 0.0, seed: int = 0) -> dict:
    """
    Best-of-k inference selected by the halt logit, under the
    fill-in-the-blanks decode rule; see
    :func:`discopy.neural.engine.evaluate_selected`.

    Parameters:
        model : The trained model.
        split : The split to evaluate on.
        rollouts : The number of independent trajectories per problem.
        sigma : The noise level on the answer trace.
        max_sup : The cap on supervision steps.
        batch_size : The evaluation batch size.
        threshold : The margin the halt logit must clear.
        seed : The seed of the noise.
    """
    return _engine.evaluate_selected(
        model, split, Decoder.decode, rollouts=rollouts, sigma=sigma,
        max_sup=max_sup, batch_size=batch_size, threshold=threshold,
        seed=seed)
