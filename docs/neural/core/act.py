# -*- coding: utf-8 -*-

"""
Adaptive computation time, bound to the family's decode rule and
vocabulary.

The halt head, the slot-refill trainer and the two early-stopping
evaluations are :mod:`discopy.neural.engine`'s and apply to any skeleton;
the library speaks only of ``inputs``, ``targets`` and ``solved``.  This
module is where the fill-in-the-blanks family binds its own words to
that interface: a :class:`~core.study.Split` with ``puzzles`` and
``solutions``, the decode rule of :class:`core.heads.Decoder`, and the
``board`` the study's tables have always reported.  A task whose decode
rule differs binds its own, the same few lines.

``PuzzleStream`` remains this layer's name for the library's
:class:`~discopy.neural.engine.ExampleStream`, so existing training
scripts keep resolving.
"""

from __future__ import annotations

from discopy.neural.engine import (  # noqa: F401 -- re-exported
    ACTEngine, ACTTrainer, ExampleStream, HaltHead)
from discopy.neural import engine as _engine
from core.heads import Decoder

#: The family's name for the library's example stream: the study feeds it
#: puzzles.
PuzzleStream = ExampleStream


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

    Returns:
        ``cell`` and ``board`` accuracy and mean ``depth``, in the
        study's vocabulary: ``board`` is the library's ``solved``.
    """
    result = _engine.evaluate_act(
        model, split.puzzles, split.solutions, Decoder.decode,
        max_sup=max_sup, batch_size=batch_size, threshold=threshold)
    return {"cell": result["cell"], "board": result["solved"],
            "depth": result["depth"]}


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

    Returns:
        ``cell``, ``board`` and ``depth`` of the selected rollout, plus
        ``board_mean`` and ``board_oracle``, in the study's vocabulary.
    """
    result = _engine.evaluate_selected(
        model, split.puzzles, split.solutions, Decoder.decode,
        rollouts=rollouts, sigma=sigma, max_sup=max_sup,
        batch_size=batch_size, threshold=threshold, seed=seed)
    return {"cell": result["cell"], "board": result["solved"],
            "board_mean": result["solved_mean"],
            "board_oracle": result["solved_oracle"],
            "depth": result["depth"]}
