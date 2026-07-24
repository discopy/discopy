# -*- coding: utf-8 -*-

"""
The training harness of the solver family: deep supervision, evaluation
and batching, generic in the task.

The two supervision schemes live in :func:`train_epoch`: the single-run
solvers are supervised on every round of one differentiated run, the
recursion solver on every supervision step of its segmented outer loop.
Everything else -- optimizer stepping, clipping, decode rule, metrics --
is identical, so that a difference in the results is a difference between
the models.

A task keeps its own checkpointing, learning-rate protocol and registry
on top of this module -- see ``sudoku/train.py`` -- and passes its
augmentation as the ``augment`` callable of :func:`batches`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from core.study import Budget, Split

CE = torch.nn.functional.cross_entropy

#: Gradient-norm clipping of every optimizer step of :func:`train_epoch`.
GRAD_CLIP = 1.0


def device_of(model) -> torch.device:
    """ The device the model lives on. """
    return next(model.parameters()).device


def seed_everything(seed: int) -> None:
    """ Fix every source of randomness we use. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def logits_of(model, clues, compute: int = None, deep: bool = False):
    """
    The class logits of any solver of the family under a common interface.

    Parameters:
        model : The solver.
        clues : The puzzles, of shape ``(batch, n_cells)``.
        compute : The test-time compute, i.e. rounds for the single-run
                  solvers and supervision steps for the recursion solver,
                  or ``None`` for the trained one.
        deep : Whether to return the logits of every supervised checkpoint.
    """
    if model.outer_loop:
        return model(clues, deep=deep, n_sup=compute)
    return model(clues, deep=deep, rounds=compute)


def decode(logits, clues):
    """ Argmax classes with the clues written back over the predictions. """
    predicted = logits.argmax(-1) + 1
    return torch.where(clues > 0, clues, predicted)


# --- evaluation -----------------------------------------------------------

@torch.no_grad()
def evaluate(model, split: Split, compute: int = None,
             batch_size: int = 256, buckets: bool = False) -> dict:
    """
    Cell and board accuracy on a split, optionally bucketed by givens.

    Parameters:
        model : The solver.
        split : The split to evaluate on.
        compute : The test-time compute, ``None`` for the trained one.
        batch_size : The evaluation batch size.
        buckets : Whether to also return the per-givens breakdown.
    """
    model.eval()
    device = device_of(model)
    correct, solved = [], []
    for start in range(0, len(split), batch_size):
        stop = start + batch_size
        clues = torch.as_tensor(
            split.puzzles[start:stop], dtype=torch.long, device=device)
        target = torch.as_tensor(
            split.solutions[start:stop], dtype=torch.long, device=device)
        matches = decode(logits_of(model, clues, compute), clues) == target
        correct.append(matches.float().mean(1).cpu().numpy())
        solved.append(matches.all(1).cpu().numpy())
    correct, solved = np.concatenate(correct), np.concatenate(solved)
    result = {"cell": float(correct.mean()), "board": float(solved.mean())}
    if buckets:
        givens = split.givens
        result["by_givens"] = pd.DataFrame({
            "givens": givens, "cell": correct, "board": solved.astype(float),
        }).groupby("givens").agg(
            cell=("cell", "mean"), board=("board", "mean"),
            n=("board", "size")).reset_index()
    return result


@torch.no_grad()
def sweep_compute(model, split: Split, values,
                  batch_size: int = 256) -> pd.DataFrame:
    """ Accuracy as a function of test-time compute. """
    return pd.DataFrame([
        dict(compute=value, **{
            key: score for key, score in
            evaluate(model, split, compute=value,
                     batch_size=batch_size).items()})
        for value in values])


# --- training -------------------------------------------------------------

def batches(split: Split, budget: Budget, rng, device, augment=None):
    """
    One shuffled epoch of ``(clues, target)`` pairs, with the task's
    augmentation applied on the fly when given.

    Parameters:
        split : The training split.
        budget : The budget giving the batch size.
        rng : The ``numpy`` generator behind the shuffle.
        device : The device the batches land on.
        augment : ``(puzzles, solutions, rng) -> (puzzles, solutions)``,
                  or ``None`` for no augmentation.
    """
    order = rng.permutation(len(split))
    for start in range(0, len(order), budget.batch_size):
        index = order[start:start + budget.batch_size]
        puzzles, solutions = split.puzzles[index], split.solutions[index]
        if augment is not None:
            puzzles, solutions = augment(puzzles, solutions, rng)
        yield (torch.as_tensor(puzzles, dtype=torch.long, device=device),
               torch.as_tensor(solutions, dtype=torch.long,
                               device=device) - 1)


def train_epoch(model, split, budget, optimizer, rng, device,
                augment=None) -> dict:
    """
    One epoch under the model's own deep supervision.

    The single-run solvers average the cross-entropy over every round of a
    single backward graph, so one batch is one optimizer step. The
    recursion solver is supervised once per detached segment of its outer
    loop, so one batch is ``n_sup`` optimizer steps -- the residual
    asymmetry between the two schemes, which we report rather than hide.

    Returns the mean loss *per supervised checkpoint* for both schemes, so
    that the two losses are on the same scale.
    """
    model.train()
    total, checkpoints, steps = 0.0, 0, 0
    for clues, target in batches(split, budget, rng, device, augment):
        flat = target.reshape(-1)
        if model.outer_loop:
            state = model.initial(clues)
            for _ in range(model.n_sup):
                state, logits = model.step(state)
                loss = CE(logits.reshape(-1, model.n), flat)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                state, steps = state.detach(), steps + 1
                total, checkpoints = total + loss.item(), checkpoints + 1
        else:
            every = model(clues, deep=True)
            losses = [CE(logits.reshape(-1, model.n), flat)
                      for logits in every]
            loss = sum(losses) / len(losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            steps += 1
            total += sum(item.item() for item in losses)
            checkpoints += len(losses)
    return {"loss": total / max(checkpoints, 1), "opt_steps": steps,
            "checkpoints": checkpoints}
