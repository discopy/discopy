# -*- coding: utf-8 -*-

"""
The training ingredients shared by the searched recipes: the optimizer,
schedule and weight average their searches selected, GPU-resident batching,
and the segmented outer loop of a deep-recursion solver as one reusable
function.

Everything here is deliberately generic over the model and the task: a
recipe script owns nothing but its hyperparameters, its widths and its
evaluation protocol.  :func:`train_chunk` only asks the model for the
``initial`` / ``step`` / ``n_sup`` protocol of a segmented solver.
"""

from __future__ import annotations

import contextlib
import math

import torch

CE = torch.nn.functional.cross_entropy

#: Default gradient-norm clip of every recipe.
GRAD_CLIP = 1.0


def adamw(model, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """ Fused AdamW decaying matrix weights only, as in the searches. """
    decay, rest = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and "embedding" not in name
         else rest).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": rest, "weight_decay": 0.0}], lr=lr,
        fused=next(model.parameters()).is_cuda)


def cosine_schedule(optimizer, warmup: int, total: int, floor: float = 0.05):
    """ Linear warmup then cosine decay to ``floor`` times the peak. """
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return floor + (1 - floor) * 0.5 * (
            1 + math.cos(math.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """
    An exponential moving average of the weights, folded in after every
    optimizer step. ``averaged`` swaps the averaged weights in (for
    evaluation and checkpointing) and restores the live ones after.
    """
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model) -> None:
        state = model.state_dict()
        torch._foreach_lerp_(
            [self.shadow[key] for key in self.shadow],
            [state[key] for key in self.shadow], 1.0 - self.decay)

    @contextlib.contextmanager
    def averaged(self, model):
        backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)


def to_device(split, device):
    """ The whole split as two device-resident ``long`` tensors. """
    clues = torch.as_tensor(split.puzzles, dtype=torch.long, device=device)
    targets = torch.as_tensor(
        split.solutions, dtype=torch.long, device=device) - 1
    return clues, targets


def batches(clues, targets, batch_size: int, rng):
    """ One shuffled epoch of full batches, indexed on-device. """
    order = torch.as_tensor(
        rng.permutation(clues.shape[0]), device=clues.device)
    full = order.shape[0] - order.shape[0] % batch_size
    for start in range(0, full or order.shape[0], batch_size):
        index = order[start:start + batch_size]
        yield clues[index], targets[index]


def stream(clues, targets, batch_size: int, rng):
    """ An endless stream of batches, reshuffling after every epoch. """
    while True:
        yield from batches(clues, targets, batch_size, rng)


def train_chunk(model, batch_stream, optimizer, scheduler, iterations: int,
                ema: EMA = None, grad_clip: float = GRAD_CLIP) -> float:
    """
    ``iterations`` batches of model C's segmented outer loop, returning the
    mean loss per supervised checkpoint.

    Each batch runs ``model.n_sup`` supervision steps, each with its own
    backward pass and optimizer step, detaching the carried state in
    between. With ``iterations`` set to the number of batches per epoch and
    a ``stream`` that reshuffles per epoch, one call is exactly one epoch.

    Parameters:
        model : A ``TRMSolver`` (or subclass) to train.
        batch_stream : An iterator of ``(clues, targets)`` device batches.
        optimizer : The optimizer, stepped once per supervision step.
        scheduler : The learning-rate scheduler, stepped alongside.
        ema : The weight average updated after every step, or ``None``.
        iterations : The number of batches to consume.
        grad_clip : The gradient-norm clip of every step.
    """
    model.train()
    device = next(model.parameters()).device
    total, checkpoints = torch.zeros((), device=device), 0
    for _ in range(iterations):
        clues, target = next(batch_stream)
        flat = target.reshape(-1)
        torch.compiler.cudagraph_mark_step_begin()
        state = model.initial(clues)
        for _ in range(model.n_sup):
            torch.compiler.cudagraph_mark_step_begin()
            state, logits = model.step(state)
            loss = CE(logits.reshape(-1, model.n), flat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            # clone, not just detach: under reduce-overhead the carried state
            # is a graph output whose buffer the next replay overwrites.
            state = state.detach().clone()
            total, checkpoints = total + loss.detach(), checkpoints + 1
    return (total / max(checkpoints, 1)).item()
