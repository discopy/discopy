# -*- coding: utf-8 -*-

"""
Training the sudoku models -- an ordinary PyTorch training loop.

    python train.py goi --seed 0            # the recorded baseline
    python train.py trm --quick             # a few-minute miniature
    python train.py simple                  # the 0.9933-boards recipe
    python train.py extreme                 # the sudoku-extreme recipe
    python train.py act --steps 200000      # adaptive computation time

Nothing here is in :mod:`discopy.neural`, and that is the point: a
:class:`~model.Solver` is a :class:`torch.nn.Module`, so it trains with an
ordinary optimizer, an ordinary scheduler and an ordinary loop.  What the
file contains is the *protocol* of the study, in four layers:

* :func:`train_epoch` -- the two supervision schemes.  The single-run
  models average the cross-entropy over every round of one backward graph,
  so one batch is one optimizer step; the recursion is supervised once per
  detached segment, so one batch is ``steps`` optimizer steps.  That
  residual asymmetry is reported rather than hidden.
* :func:`train_model` -- the registry: every run is cached under
  ``artifacts/`` as weights plus history plus metadata, so re-running a
  script re-loads instead of re-training, and :func:`lr_search` is a small
  grid whose cache is keyed by its own epoch count.
* :func:`train_chunk` and the ingredients around it -- fused AdamW on
  matrix weights only, linear warmup into cosine decay, an exponential
  moving average of the weights and GPU-resident batching: what the optuna
  searches selected, which the two recorded recipes reuse verbatim.
* :class:`ACTTrainer` -- the slot-refill loop of
  :cite:t:`JolicoeurMartineau25`: every batch slot advances its own puzzle
  and is refilled with a fresh one the moment it halts, so the device
  always steps a full batch of useful work.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from dataclasses import asdict, replace

import numpy as np
import torch

import dataset
import evaluate as evaluations
import model as zoo
from config import (
    ARTIFACTS, DEFAULT_LR, EXTREME_TRM, FULL, GRAD_CLIP, LR_GRID, QUICK,
    SIMPLE_TRM, SWEEP, Budget)

CE = torch.nn.functional.cross_entropy


def seed_everything(seed: int) -> None:
    """ Fix every source of randomness we use. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def device_of(module) -> torch.device:
    """ The device a module lives on. """
    return next(module.parameters()).device


def default_device() -> torch.device:
    """ The GPU if there is one. """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- one epoch, under the model's own supervision scheme -------------------

def batches(split, batch_size: int, rng, device, augment=None):
    """
    One shuffled epoch of ``(clues, zero-indexed target)`` pairs, with the
    sudoku symmetry applied on the fly when asked for.

    Parameters:
        split : The training split.
        batch_size : The number of puzzles per batch.
        rng : The ``numpy`` generator behind the shuffle.
        device : The device the batches land on.
        augment : ``(puzzles, solutions, rng) -> (puzzles, solutions)``.
    """
    order = rng.permutation(len(split))
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        puzzles, solutions = split.puzzles[index], split.solutions[index]
        if augment is not None:
            puzzles, solutions = augment(puzzles, solutions, rng)
        yield (torch.as_tensor(puzzles, dtype=torch.long, device=device),
               torch.as_tensor(solutions, dtype=torch.long, device=device) - 1)


def train_epoch(model, split, budget: Budget, optimizer, rng, device,
                augment=None) -> dict:
    """
    One epoch under the model's own deep supervision.

    Returns the mean loss *per supervised checkpoint* for both schemes, so
    that the two losses are on the same scale, together with how many
    optimizer steps and checkpoints it took.
    """
    model.train()
    total, checkpoints, steps = 0.0, 0, 0
    for clues, target in batches(
            split, budget.batch_size, rng, device, augment):
        flat = target.reshape(-1)
        if model.segmented:
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
            losses = [CE(logits.reshape(-1, model.n), flat)
                      for logits in model(clues, deep=True)]
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


# --- the registry ----------------------------------------------------------

def checkpoint_path(name: str, budget: Budget, seed: int):
    """ Where a run's artifacts live. """
    return ARTIFACTS / f"{budget.name}-{name}-seed{seed}.pt"


def train_model(name: str, budget: Budget, seed: int, train_split,
                valid_split, lr: float = None, device=None,
                resume: bool = True, log=print):
    """
    Train one model with one seed, or load it back when it is already
    cached.

    Parameters:
        name : ``"goi"``, ``"rrn"``, ``"trm"`` or ``"act"``.
        budget : The budget giving data size, epochs, depth and batch size.
        seed : The seed, fixed *before* the model is built so that the
               initialisation is reproducible too.
        train_split : The training puzzles, already subsampled.
        valid_split : The split evaluated after every epoch.
        lr : The learning rate, from :data:`config.DEFAULT_LR` by default.
        resume : Whether to load a cached checkpoint if one exists.
        log : Where to print progress.

    Returns:
        The model, its per-epoch history and its metadata.
    """
    device = device or default_device()
    lr = DEFAULT_LR[name] if lr is None else lr
    path = checkpoint_path(name, budget, seed)

    seed_everything(seed)
    model = zoo.build(name, budget).to(device)
    if resume and path.exists():
        stored = zoo.load_checkpoint(model, path)
        return model, stored["history"], stored["meta"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    augment = dataset.augment if budget.augment else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, start = [], time.perf_counter()
    for epoch in range(budget.epochs):
        tick = time.perf_counter()
        stats = train_epoch(model, train_split, budget, optimizer, rng,
                            device, augment=augment)
        scores = evaluations.evaluate(model, valid_split)
        history.append({"epoch": epoch + 1,
                        "seconds": time.perf_counter() - tick,
                        **stats, **scores})
        log(f"  {name} seed {seed} epoch {epoch + 1:2d}/{budget.epochs}  "
            f"loss {stats['loss']:.4f}  cell {scores['cell']:.4f}  "
            f"board {scores['board']:.4f}  ({history[-1]['seconds']:.0f}s)")
    meta = {
        "model": name, "seed": seed, "lr": lr, "budget": asdict(budget),
        "parameters": zoo.count_parameters(model),
        "wires": model.n_wires, "cells": model.n_cells,
        "seconds": time.perf_counter() - start,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2 ** 20
            if device.type == "cuda" else float("nan"))}
    torch.save({"state_dict": model.state_dict(), "history": history,
                "meta": meta}, path)
    return model, history, meta


def lr_search(name: str, budget: Budget, train_split, valid_split,
              grid=LR_GRID, device=None, log=print) -> list:
    """
    A small learning-rate grid on the validation split, cached to disk.

    One seed, a reduced number of epochs and a reduced training subsample:
    enough to rule out a badly scaled learning rate without spending the
    experiment budget on tuning.
    """
    path = ARTIFACTS / f"{budget.name}-lr-{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    # the epochs go in the name: a cached search must never be reused after
    # `lr_epochs` changes, or a 1-epoch result would be read back as if it
    # had been trained for the new number of epochs.
    small = replace(budget, epochs=budget.lr_epochs,
                    name=f"{budget.name}-lr{budget.lr_epochs}e-{name}")
    subsample = train_split.subsample(budget.lr_n_train)
    rows = []
    for lr in grid:
        _, history, _ = train_model(
            name, replace(small, name=f"{small.name}-{lr:g}"), 0,
            subsample, valid_split, lr=lr, device=device,
            log=lambda *a: None)
        rows.append({"model": name, "lr": lr, **{
            key: history[-1][key] for key in ("cell", "board", "loss")}})
        log(f"  {name} lr {lr:g}: cell {rows[-1]['cell']:.4f}  "
            f"board {rows[-1]['board']:.4f}")
    path.write_text(json.dumps(rows, indent=2))
    return rows


# --- the ingredients the searched recipes share ----------------------------

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
    optimizer step.  :meth:`averaged` swaps the averaged weights in -- for
    evaluation and checkpointing -- and restores the live ones after.
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
        backup = {key: value.detach().clone()
                  for key, value in model.state_dict().items()
                  if key in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)


def to_device(split, device):
    """ A whole split as two device-resident ``long`` tensors. """
    clues = torch.as_tensor(split.puzzles, dtype=torch.long, device=device)
    targets = torch.as_tensor(
        split.solutions, dtype=torch.long, device=device) - 1
    return clues, targets


def stream(clues, targets, batch_size: int, rng):
    """ An endless stream of full batches, reshuffling after every epoch. """
    while True:
        order = torch.as_tensor(
            rng.permutation(clues.shape[0]), device=clues.device)
        full = order.shape[0] - order.shape[0] % batch_size
        for start in range(0, full or order.shape[0], batch_size):
            index = order[start:start + batch_size]
            yield clues[index], targets[index]


def train_chunk(model, batch_stream, optimizer, scheduler, iterations: int,
                ema: EMA = None, grad_clip: float = GRAD_CLIP) -> float:
    """
    ``iterations`` batches of the segmented outer loop, returning the mean
    loss per supervised checkpoint.

    Each batch runs ``model.n_sup`` supervision steps, each with its own
    backward pass and optimizer step, detaching the carried state in
    between.
    """
    model.train()
    device = device_of(model)
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
            # clone, not just detach: under reduce-overhead the carried
            # state is a graph output whose buffer the next replay reuses.
            state = state.detach().clone()
            total, checkpoints = total + loss.detach(), checkpoints + 1
    return (total / max(checkpoints, 1)).item()


# --- adaptive computation time ---------------------------------------------

class ExampleStream:
    """
    A device-resident stream of training examples for the slot refill.

    Holds the whole training set on the device and a shuffled order;
    :meth:`take` maps a halt mask to the next examples without leaving the
    device, by ranking the halted slots with a cumulative sum.  The order
    is reshuffled -- from the ``numpy`` generator, so runs are reproducible
    -- whenever the cursor has run past the end.
    """
    def __init__(self, inputs, targets, rng):
        assert len(inputs) == len(targets), "inputs and targets disagree"
        self.inputs, self.targets, self.rng = inputs, targets, rng
        self.consumed_before = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = torch.as_tensor(
            self.rng.permutation(len(self.inputs)),
            device=self.inputs.device)
        self.cursor = torch.zeros(
            (), dtype=torch.long, device=self.inputs.device)

    def total_consumed(self) -> int:
        """
        Total examples drawn so far, across reshuffles: an absolute count
        comparable across trials regardless of their mean halting depth,
        unlike a count of optimizer steps.
        """
        return self.consumed_before + int(self.cursor)

    def reshuffle_if_spent(self) -> None:
        """ Reshuffle once the cursor has passed the end of the order. """
        if int(self.cursor) >= len(self.order):
            self.consumed_before += len(self.order)
            self._reshuffle()

    def take(self, halted):
        """
        The next examples, one per *halted* slot: slot ``i`` receives the
        ``rank(i)``-th example after the cursor, where ``rank`` counts
        halted slots up to ``i``.  Entries of non-halted slots are
        arbitrary and meant to be discarded by the caller's ``where``.
        """
        ranks = torch.cumsum(halted.long(), 0) - 1
        index = self.order[(self.cursor + ranks) % len(self.order)]
        self.cursor = self.cursor + halted.sum()
        return self.inputs[index], self.targets[index]


class ACTTrainer:
    """
    The slot-refill training loop: a persistent batch of carries, one
    optimizer step per iteration, the loss of the paper's pseudocode.

    The paper's per-example ``break`` is realised as **slot refill**: every
    batch slot advances its own problem, and the moment a slot halts -- its
    halt logit turns positive, or it hits the cap -- it is refilled with a
    fresh problem *within the same batch*.  The device always steps a full
    batch of useful work, and the saving appears as examples per second: at
    a mean halting depth of ``d``, one optimizer step consumes
    ``batch_size / d`` problems instead of ``batch_size / steps``.  The
    refill is computed entirely on the device, so the loop stays free of
    host round-trips.

    Parameters:
        model : The ACT model to train.
        stream : The :class:`ExampleStream` feeding the refill.
        batch_size : The number of slots.
        grad_clip : The gradient-norm clip of every optimizer step.
        halt_weight : The weight of the halt loss in the total.
        halt_threshold : The margin the halt logit must clear to stop a
                         slot, ``0`` for the paper's ``q > 0``.
    """
    def __init__(self, model, stream: ExampleStream, batch_size: int,
                 grad_clip: float = GRAD_CLIP, halt_weight: float = 1.0,
                 halt_threshold: float = 0.0):
        assert model.map.solver.cycles > 1, \
            "one cycle would differentiate the refill"
        self.model, self.stream = model, stream
        self.grad_clip = grad_clip
        self.halt_weight, self.halt_threshold = halt_weight, halt_threshold
        device = stream.inputs.device
        halted = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.inputs, self.targets = stream.take(halted)
        with torch.no_grad():
            self.state = model.initial(self.inputs)
        self.steps = torch.zeros(batch_size, dtype=torch.long, device=device)

    def _run_raw(self, optimizer, scheduler, ema, iterations: int) -> dict:
        """
        The core loop of ``iterations`` optimizer steps, returning the raw
        accumulated sums.  Every statistic accumulates in a device tensor
        and is read back once at the end, so the loop never blocks.
        """
        model, stream = self.model, self.stream
        model.train()
        device = stream.inputs.device
        zeros = torch.zeros((), device=device)
        totals = {key: zeros.clone() for key in (
            "loss", "ce", "q", "halted", "solved", "depth", "capped")}
        step_zero = torch.zeros_like(self.steps)

        for _ in range(iterations):
            torch.compiler.cudagraph_mark_step_begin()
            state, logits, halt = model.act_step(self.state)
            ce_loss = CE(logits.reshape(-1, model.n),
                         self.targets.reshape(-1))
            with torch.no_grad():
                correct = (logits.argmax(-1) == self.targets).all(-1)
            q_loss = model.halt_loss(halt, logits, self.targets)
            loss = ce_loss + self.halt_weight * q_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            with torch.no_grad():
                steps = self.steps + 1
                wants_halt = model.halt_logit(halt.detach()) \
                    > self.halt_threshold
                capped = steps >= model.n_sup
                halted = wants_halt | capped

                totals["loss"] += loss.detach()
                totals["ce"] += ce_loss.detach()
                totals["q"] += q_loss.detach()
                totals["halted"] += halted.sum()
                totals["solved"] += (halted & correct).sum()
                totals["depth"] += (steps * halted).sum()
                totals["capped"] += (capped & ~wants_halt).sum()

                new_inputs, new_targets = stream.take(halted)
                mask = halted.unsqueeze(-1)
                self.inputs = torch.where(mask, new_inputs, self.inputs)
                self.targets = torch.where(mask, new_targets, self.targets)
                fresh = model.initial(self.inputs)
                # clone, not just detach: under reduce-overhead the state
                # is a graph output whose buffer the next replay reuses.
                self.state = torch.where(mask, fresh, state.detach().clone())
                self.steps = torch.where(halted, step_zero, steps)

        return {key: value.item() for key, value in totals.items()}

    @staticmethod
    def _normalize(raw: dict, iterations: int) -> dict:
        """ Raw sums over ``iterations`` steps, as means. """
        halted = max(raw["halted"], 1.0)
        return {"loss": raw["loss"] / iterations, "ce": raw["ce"] / iterations,
                "q": raw["q"] / iterations, "halted": int(raw["halted"]),
                "solved": raw["solved"] / halted,
                "depth": raw["depth"] / halted,
                "capped": raw["capped"] / halted}

    def run(self, optimizer, scheduler, ema, iterations: int) -> dict:
        """
        ``iterations`` optimizer steps of ACT training.

        Returns:
            ``loss`` (mean total), ``ce`` and ``q`` (its two terms),
            ``halted`` (examples finished), ``solved`` (of which whose
            final answer was correct), ``depth`` (their mean number of
            supervision steps) and ``capped`` (the fraction that hit the
            cap rather than halting by ``q > 0``).
        """
        return self._normalize(
            self._run_raw(optimizer, scheduler, ema, iterations), iterations)

    def run_until(self, optimizer, scheduler, ema, target_consumed: int,
                  check_every: int = 50) -> dict:
        """
        Optimizer steps until the stream's cumulative consumed count
        reaches ``target_consumed``, the natural budget for ACT: the same
        number of examples spans a different number of optimizer steps
        depending on the model's mean halting depth, so pinning the budget
        to steps would make trials with different depths see different
        amounts of data.
        """
        pooled = {key: 0.0 for key in (
            "loss", "ce", "q", "halted", "solved", "depth", "capped")}
        total_iterations = 0
        while self.stream.total_consumed() < target_consumed:
            raw = self._run_raw(optimizer, scheduler, ema, check_every)
            self.stream.reshuffle_if_spent()
            for key in pooled:
                pooled[key] += raw[key]
            total_iterations += check_every
        result = self._normalize(pooled, max(total_iterations, 1))
        result["iterations"] = total_iterations
        result["consumed"] = self.stream.total_consumed()
        return result


# --- the recorded runs -----------------------------------------------------

def baseline(name: str, seed: int = 0, quick: bool = False) -> int:
    """
    One of the three matched-budget baselines of ``README.md``, then the
    single test evaluation and a sweep of test-time compute.
    """
    from config import BEST_ROUNDS
    budget = replace(QUICK, name="quick-best") if quick else replace(
        FULL, name="best", rounds=BEST_ROUNDS[name])
    splits = dataset.load()
    print(f"model {name}, seed {seed}, budget {budget.name}")
    model, _, meta = train_model(
        name, budget, seed, splits["train"].subsample(budget.n_train),
        splits["valid"].subsample(budget.n_valid))
    print(f"  {meta['parameters']:,} parameters, {meta['wires']} wires")

    test = splits["test"].subsample(budget.n_test)
    scores = evaluations.evaluate(model, test)
    print(f"\ntest ({len(test):,} puzzles): cell {scores['cell']:.4f}  "
          f"board {scores['board']:.4f}")
    for row in evaluations.sweep_compute(model, test, SWEEP[name]):
        print(f"  at {row['compute']:4d} "
              f"{'supervision steps' if model.segmented else 'rounds'}: "
              f"board {row['board']:.4f}")
    return 0


def recipe(hyperparameters: dict, extreme: bool, iterations: int = None,
           compile_rounds: bool = True) -> int:
    """
    One of the two searched recipes, verbatim: the winning trial's
    hyperparameters, fresh initialisation and data order, best-so-far
    checkpointing and one final test evaluation.
    """
    hp = hyperparameters
    print("hyperparameters:")
    for key, value in hp.items():
        print(f"  {key}: {value}")
    torch.set_float32_matmul_precision("high")  # TF32, as the search ran
    device = default_device()
    splits = dataset.load_extreme(hp["variant"]) if extreme else dataset.load()
    clues, targets = to_device(
        splits["train"].subsample(hp["n_train"]), device)
    valid = splits["valid"].subsample(hp["n_valid"])

    model = zoo.trm(hp["widths"], rounds=hp["rounds"], cycles=hp["cycles"],
                    steps=hp["steps"]).to(device)
    print(f"parameters: {zoo.count_parameters(model):,}")
    if compile_rounds and device.type == "cuda":
        model.map.compile_rounds(mode="reduce-overhead")

    checkpoint = ARTIFACTS / hp["checkpoint"]
    ema = EMA(model, hp["ema_decay"]) if "ema_decay" in hp else None
    if extreme:
        chunks = (iterations or hp["iterations"]) // hp["eval_every"]
        per_chunk, total = hp["eval_every"], chunks * hp["eval_every"]
        compute = hp["eval_compute"]
    else:
        chunks = iterations or hp["epochs"]
        per_chunk = len(clues) // hp["batch_size"]
        total, compute = chunks * per_chunk, (None, )
    optimizer = adamw(model, hp["lr"], hp["weight_decay"])
    scheduler = cosine_schedule(
        optimizer, int(hp["warmup_frac"] * total * hp["steps"]),
        total * hp["steps"])
    batch_stream = stream(clues, targets, hp["batch_size"],
                          np.random.default_rng())

    best = -1.0  # below any board rate, so the first check always saves
    for chunk in range(1, chunks + 1):
        tick = time.perf_counter()
        loss = train_chunk(model, batch_stream, optimizer, scheduler,
                           per_chunk, ema=ema)
        with (ema.averaged(model) if ema is not None
              else contextlib.nullcontext()):
            scores = {c: evaluations.evaluate(
                model, valid, compute=c, batch_size=max(len(valid), 1))
                for c in compute}
            top = max(compute, key=lambda c: scores[c]["board"])
            if scores[top]["board"] > best:
                best = scores[top]["board"]
                torch.save({"state_dict": model.state_dict(),
                            "hyperparameters": {
                                key: value for key, value in hp.items()
                                if key != "widths"},
                            "widths": hp["widths"].asdict(),
                            "chunk": chunk, "compute": top,
                            "valid_board": best}, checkpoint)
        print(f"chunk {chunk:3d}/{chunks}  loss {loss:.4f}  "
              f"cell {scores[top]['cell']:.4f}  board "
              + "/".join(f"{scores[c]['board']:.4f}" for c in compute)
              + f"  ({time.perf_counter() - tick:.0f}s)", flush=True)

    zoo.load_checkpoint(model, checkpoint)
    model.to(device)
    if "n_test" in hp:
        scores = evaluations.evaluate(
            model, splits["test"].subsample(hp["n_test"]))
        print(f"\ntest  cell {scores['cell']:.4f}  "
              f"board {scores['board']:.4f}")
    print(f"best valid board {best:.4f}  (checkpoint {checkpoint.name})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", choices=("goi", "rrn", "trm", "simple", "extreme"),
        help="a matched-budget baseline or one of the searched recipes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="the few-minute miniature budget")
    parser.add_argument("--iterations", type=int, default=None,
                        help="epochs (simple) or iterations (extreme)")
    parser.add_argument("--no-compile", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.run in ("simple", "extreme"):
        return recipe(
            SIMPLE_TRM if arguments.run == "simple" else EXTREME_TRM,
            extreme=arguments.run == "extreme",
            iterations=arguments.iterations,
            compile_rounds=not arguments.no_compile)
    return baseline(arguments.run, arguments.seed, arguments.quick)


if __name__ == "__main__":
    raise SystemExit(main())
