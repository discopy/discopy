# -*- coding: utf-8 -*-

"""
Optuna search for model C -- the TRM recursion of ``train_c_trm.py``.

    pip install optuna
    CUDA_VISIBLE_DEVICES=2 python optuna_c_trm.py --trials 40

Each trial trains one configuration from scratch and returns its **best
validation board-solve rate across epochs**, i.e. with checkpoint
selection. The space targets the diagnosed weaknesses of the recorded
configuration:

* ``ema_decay`` (optional) -- an exponential moving average of the
  weights, the TRM recipe detail whose absence matches the observed
  peak-then-drift (0.911 -> 0.885 on the n=8 variant). C takes ``N_sup``
  noisy optimizer steps per batch with shallow gradients, so it is the
  model most exposed to late-training parameter noise. When enabled,
  validation and the saved checkpoint use the averaged weights.
* ``n``, ``T``, ``n_sup`` -- the recursion shape; the n=8 probe looked
  better than the recorded n=6 but was never finished. Note the objective
  is quality only: deeper shapes cost proportionally more wall-clock per
  trial, which the pruner partially compensates for.
* ``lr`` + ``warmup_frac`` -- warmup + cosine decay over the *optimizer*
  steps (of which there are ``N_sup`` per batch), against the baseline's
  constant learning rate.
* ``weight_decay`` -- AdamW on matrix weights only, never on biases,
  norms or embeddings.

The seed is drawn at random for every trial and recorded in the trial's
user attributes, so the search ranks configurations rather than lucky
seeds -- re-validate the winners over fixed seeds. Symmetry augmentation
is deliberately absent: the map is equivariant to the positional sudoku
group by construction. The test split is never touched here; retrain the
winner (ideally on all 180k puzzles, for longer) and evaluate it once.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os

import numpy as np
import optuna
import torch

from experiments import data as datasets
from experiments import models as zoo
from experiments.config import ARTIFACTS, GRAD_CLIP
from experiments.train import evaluate, seed_everything

CE = torch.nn.functional.cross_entropy


def random_seed(trial: optuna.Trial) -> int:
    """ A fresh random seed per trial, recorded so the run is replayable. """
    seed = int.from_bytes(os.urandom(4), "little") % (2 ** 31)
    trial.set_user_attr("seed", seed)
    seed_everything(seed)
    return seed


def adamw(model, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """ AdamW decaying matrix weights only: no biases, norms, embeddings. """
    decay, rest = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 and "embedding" not in name
             else rest).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": rest, "weight_decay": 0.0}], lr=lr)


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
    An exponential moving average of the float entries of a state dict.

    ``update`` folds the live weights in after every optimizer step;
    ``averaged`` is a context manager that swaps the averaged weights in
    (for evaluation or checkpointing) and restores the live ones after.
    """
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model) -> None:
        # one fused foreach call rather than one tiny kernel per tensor;
        # measured 0.28ms -> 0.02ms per call (called n_sup times per batch,
        # so this was minor either way).
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


def batches(split, batch_size: int, rng, device):
    """ One shuffled epoch of ``(clues, target)``; targets are 0-indexed. """
    order = rng.permutation(len(split))
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        yield (torch.as_tensor(split.puzzles[index], dtype=torch.long,
                               device=device),
               torch.as_tensor(split.solutions[index], dtype=torch.long,
                               device=device) - 1)


def train_one_epoch(model, split, optimizer, scheduler, ema,
                    batch_size, rng, device) -> float:
    """ One epoch of the segmented outer loop, ``n_sup`` steps per batch. """
    model.train()
    total, checkpoints = 0.0, 0
    for clues, target in batches(split, batch_size, rng, device):
        flat = target.reshape(-1)
        state = model.initial(clues)
        for _ in range(model.n_sup):
            state, logits = model.step(state)
            loss = CE(logits.reshape(-1, model.n), flat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            state = state.detach()
            total, checkpoints = total + loss.item(), checkpoints + 1
    return total / max(checkpoints, 1)


def save_if_best(trial: optuna.Trial, board: float, state: dict,
                 study_name: str) -> None:
    """ Keep the weights of the best trial seen so far, and only those. """
    try:
        previous = trial.study.best_value
    except ValueError:
        previous = -1.0
    if state is not None and board > previous:
        torch.save(
            {"state_dict": state, "params": trial.params,
             "seed": trial.user_attrs["seed"], "valid_board": board},
            ARTIFACTS / f"optuna-{study_name}-trial{trial.number}.pt")


def objective(trial, arguments, train_split, valid_split, device) -> float:
    seed = random_seed(trial)
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    n = trial.suggest_categorical("n", [6, 8])
    cycles = trial.suggest_categorical("T", [2, 3])
    n_sup = trial.suggest_categorical("n_sup", [8, 12])
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    warmup_frac = trial.suggest_float("warmup_frac", 0.0, 0.1)
    use_ema = trial.suggest_categorical("use_ema", [True, False])
    ema_decay = trial.suggest_float(
        "ema_decay", 0.99, 0.9995) if use_ema else None

    model = zoo.TRMSolver(rounds=n, cycles=cycles, n_sup=n_sup).to(device)
    if arguments.compile:
        # every trial builds fresh modules, so recompile from a clean slate:
        # without the reset, dynamo's per-code recompile limit (8) would
        # silently fall back to eager after the first few trials.
        torch._dynamo.reset()
        model.compile_cells()
    total = math.ceil(len(train_split) / arguments.batch_size) \
        * arguments.epochs * n_sup
    optimizer = adamw(model, lr, weight_decay)
    scheduler = cosine_schedule(optimizer, int(warmup_frac * total), total)
    ema = EMA(model, ema_decay) if use_ema else None
    rng = np.random.default_rng(seed)

    best, best_state = 0.0, None
    try:
        for epoch in range(1, arguments.epochs + 1):
            loss = train_one_epoch(
                model, train_split, optimizer, scheduler, ema,
                arguments.batch_size, rng, device)
            with ema.averaged(model) if ema else contextlib.nullcontext():
                scores = evaluate(model, valid_split)
                if scores["board"] > best:
                    best, best_state = scores["board"], {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()}
                    trial.set_user_attr("best_epoch", epoch)
            print(f"  trial {trial.number} epoch {epoch}/{arguments.epochs}"
                  f"  loss {loss:.4f}  cell {scores['cell']:.4f}"
                  f"  board {scores['board']:.4f}")
            trial.report(scores["board"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        save_if_best(trial, best, best_state, arguments.study_name)
    except torch.cuda.OutOfMemoryError:
        trial.set_user_attr("oom", True)
        raise optuna.TrialPruned()
    finally:
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--n-valid", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=None,
                        help="stop starting new trials after this many s")
    parser.add_argument("--storage", default="sqlite:///optuna_c_trm.db")
    parser.add_argument("--study-name", default="c-trm")
    parser.add_argument("--compile", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="torch.compile the round step (same numerics "
                             "up to rounding error, ~2x wall-clock)")
    parser.add_argument("--device", default="cuda"
                        if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args(argv)
    device = torch.device(arguments.device)

    splits = datasets.load()
    train_split = splits["train"].subsample(arguments.n_train)
    valid_split = splits["valid"].subsample(arguments.n_valid)

    study = optuna.create_study(
        study_name=arguments.study_name, storage=arguments.storage,
        direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=2))
    study.optimize(
        lambda trial: objective(
            trial, arguments, train_split, valid_split, device),
        n_trials=arguments.trials, timeout=arguments.timeout)

    print(f"\nbest valid boards {study.best_value:.4f} "
          f"(seed {study.best_trial.user_attrs.get('seed')}, "
          f"epoch {study.best_trial.user_attrs.get('best_epoch')})")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())