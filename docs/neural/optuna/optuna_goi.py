# -*- coding: utf-8 -*-
 
"""
Optuna search for model A -- the GoI factor graph of ``train_a_goi.py``.
 
    pip install optuna
    CUDA_VISIBLE_DEVICES=0 python optuna_a_goi.py --trials 40
 
Each trial trains one configuration from scratch and returns its **best
validation board-solve rate across epochs** -- i.e. with the checkpoint
selection that ``train_a_goi.py`` lacks (its best epoch scored 0.9113
against the 0.8906 it saved). The space targets the diagnosed weaknesses
of the recorded configuration, not the architecture:
 
* ``lr`` + ``warmup_frac`` -- every trial runs warmup + cosine decay to a
  floor of 5% of the peak, against the baseline's constant 1e-3, which is
  what let it drift after its best epoch.
* ``rounds`` -- the 64-round weights kept improving at 144 test rounds, so
  trained depth is binding; the grid probes deeper training.
* ``round_weight_gamma`` -- deep supervision weighted ``(t+1)**gamma``
  (``gamma=0`` recovers the baseline's uniform average); later rounds are
  the ones that carry the answer.
* ``weight_decay`` -- AdamW on matrix weights only, never on biases,
  norms or embeddings.
 
The seed is drawn at random for every trial and recorded in the trial's
user attributes, so the search ranks configurations rather than lucky
seeds -- re-validate the top few configurations over fixed seeds before
trusting them. Symmetry augmentation is deliberately absent from the
space: the map is equivariant to the positional sudoku group by
construction, so it cannot add information. The test split is never
touched here; retrain the winner (ideally on all 180k puzzles, for
longer) and evaluate it once.
 
Trials report to the study after every epoch, so a median pruner kills
bad configurations early; the study lives in a per-model sqlite file and
resumes if the process is restarted.
"""
 
from __future__ import annotations
 
import argparse
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
 
 
def round_weights(rounds: int, gamma: float) -> list[float]:
    """ Deep-supervision weights ``(t+1)**gamma``, normalised to sum 1. """
    raw = np.arange(1, rounds + 1, dtype=np.float64) ** gamma
    return (raw / raw.sum()).tolist()
 
 
def batches(split, batch_size: int, rng, device):
    """ One shuffled epoch of ``(clues, target)``; targets are 0-indexed. """
    order = rng.permutation(len(split))
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        yield (torch.as_tensor(split.puzzles[index], dtype=torch.long,
                               device=device),
               torch.as_tensor(split.solutions[index], dtype=torch.long,
                               device=device) - 1)
 
 
def train_one_epoch(model, split, optimizer, scheduler, weights,
                    batch_size, rng, device) -> float:
    """ One epoch of weighted deep supervision, one step per batch. """
    model.train()
    total, seen = 0.0, 0
    for clues, target in batches(split, batch_size, rng, device):
        flat = target.reshape(-1)
        loss = sum(
            weight * CE(logits.reshape(-1, model.n), flat)
            for weight, logits in zip(weights, model(clues, deep=True)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total, seen = total + loss.item(), seen + 1
    return total / max(seen, 1)
 
 
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
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    rounds = trial.suggest_categorical("rounds", [48, 64, 96])
    gamma = trial.suggest_float("round_weight_gamma", 0.0, 3.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    warmup_frac = trial.suggest_float("warmup_frac", 0.0, 0.1)

    model = zoo.GoISolver(rounds=rounds).to(device)
    if arguments.compile:
        # every trial builds fresh modules, so recompile from a clean slate:
        # without the reset, dynamo's per-code recompile limit (8) would
        # silently fall back to eager after the first few trials.
        torch._dynamo.reset()
        model.compile_cells()
    total = math.ceil(len(train_split) / arguments.batch_size) \
        * arguments.epochs
    optimizer = adamw(model, lr, weight_decay)
    scheduler = cosine_schedule(optimizer, int(warmup_frac * total), total)
    weights = round_weights(rounds, gamma)
    rng = np.random.default_rng(seed)
 
    best, best_state = 0.0, None
    try:
        for epoch in range(1, arguments.epochs + 1):
            loss = train_one_epoch(
                model, train_split, optimizer, scheduler, weights,
                arguments.batch_size, rng, device)
            scores = evaluate(model, valid_split)
            print(f"  trial {trial.number} epoch {epoch}/{arguments.epochs}"
                  f"  loss {loss:.4f}  cell {scores['cell']:.4f}"
                  f"  board {scores['board']:.4f}")
            if scores["board"] > best:
                best, best_state = scores["board"], {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
                trial.set_user_attr("best_epoch", epoch)
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
    parser.add_argument("--storage", default="sqlite:///optuna_a_goi.db")
    parser.add_argument("--study-name", default="a-goi")
    parser.add_argument("--compile", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="torch.compile the round step (same numerics "
                             "up to rounding error, ~4x wall-clock)")
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