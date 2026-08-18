# -*- coding: utf-8 -*-

"""
Optuna search for model B -- the RRN peer clique of ``train_b_rrn.py``.

    pip install optuna
    CUDA_VISIBLE_DEVICES=1 python optuna_b_rrn.py --trials 40

Each trial trains one configuration from scratch and returns its **best
validation board-solve rate across epochs**, i.e. with checkpoint
selection. The space targets what the recorded configuration is missing
relative to Palm et al. (2018), not the architecture:

* ``rounds`` up to 40 -- Palm trains at 32 steps; the honest depth probe
  (1e-3 at 32 rounds) was cut short and is genuinely untested. Memory is
  roughly linear in rounds for this model (972 wide wires); a trial that
  does not fit is pruned as OOM rather than crashing the study.
* ``lr`` capped at 2e-3 + ``warmup_frac`` -- 3e-3 at 32 rounds is a known
  collapse; every trial runs warmup + cosine decay to a 5% floor.
* ``weight_decay`` -- the reference recipe regularises its weights; AdamW
  here decays matrix weights only, never biases, norms or embeddings.
* ``round_weight_gamma`` -- deep supervision weighted ``(t+1)**gamma``
  (``gamma=0`` is the baseline's uniform average over rounds).

The model itself is built at :data:`SEARCH_WIDTHS`, smaller than the
recorded ``WIDTHS["rrn"]`` in ``sudoku.config``: this search ranks
hyperparameters as cheaply as possible, it does not need to match the
final accuracy number or the cross-model parameter budget. Retrain the
winning hyperparameters at the full recorded widths before trusting the
accuracy.

``--n-train`` and ``--n-valid`` default to the full 180k/18k puzzles of
the paper's train and validation splits (not a subsample), so board-solve
rates reported here are on the same data Palm et al. used and comparable
to their numbers. Lowering either still works for a quicker smoke test,
just note the run is then on less data -- and raising ``--epochs`` or
``--n-train`` further multiplies trial cost accordingly.

The seed is drawn at random for every trial and recorded in the trial's
user attributes, so the search ranks configurations rather than lucky
seeds -- re-validate the winners over fixed seeds. Symmetry augmentation
is deliberately absent: the clique with a shared cell and no positional
features is equivariant to the positional sudoku group by construction,
so augmenting with it is a no-op. The test split is never touched here.

``--gpus`` runs one worker process per GPU on this host, sharing one
study, capped at however many are actually visible -- ``--gpus 8`` on a
3-GPU box just uses the 3; the combined trial budget is ``--trials`` times
that count.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

# so the script imports ``sudoku`` regardless of the caller's cwd or
# PYTHONPATH, e.g. when launched directly from an IDE's run button.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neural"))
from sudoku import data as datasets
from sudoku import models as zoo
from sudoku.config import ARTIFACTS, GRAD_CLIP, Widths
from sudoku.train import evaluate, seed_everything

CE = torch.nn.functional.cross_entropy

# TF32 matmuls on Ampere+/H100: near-free throughput for the float32 GEMMs
# these solvers are (no float64, no explicit half). Without it torch prints
# that the tensor cores are not enabled and every matmul runs in full fp32.
# Set at import so it applies in every worker process regardless of entry path.
torch.set_float32_matmul_precision("high")

#: Deliberately smaller than ``sudoku.config.WIDTHS["rrn"]`` (24/96/172):
#: this search ranks hyperparameters, so cheaper trials matter more than
#: matching the recorded model's size.
SEARCH_WIDTHS = Widths(dim=16, state_dim=64, hidden=128)


def available_gpu_ids() -> list[str]:
    """ Physical CUDA device ids visible to this process, as strings. """
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited:
        return [id_.strip() for id_ in inherited.split(",") if id_.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def child_argv(arguments: argparse.Namespace) -> list[str]:
    """ A CLI invocation of this script matching ``arguments``, ``--gpus 1``. """
    argv = [
        "--trials", str(arguments.trials),
        "--epochs", str(arguments.epochs),
        "--n-train", str(arguments.n_train),
        "--n-valid", str(arguments.n_valid),
        "--batch-size", str(arguments.batch_size),
        "--storage", arguments.storage,
        "--study-name", arguments.study_name,
        "--device", "cuda",
        "--gpus", "1",
        "--compile" if arguments.compile else "--no-compile"]
    if arguments.timeout is not None:
        argv += ["--timeout", str(arguments.timeout)]
    if arguments.compile_mode:
        argv += ["--compile-mode", arguments.compile_mode]
    return argv


def run_on_gpus(arguments: argparse.Namespace, ids: list[str]) -> int:
    """
    One subprocess per id in ``ids``, pinned to it via ``CUDA_VISIBLE_DEVICES``
    and sharing ``arguments.storage``/``arguments.study_name``, so they
    collaborate on one study rather than running independent searches; the
    combined trial budget is ``arguments.trials`` times ``len(ids)``. Blocks
    until every worker exits, returning the worst of their exit codes.
    """
    argv = child_argv(arguments)
    script = os.path.abspath(__file__)
    children = [
        subprocess.Popen([sys.executable, script, *argv],
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_id))
        for gpu_id in ids]
    return max(child.wait() for child in children)


def report(study: optuna.Study) -> None:
    """ Print the best trial's board-solve rate, seed and hyperparameters. """
    print(f"\nbest valid boards {study.best_value:.4f} "
          f"(seed {study.best_trial.user_attrs.get('seed')}, "
          f"epoch {study.best_trial.user_attrs.get('best_epoch')})")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


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
    # fused: the whole update is one multi-tensor kernel per step instead of
    # a foreach group -- same update rule up to rounding, fewer launches,
    # which matters at thousands of optimizer steps per epoch. CUDA only.
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


def round_weights(rounds: int, gamma: float) -> list[float]:
    """ Deep-supervision weights ``(t+1)**gamma``, normalised to sum 1. """
    raw = np.arange(1, rounds + 1, dtype=np.float64) ** gamma
    return (raw / raw.sum()).tolist()


def to_device(split, device):
    """
    The whole split resident on ``device`` as two ``long`` tensors: the clues
    and the 0-indexed targets.

    The benchmark is tiny -- 180k puzzles of 81 cells is ~117 MB per tensor as
    ``int64`` -- so it fits in VRAM with room to spare. Keeping it there makes
    a batch a single on-device gather with no host->device copy on the
    training step, which is the fastest possible feed for data this small and
    is why there is no ``DataLoader``/``num_workers`` here: a worker pool would
    only serialise these micro-batches over a pipe for no gain, so the
    effective best worker count is none.
    """
    clues = torch.as_tensor(split.puzzles, dtype=torch.long, device=device)
    targets = torch.as_tensor(
        split.solutions, dtype=torch.long, device=device) - 1
    return clues, targets


def batches(clues, targets, batch_size: int, rng):
    """
    One shuffled epoch of ``(clues, target)`` by indexing the GPU-resident
    tensors. The next batch's gather is enqueued on the CUDA stream ahead of
    the step that consumes it rather than copied from the host in between, so
    the prefetch is implicit in never leaving the device.

    The tail batch (``len % batch_size`` puzzles) is dropped so every step
    sees the same shapes: dynamo then compiles one graph per trial instead of
    a second one for the odd tail, and the loop stays CUDA-graph-friendly.
    The shuffle is fresh every epoch, so no puzzle is systematically skipped.
    """
    order = torch.as_tensor(
        rng.permutation(clues.shape[0]), device=clues.device)
    full = order.shape[0] - order.shape[0] % batch_size
    for start in range(0, full or order.shape[0], batch_size):
        index = order[start:start + batch_size]
        yield clues[index], targets[index]


def train_one_epoch(model, clues_all, targets_all, optimizer, scheduler,
                    weights, batch_size, rng) -> float:
    """
    One epoch of weighted deep supervision, one step per batch.

    The running loss is kept in a device tensor and read back once per epoch:
    a per-batch ``loss.item()`` blocks the CPU on the GPU just to fetch a
    scalar for logging, stalling between steps instead of racing ahead to
    enqueue the next batch's forward -- the idle gap ``nvidia-smi`` reports.
    """
    model.train()
    device = clues_all.device
    total, seen = torch.zeros((), device=device), 0
    for clues, target in batches(clues_all, targets_all, batch_size, rng):
        flat = target.reshape(-1)
        # a new cudagraph generation: under ``--compile-mode reduce-overhead``
        # the previous batch's graph outputs may now be reused (a cheap no-op
        # under the default mode).
        torch.compiler.cudagraph_mark_step_begin()
        loss = sum(
            weight * CE(logits.reshape(-1, model.n), flat)
            for weight, logits in zip(weights, model(clues, deep=True)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total, seen = total + loss.detach(), seen + 1
    return (total / max(seen, 1)).item()


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


def objective(trial, arguments, train_clues, train_targets,
              valid_split, device) -> float:
    seed = random_seed(trial)
    lr = trial.suggest_float("lr", 2e-4, 2e-3, log=True)
    rounds = trial.suggest_categorical("rounds", [20, 24, 32, 40])
    gamma = trial.suggest_float("round_weight_gamma", 0.0, 3.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    warmup_frac = trial.suggest_float("warmup_frac", 0.0, 0.1)

    model = zoo.RRNSolver(SEARCH_WIDTHS, rounds=rounds).to(device)
    if arguments.compile:
        # every trial builds fresh modules, so recompile from a clean slate:
        # without the reset, dynamo's per-code recompile limit (8) would
        # silently fall back to eager after the first few trials.
        torch._dynamo.reset()
        model.compile_cells(**({"mode": arguments.compile_mode}
                               if arguments.compile_mode else {}))
    # floor, matching the tail-batch drop in ``batches``, so the cosine
    # schedule finishes exactly at the last optimizer step.
    total = max(train_clues.shape[0] // arguments.batch_size, 1) \
        * arguments.epochs
    optimizer = adamw(model, lr, weight_decay)
    scheduler = cosine_schedule(optimizer, int(warmup_frac * total), total)
    weights = round_weights(rounds, gamma)
    rng = np.random.default_rng(seed)

    best, best_state = 0.0, None
    try:
        for epoch in range(1, arguments.epochs + 1):
            loss = train_one_epoch(
                model, train_clues, train_targets, optimizer, scheduler,
                weights, arguments.batch_size, rng)
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=180_000)
    parser.add_argument("--n-valid", type=int, default=18_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=None,
                        help="stop starting new trials after this many s")
    parser.add_argument("--storage", default="sqlite:///optuna_b_rrn.db")
    parser.add_argument("--study-name", default="b-rrn")
    parser.add_argument("--compile", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="torch.compile the round step (same numerics "
                             "up to rounding error, ~6x wall-clock)")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode for the round step; the "
                             "default reduce-overhead replays it as a CUDA "
                             "graph, the fix for launch-bound loops (relies "
                             "on the static shapes the tail-batch drop "
                             "guarantees, verified bit-exact vs 'default')")
    parser.add_argument("--device", default="cuda"
                        if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPUs to use in parallel, one worker process "
                             "each sharing one study; capped at however "
                             "many are actually visible")
    arguments = parser.parse_args(argv)

    if (arguments.gpus > 1 and arguments.device.startswith("cuda")
            and torch.cuda.is_available()):
        ids = available_gpu_ids()[:arguments.gpus]
        if len(ids) > 1:
            code = run_on_gpus(arguments, ids)
            report(optuna.load_study(
                study_name=arguments.study_name, storage=arguments.storage))
            return code

    device = torch.device(arguments.device)

    splits = datasets.load()
    train_split = splits["train"].subsample(arguments.n_train)
    valid_split = splits["valid"].subsample(arguments.n_valid)
    # resident on the GPU once, shared read-only across every trial's epochs.
    train_clues, train_targets = to_device(train_split, device)

    study = optuna.create_study(
        study_name=arguments.study_name, storage=arguments.storage,
        direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=2))
    study.optimize(
        lambda trial: objective(
            trial, arguments, train_clues, train_targets, valid_split, device),
        n_trials=arguments.trials, timeout=arguments.timeout)

    report(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())