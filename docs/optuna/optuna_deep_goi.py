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

The model itself is built at :data:`SEARCH_WIDTHS`, smaller than the
recorded ``WIDTHS["trm"]`` in ``experiments.config``: this search ranks
hyperparameters as cheaply as possible, it does not need to match the
final accuracy number or the cross-model parameter budget. Retrain the
winning hyperparameters at the full recorded widths before trusting the
accuracy.

``--n-train`` and ``--n-valid`` default to the full 180k/18k puzzles of
the paper's train and validation splits (not a subsample), so board-solve
rates reported here are on the same data Palm et al. used and comparable
to their numbers. Lowering either still works for a quicker smoke test,
just note the run is then on less data.

The seed is drawn at random for every trial and recorded in the trial's
user attributes, so the search ranks configurations rather than lucky
seeds -- re-validate the winners over fixed seeds. Symmetry augmentation
is deliberately absent: the map is equivariant to the positional sudoku
group by construction. The test split is never touched here; retrain the
winner at the full recorded widths, for longer, and evaluate it once.

``--gpus`` runs one worker process per GPU on this host, sharing one
study, capped at however many are actually visible -- ``--gpus 8`` on a
3-GPU box just uses the 3; the combined trial budget is ``--trials`` times
that count.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

# so the script imports ``experiments`` regardless of the caller's cwd or
# PYTHONPATH, e.g. when launched directly from an IDE's run button.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neural.experiments import data as datasets
from neural.experiments import models as zoo
from neural.experiments.config import ARTIFACTS, GRAD_CLIP, Widths
from neural.experiments.train import evaluate, seed_everything

CE = torch.nn.functional.cross_entropy

# TF32 matmuls on Ampere+/H100: near-free throughput for the float32 GEMMs
# these solvers are (no float64, no explicit half). Without it torch prints
# that the tensor cores are not enabled and every matmul runs in full fp32.
# Set at import so it applies in every worker process regardless of entry path.
torch.set_float32_matmul_precision("high")

#: Deliberately smaller than ``experiments.config.WIDTHS["trm"]``
#: (24/88/172/48): this search ranks hyperparameters, so cheaper trials
#: matter more than matching the recorded model's size.
SEARCH_WIDTHS = Widths(dim=24, state_dim=64, hidden=128, y_dim=32)

#: The best configuration of the first-round study ("c-trm" trial 1, valid
#: board 0.9917 at :data:`SEARCH_WIDTHS`), enqueued as the first trial of a
#: fresh study so the narrowed space is always compared against the recorded
#: winner under identical conditions (same code, same data, fresh seed).
BASELINE = {
    "lr": 7.5846175213136e-4, "n": 8, "T": 4, "n_sup": 8, "use_ema": False,
    "warmup_frac": 0.07154900475109112,
    "weight_decay": 1.3603987719515442e-4}


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


def train_one_epoch(model, clues_all, targets_all, optimizer, scheduler, ema,
                    batch_size, rng) -> float:
    """
    One epoch of the segmented outer loop, ``n_sup`` steps per batch.

    The running loss is accumulated in a device tensor and read back once at
    the end of the epoch. With this model the ``n_sup`` optimizer steps per
    batch are individually cheap, so a per-step ``loss.item()`` -- which blocks
    the CPU until the GPU drains just to fetch a scalar for logging -- was the
    dominant idle gap: the CPU stalling between kernels instead of racing ahead
    to enqueue the next step's work.
    """
    model.train()
    device = clues_all.device
    total, checkpoints = torch.zeros((), device=device), 0
    for clues, target in batches(clues_all, targets_all, batch_size, rng):
        flat = target.reshape(-1)
        # a new cudagraph generation: under ``--compile-mode reduce-overhead``
        # the previous step's graph outputs may now be reused (a cheap no-op
        # under the default mode).
        torch.compiler.cudagraph_mark_step_begin()
        state = model.initial(clues)
        for _ in range(model.n_sup):
            torch.compiler.cudagraph_mark_step_begin()
            state, logits = model.step(state)
            loss = CE(logits.reshape(-1, model.n), flat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            # clone, not just detach: under reduce-overhead the carried state
            # is a graph output whose buffer the next replay overwrites.
            state = state.detach().clone()
            total, checkpoints = total + loss.detach(), checkpoints + 1
    return (total / max(checkpoints, 1)).item()


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
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    n = trial.suggest_categorical("n", [6, 8])
    # 4 is the baseline's cycle count and must stay admissible for the
    # enqueued BASELINE trial; 3 probes one cycle cheaper.
    cycles = trial.suggest_categorical("T", [3, 4])
    n_sup = trial.suggest_categorical("n_sup", [6, 8])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    warmup_frac = trial.suggest_float("warmup_frac", 0.0, 0.1)
    use_ema = trial.suggest_categorical("use_ema", [True, False])
    ema_decay = trial.suggest_float(
        "ema_decay", 0.99, 0.9995) if use_ema else None

    model = zoo.TRMSolver(
        SEARCH_WIDTHS, rounds=n, cycles=cycles, n_sup=n_sup).to(device)
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
        * arguments.epochs * n_sup
    optimizer = adamw(model, lr, weight_decay)
    scheduler = cosine_schedule(optimizer, int(warmup_frac * total), total)
    ema = EMA(model, ema_decay) if use_ema else None
    rng = np.random.default_rng(seed)

    best, best_state = 0.0, None
    try:
        for epoch in range(1, arguments.epochs + 1):
            loss = train_one_epoch(
                model, train_clues, train_targets, optimizer, scheduler, ema,
                arguments.batch_size, rng)
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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=180_000)
    parser.add_argument("--n-valid", type=int, default=18_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=None,
                        help="stop starting new trials after this many s")
    parser.add_argument("--storage", default="sqlite:///optuna_c_trm.db")
    # v2: the shape choices were narrowed (n=8, T=3, n_sup in {6, 8}) after
    # the first round; optuna forbids changing a categorical's choice set
    # within a study, so the narrowed space gets a fresh study in the same
    # sqlite file. The first round's trials stay readable under "c-trm".
    parser.add_argument("--study-name", default="c-trm-v2")
    parser.add_argument("--compile", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="torch.compile the round step (same numerics "
                             "up to rounding error, ~2x wall-clock)")
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
    # the first-round winner runs first, as the in-study baseline;
    # skip_if_exists stops restarts and sibling workers from re-queueing it
    # once it is in the study (simultaneous first launches can still race,
    # in which case the baseline just runs twice -- harmless).
    study.enqueue_trial(BASELINE, skip_if_exists=True)
    study.optimize(
        lambda trial: objective(
            trial, arguments, train_clues, train_targets, valid_split, device),
        n_trials=arguments.trials, timeout=arguments.timeout)

    report(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())