# -*- coding: utf-8 -*-

"""
Optuna search for the TRM recursion on the sudoku-extreme benchmark.

    pip install optuna
    CUDA_VISIBLE_DEVICES=0 python optuna_trm_extreme.py --trials 40

The protocol of ``optuna_deep_goi.py``, moved to the harder dataset:
each trial trains one configuration from scratch on
``sudoku_extreme_special_large`` (1,001,000 examples: 1,000 base puzzles
plus 1,000 transposed, relabeled augmentations each, see
:mod:`sudoku.sudoku_extreme`) and returns its **best validation
board-solve rate across evaluations**, i.e. with checkpoint selection.

Training is measured in *iterations* (batches) rather than epochs: with
the default batch size of 512, one epoch of the training set is ~1,955
iterations, so the default ``--iterations 6000`` is about three epochs.
Validation runs every ``--eval-every 200`` iterations -- roughly every
100k examples -- so a trial prints, reports to the pruner, and can
checkpoint about 30 times. Each evaluation scores ``--n-valid 2000``
held-out puzzles as a single GPU batch at :data:`EVAL_COMPUTE` = 8, 16
and 32 supervision steps; the trial's reported value is the best of the
three, since the deployment protocol picks the inference depth anyway.
All three depths on 2,000 puzzles cost about as much as one pass at
trained depth over the full 18,000 (sampling error ~1% at board 0.3);
re-score the saved winner on the full split for the honest number.

The space is centered one notch *above* the shape that won on the Palm
benchmark (n=8, T=4, n_sup=6, valid board 0.9933): the extreme puzzles
are harder, so the recursion gets slightly more iterations to work with,
``n`` in {8, 10}, ``T`` in {4, 6} and ``n_sup`` in {8, 12}. The learning
rate range tops out at 3e-3 since the previous winner (1.5e-3) sat near
the old 2e-3 edge. The remaining dimensions -- EMA, weight decay on
matrix weights only, warmup + cosine decay over the *optimizer* steps
(``n_sup`` per iteration) -- are unchanged.

The model is built at :data:`SEARCH_WIDTHS`, three times the widths of
the Palm-benchmark searches (~1.0M parameters): the extreme puzzles need
more capacity -- reference results on this benchmark use models in the
millions of parameters -- and part of the extra math still hides under
the kernel-launch overhead that dominates at these sizes. Retrain the
winner -- wider and longer -- before trusting the accuracy, and only
then evaluate it once on the test split, which is never touched here
(``valid`` is 18k puzzles held out from the authors' train.csv, disjoint
from the base subsample).

The seed is drawn at random for every trial and recorded in the trial's
user attributes, so the search ranks configurations rather than lucky
seeds. The dataset artifacts are built on first use; with ``--gpus`` > 1
run a single worker once (or ``python -m sudoku.sudoku_extreme``)
before a multi-worker launch, so the children do not race the build.

``--gpus`` runs one worker process per GPU on this host, sharing one
study, capped at however many are actually visible; the combined trial
budget is ``--trials`` times that count.
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

# so the script imports ``sudoku`` regardless of the caller's cwd or
# PYTHONPATH, e.g. when launched directly from an IDE's run button.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neural"))
from sudoku import models as zoo
from sudoku import sudoku_extreme
from sudoku.config import ARTIFACTS, GRAD_CLIP, Widths
from sudoku.train import evaluate, seed_everything

CE = torch.nn.functional.cross_entropy

# TF32 matmuls on Ampere+/H100: near-free throughput for the float32 GEMMs
# these solvers are (no float64, no explicit half). Without it torch prints
# that the tensor cores are not enabled and every matmul runs in full fp32.
# Set at import so it applies in every worker process regardless of entry path.
torch.set_float32_matmul_precision("high")

#: Three times the widths of the Palm-benchmark searches (~1.0M parameters
#: vs 113k): the extreme puzzles need more capacity -- reference results
#: on this benchmark use models in the millions of parameters -- and part
#: of the extra math still hides under the kernel-launch overhead that
#: dominates at these sizes.
SEARCH_WIDTHS = Widths(dim=72, state_dim=192, hidden=384, y_dim=96)

#: Supervision steps the periodic evaluation sweeps; a trial is ranked and
#: checkpointed on the best of the three, so the search rewards models that
#: benefit from test-time compute -- what the final protocol will exploit.
EVAL_COMPUTE = (8, 16, 32)

#: The whole validation subset is scored as one forward-only GPU batch:
#: fewer kernel launches, and a few GB of transient memory at most.
EVAL_BATCH = 2000


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
        "--iterations", str(arguments.iterations),
        "--eval-every", str(arguments.eval_every),
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
    try:
        best_value, best_trial = study.best_value, study.best_trial
    except ValueError:
        print("\nno completed trials")
        return
    print(f"\nbest valid boards {best_value:.4f} "
          f"(seed {best_trial.user_attrs.get('seed')}, "
          f"iteration {best_trial.user_attrs.get('best_iteration')}, "
          f"n_sup at eval {best_trial.user_attrs.get('best_compute')})")
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

    Even the 1,001,000-example extreme training set is only ~650 MB per
    tensor as ``int64``, so it fits in VRAM with room to spare. Keeping it
    there makes a batch a single on-device gather with no host->device copy
    on the training step -- the fastest possible feed for data this small,
    and why there is no ``DataLoader``/``num_workers`` here.
    """
    clues = torch.as_tensor(split.puzzles, dtype=torch.long, device=device)
    targets = torch.as_tensor(
        split.solutions, dtype=torch.long, device=device) - 1
    return clues, targets


def batches(clues, targets, batch_size: int, rng):
    """
    One shuffled epoch of ``(clues, target)`` by indexing the GPU-resident
    tensors. The tail batch is dropped so every step sees the same shapes:
    dynamo compiles one graph per trial and the loop stays
    CUDA-graph-friendly; the shuffle is fresh every epoch, so no example is
    systematically skipped.
    """
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


def train_chunk(model, batch_stream, optimizer, scheduler, ema,
                iterations: int) -> float:
    """
    ``iterations`` batches of the segmented outer loop, ``n_sup`` optimizer
    steps each. The running loss lives in a device tensor and is read back
    once at the end, so the loop never blocks on the GPU mid-chunk.
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
                 study_name: str, extra: dict = None) -> None:
    """
    Keep the weights of the best trial seen so far, and only those.

    Parameters:
        extra : Additional evaluation results to store alongside the
                checkpoint, e.g. cell accuracy or adaptive-compute stats
                (merged into the saved dict; ``None`` for none).
    """
    try:
        previous = trial.study.best_value
    except ValueError:
        previous = -1.0
    if state is not None and board > previous:
        torch.save(
            {"state_dict": state, "params": trial.params,
             "seed": trial.user_attrs["seed"], "valid_board": board,
             **(extra or {})},
            ARTIFACTS / f"optuna-{study_name}-trial{trial.number}.pt")


def objective(trial, arguments, train_clues, train_targets,
              valid_split, device) -> float:
    seed = random_seed(trial)
    lr = trial.suggest_float("lr", 2e-4, 3e-3, log=True)
    n = trial.suggest_categorical("n", [8, 10])
    cycles = trial.suggest_categorical("T", [4, 6])
    n_sup = trial.suggest_categorical("n_sup", [8, 12])
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
    total = arguments.iterations * n_sup
    optimizer = adamw(model, lr, weight_decay)
    scheduler = cosine_schedule(optimizer, int(warmup_frac * total), total)
    ema = EMA(model, ema_decay) if use_ema else None
    rng = np.random.default_rng(seed)
    batch_stream = stream(
        train_clues, train_targets, arguments.batch_size, rng)

    checks = arguments.iterations // arguments.eval_every
    best, best_state = 0.0, None
    try:
        for check in range(1, checks + 1):
            loss = train_chunk(model, batch_stream, optimizer, scheduler,
                               ema, arguments.eval_every)
            with ema.averaged(model) if ema else contextlib.nullcontext():
                scores = {
                    compute: evaluate(model, valid_split, compute=compute,
                                      batch_size=EVAL_BATCH)
                    for compute in EVAL_COMPUTE}
                top = max(EVAL_COMPUTE, key=lambda c: scores[c]["board"])
                board = scores[top]["board"]
                if board > best:
                    best, best_state = board, {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()}
                    trial.set_user_attr(
                        "best_iteration", check * arguments.eval_every)
                    trial.set_user_attr("best_compute", top)
            boards = "/".join(
                f"{scores[compute]['board']:.4f}" for compute in EVAL_COMPUTE)
            print(f"  trial {trial.number} iteration "
                  f"{check * arguments.eval_every}/{arguments.iterations}"
                  f"  loss {loss:.4f}  cell {scores[top]['cell']:.4f}"
                  f"  board {boards} @n_sup "
                  + "/".join(map(str, EVAL_COMPUTE)))
            trial.report(board, check)
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
    parser.add_argument("--iterations", type=int, default=6000,
                        help="training batches per trial (~3 epochs of the "
                             "1M examples at the default batch size)")
    parser.add_argument("--eval-every", type=int, default=200,
                        help="iterations between validation evaluations, "
                             "prints and pruner reports")
    parser.add_argument("--n-train", type=int, default=1_001_000)
    parser.add_argument("--n-valid", type=int, default=2000,
                        help="validation puzzles per periodic evaluation "
                             "(a prefix of the pre-shuffled split; ~1% "
                             "sampling error at board 0.3 -- re-score the "
                             "saved winner on the full 18k)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=None,
                        help="stop starting new trials after this many s")
    parser.add_argument("--storage", default="sqlite:///optuna_trm_extreme.db")
    # -3x: each width setting gets its own study, so results from different
    # model sizes are never mixed (the original "trm-extreme" study only
    # ever held pre-guard failed trials). -tt: the objective changed to
    # best-over-EVAL_COMPUTE, which is not comparable to the single-depth
    # values of the earlier studies, so it gets a fresh study too.
    parser.add_argument("--study-name", default="trm-extreme-3x-tt")
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
                             "guarantees)")
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

    splits = sudoku_extreme.load("special_large")
    train_split = splits["train"].subsample(arguments.n_train)
    valid_split = splits["valid"].subsample(arguments.n_valid)
    # resident on the GPU once, shared read-only across every trial.
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
