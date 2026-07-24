# -*- coding: utf-8 -*-

"""
Optuna search for the TRM recursion on sudoku-extreme, with the adaptive
computation time (ACT) of the tiny recursive model paper.

    pip install optuna
    CUDA_VISIBLE_DEVICES=0 python optuna_trm_extreme_act.py --trials 40

The protocol of ``optuna_trm_extreme.py`` -- same dataset
(``sudoku_extreme_special_large``), same search space, same widths
(:data:`optuna_trm_extreme.SEARCH_WIDTHS`, ~1.0M parameters), same kind
of objective (best validation board-solve rate, with checkpoint
selection) -- with two changes: training runs the paper's deep
supervision with early stopping,

    loss  = softmax_cross_entropy(y_hat, y_true)
    loss += binary_cross_entropy(q_hat, (y_hat == y_true))
    ...
    if q_hat > 0: break     # early-stopping

where ``q_hat = Q_head(y)`` is the halt signal read off the current
answer, and the fixed-compute sweep is scored at a single depth,
:data:`EVAL_COMPUTE` = ``(16,)``, rather than the plain script's
``(8, 16, 32)`` -- one evaluation instead of three, since the adaptive
protocol below is the point of this script, and the fixed-compute
number is now just a comparability anchor to the plain search.

Two departures from the paper's letter, both in the direction of
protecting accuracy: the head is **detached** (its loss trains the head
alone, so the trunk's gradients are identical to the plain loop's --
the coupled head of trials 0-7 cost ~4 board points), and it is the
**soft-minimum per-cell head** of :class:`sudoku.act.ACTSolver`
rather than a pooled scalar, which halts on the *least* confident cell
and is conservative by construction; ``--halt-threshold`` adds margin
on top.  The per-example ``break`` is realised as the slot refill of
:class:`sudoku.act.ACTTrainer`: a puzzle leaves its batch slot the
moment its halt logit clears the threshold (or at the ``n_sup`` cap)
and a fresh puzzle takes its place, so the GPU always steps a full
batch.
The point is throughput: once the model solves most boards in a couple
of supervision steps, one optimizer step consumes ``batch / depth``
examples instead of ``batch / n_sup`` -- the same wall-clock buys
``n_sup / depth`` times more data.

**The budget and the evaluation cadence are both counted in puzzles
consumed**, not optimizer steps: since one optimizer step consumes a
different number of fresh puzzles depending on the model's mean
halting depth, pinning either to a step count would make trials with
different depths train on, and get evaluated over, different amounts of
data. A trial trains for ``--epochs`` (default 8) full passes over the
``--n-train`` puzzles -- i.e. until :meth:`PuzzleStream.total_consumed`
reaches ``epochs * n_train`` -- and is evaluated every ``--eval-every``
puzzles consumed (default 200,000), regardless of how many optimizer
steps -- and so how many gradient updates -- that took. See
:meth:`sudoku.act.ACTTrainer.run_until`. ``n_sup`` is the halting
*cap*; the learning-rate schedule spans the *worst-case* number of
optimizer steps -- as if no puzzle ever halted early, i.e.
``ceil(epochs * n_train / batch_size) * n_sup``, the exact analogue of
the plain script's ``iterations * n_sup`` -- so a trial whose model
learns to halt early simply finishes before the schedule fully decays,
a known rough edge of budgeting by data rather than by steps.

Each evaluation reports the fixed-compute sweep (the objective) and
additionally the adaptive protocol -- early stopping at inference,
capped at :data:`EVAL_COMPUTE`'s single depth -- with its mean depth,
stored in the trial's user attributes. The training prints also show
the mean halting depth and the examples consumed, which is where the
speedup is visible. The best checkpoint's validation scores (cell
accuracy, adaptive board accuracy and depth, mean training halting
depth) are saved alongside its weights, not just the scalar board score
used to rank it.

Everything else -- TF32, fused AdamW on matrix weights, EMA, warmup +
cosine decay, per-trial random seeds, the pruner, ``--gpus`` workers
sharing one study -- is inherited from ``optuna_trm_extreme.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch

# the sibling script carries the shared protocol (paths, EMA, AdamW,
# schedules, GPU workers) and sets TF32 and ``sys.path`` at import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import optuna_trm_extreme as base  # noqa: E402

from sudoku import sudoku_extreme  # noqa: E402
from sudoku.act import (  # noqa: E402
    ACTSolver, ACTTrainer, PuzzleStream, evaluate_act)
from sudoku.train import evaluate  # noqa: E402

#: The fixed-compute sweep of the objective and the adaptive cap: a
#: single depth, since the adaptive protocol is the point of this script
#: and three-depth sweeps would triple evaluation cost for little signal.
EVAL_COMPUTE = (16,)

#: The best configuration of the fixed-compute round ("trm-extreme-3x"
#: trial 5: valid board 0.4632 at its trained depth, 0.4941 at 128 steps),
#: enqueued as the first trial so the study always measures ACT against
#: the identical recipe -- same shape and optimizer settings, with
#: ``n_sup=12`` now acting as the halting cap instead of a fixed depth.
BASELINE = {
    "lr": 8.97846346433278e-4, "n": 10, "T": 4, "n_sup": 12,
    "use_ema": False, "warmup_frac": 0.060740911088609954,
    "weight_decay": 6.271288420139338e-5}


def child_argv(arguments: argparse.Namespace) -> list[str]:
    """ A CLI invocation of this script matching ``arguments``, one GPU. """
    argv = [
        "--trials", str(arguments.trials),
        "--epochs", str(arguments.epochs),
        "--eval-every", str(arguments.eval_every),
        "--check-every", str(arguments.check_every),
        "--n-train", str(arguments.n_train),
        "--n-valid", str(arguments.n_valid),
        "--batch-size", str(arguments.batch_size),
        "--halt-threshold", str(arguments.halt_threshold),
        "--storage", arguments.storage,
        "--study-name", arguments.study_name,
        "--device", "cuda",
        "--gpus", "1",
        "--compile" if arguments.compile else "--no-compile",
        "--unroll" if arguments.unroll else "--no-unroll"]
    if arguments.timeout is not None:
        argv += ["--timeout", str(arguments.timeout)]
    if arguments.compile_mode:
        argv += ["--compile-mode", arguments.compile_mode]
    return argv


def run_on_gpus(arguments: argparse.Namespace, ids: list[str]) -> int:
    """ One worker per GPU id, sharing one study; worst exit code. """
    argv = child_argv(arguments)
    script = os.path.abspath(__file__)
    children = [
        subprocess.Popen([sys.executable, script, *argv],
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_id))
        for gpu_id in ids]
    return max(child.wait() for child in children)


def objective(trial, arguments, train_clues, train_targets,
              valid_split, device) -> float:
    seed = base.random_seed(trial)
    lr = trial.suggest_float("lr", 2e-4, 3e-3, log=True)
    n = trial.suggest_categorical("n", [8,10])
    cycles = trial.suggest_categorical("T", [4, 6])
    n_sup = trial.suggest_categorical("n_sup", [12, 16])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    warmup_frac = trial.suggest_float("warmup_frac", 0.0, 0.1)
    use_ema = trial.suggest_categorical("use_ema", [True, False])
    ema_decay = trial.suggest_float(
        "ema_decay", 0.99, 0.9995) if use_ema else None

    # the halt head is *detached*: early trials (numbers 0-7, mean-pooled
    # head, coupled at weight 1.0) showed the board-level BCE dragging the
    # trunk ~4 board points at 1/3 budget while halting was still inert.
    # Detached, the trunk's gradients are identical to the plain loop's,
    # so the halt loss cannot cost accuracy at any weight -- which also
    # retires the halt_weight dimension those early trials motivated.
    # The soft-minimum per-cell head replaces the mean pool: a mean
    # dilutes one wrong cell by 1/81, which is why the early heads only
    # halted around depth 23 of 32 at evaluation; the soft minimum halts
    # on the *least* confident cell, conservatively (see ACTSolver).
    model = ACTSolver(base.SEARCH_WIDTHS, rounds=n, cycles=cycles,
                      n_sup=n_sup, halt_detach=True,
                      halt_head="softmin").to(device)
    if arguments.compile:
        # fresh modules per trial: recompile from a clean slate, or the
        # recompile limit would silently fall back to eager.
        torch._dynamo.reset()
        model.compile_cells(unroll=arguments.unroll,
                            **({"mode": arguments.compile_mode}
                               if arguments.compile_mode else {}))
    total_puzzles = arguments.epochs * arguments.n_train
    # worst-case optimizer-step count for the LR schedule, as if no puzzle
    # ever halted early -- the exact analogue of the plain script's
    # `iterations * n_sup`, now derived from the epoch/puzzle budget.
    schedule_steps = math.ceil(total_puzzles / arguments.batch_size) * n_sup
    optimizer = base.adamw(model, lr, weight_decay)
    scheduler = base.cosine_schedule(
        optimizer, int(warmup_frac * schedule_steps), schedule_steps)
    ema = base.EMA(model, ema_decay) if use_ema else None
    rng = np.random.default_rng(seed)
    stream = PuzzleStream(train_clues, train_targets, rng)
    trainer = ACTTrainer(model, stream, arguments.batch_size,
                         halt_threshold=arguments.halt_threshold)

    checks = math.ceil(total_puzzles / arguments.eval_every)
    best, best_state, best_extra = 0.0, None, None
    try:
        for check in range(1, checks + 1):
            target = min(check * arguments.eval_every, total_puzzles)
            tick = time.perf_counter()
            stats = trainer.run_until(optimizer, scheduler, ema, target,
                                      check_every=arguments.check_every)
            seconds = time.perf_counter() - tick
            with ema.averaged(model) if ema else contextlib.nullcontext():
                scores = {
                    compute: evaluate(model, valid_split, compute=compute,
                                      batch_size=base.EVAL_BATCH)
                    for compute in EVAL_COMPUTE}
                adaptive = evaluate_act(model, valid_split,
                                        max_sup=max(EVAL_COMPUTE),
                                        batch_size=base.EVAL_BATCH,
                                        threshold=arguments.halt_threshold)
                top = max(EVAL_COMPUTE, key=lambda c: scores[c]["board"])
                board = scores[top]["board"]
                if board > best:
                    best, best_state = board, {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()}
                    trial.set_user_attr("best_puzzles", stats["consumed"])
                    trial.set_user_attr("best_compute", top)
                    trial.set_user_attr("train_depth", stats["depth"])
                    trial.set_user_attr("act_board", adaptive["board"])
                    trial.set_user_attr("act_depth", adaptive["depth"])
                    best_extra = {
                        "valid_cell": scores[top]["cell"],
                        "act_board": adaptive["board"],
                        "act_depth": adaptive["depth"],
                        "train_depth": stats["depth"],
                        "consumed_puzzles": stats["consumed"]}
            boards = "/".join(
                f"{scores[compute]['board']:.4f}" for compute in EVAL_COMPUTE)
            print(f"  trial {trial.number} puzzles "
                  f"{stats['consumed']:,}/{total_puzzles:,}"
                  f"  loss {stats['loss']:.4f} (q {stats['q']:.4f})"
                  f"  depth {stats['depth']:.2f}"
                  f" (cap {stats['capped']:.0%})"
                  f"  {stats['halted']:,} puzzles"
                  f" ({stats['halted'] / seconds:,.0f}/s)"
                  f"  board {boards} @n_sup "
                  + "/".join(map(str, EVAL_COMPUTE))
                  + f"  act {adaptive['board']:.4f}"
                  f"@{adaptive['depth']:.2f}")
            trial.report(board, check)
            if trial.should_prune():
                raise optuna.TrialPruned()
        base.save_if_best(trial, best, best_state, arguments.study_name,
                          extra=best_extra)
    except torch.cuda.OutOfMemoryError:
        trial.set_user_attr("oom", True)
        raise optuna.TrialPruned()
    finally:
        del model, optimizer, trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=10,
                        help="full passes over --n-train puzzles per trial "
                             "(8 epochs of the default 1,001,000-example "
                             "training set is ~8,008,000 puzzles); the "
                             "learning-rate schedule spans the worst-case "
                             "optimizer-step count for this many puzzles "
                             "at the trial's own n_sup, as if halting "
                             "never fired")
    parser.add_argument("--eval-every", type=int, default=200_000,
                        help="puzzles consumed between evaluations, prints "
                             "and pruner reports, regardless of how many "
                             "optimizer steps that took (~40 checks over "
                             "the default 8 epochs)")
    parser.add_argument("--check-every", type=int, default=50,
                        help="optimizer steps between progress checks of "
                             "the puzzles-consumed count within a training "
                             "chunk (one host read each); kept small "
                             "relative to --eval-every so the puzzle "
                             "budget is not overshot by much")
    parser.add_argument("--n-train", type=int, default=1_001_000)
    parser.add_argument("--n-valid", type=int, default=2000,
                        help="validation puzzles per periodic evaluation")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="batch slots of the refill loop")
    parser.add_argument("--halt-threshold", type=float, default=0.0,
                        help="margin the halt logit must clear to stop, "
                             "for training and adaptive evaluation alike; "
                             "0 is the paper's q > 0, which the soft-min "
                             "head already makes conservative (it lower-"
                             "bounds the least confident cell by log 81); "
                             "raise it to halt still less often")
    parser.add_argument("--timeout", type=float, default=None,
                        help="stop starting new trials after this many s")
    parser.add_argument("--storage",
                        default="sqlite:///optuna_trm_extreme_act.db")
    parser.add_argument("--study-name", default="trm-extreme-act-8k")
    parser.add_argument("--compile", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="torch.compile the round step (same numerics "
                             "up to rounding error)")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["default", "reduce-overhead",
                                 "max-autotune"])
    parser.add_argument("--unroll", default=False,
                        action=argparse.BooleanOptionalAction,
                        help="compile whole n-round cycles as single CUDA "
                             "graphs (a few %% faster, longer compile)")
    parser.add_argument("--device", default="cuda"
                        if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPUs to use in parallel, one worker process "
                             "each sharing one study")
    arguments = parser.parse_args(argv)

    if (arguments.gpus > 1 and arguments.device.startswith("cuda")
            and torch.cuda.is_available()):
        ids = base.available_gpu_ids()[:arguments.gpus]
        if len(ids) > 1:
            code = run_on_gpus(arguments, ids)
            base.report(optuna.load_study(
                study_name=arguments.study_name, storage=arguments.storage))
            return code

    device = torch.device(arguments.device)

    splits = sudoku_extreme.load("special_large")
    train_split = splits["train"].subsample(arguments.n_train)
    valid_split = splits["valid"].subsample(arguments.n_valid)
    # resident on the GPU once, shared read-only across every trial.
    train_clues, train_targets = base.to_device(train_split, device)

    study = optuna.create_study(
        study_name=arguments.study_name, storage=arguments.storage,
        direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=2))
    # the fixed-compute winner runs first, as the in-study ACT baseline,
    # then the same recipe with the larger halting cap; skip_if_exists
    # stops restarts and sibling workers from re-queueing them.
    study.enqueue_trial(BASELINE, skip_if_exists=True)
    study.enqueue_trial({**BASELINE, "n_sup": 16}, skip_if_exists=True)
    study.optimize(
        lambda trial: objective(
            trial, arguments, train_clues, train_targets, valid_split,
            device),
        n_trials=arguments.trials, timeout=arguments.timeout)

    base.report(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
