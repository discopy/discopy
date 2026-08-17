# -*- coding: utf-8 -*-

"""
The optuna search behind the ``act`` recipe: model C with adaptive
computation time on sudoku-extreme.

    pip install optuna
    python optuna_act.py --gpus 1 --workers-per-gpu 2 --timeout 259200

Every trial trains one configuration from scratch on
``sudoku_extreme_special_large`` (1,001,000 examples) at
:data:`SEARCH_WIDTHS` -- three times the matched widths, ~1.0M parameters,
:data:`config.EXTREME_TRM`'s -- and returns its **best validation
board-solve rate across evaluations**, i.e. with checkpoint selection.
Training is :class:`train.ACTTrainer`: the paper's deep supervision with
early stopping,

    loss  = softmax_cross_entropy(y_hat, y_true)
    loss += binary_cross_entropy(q_hat, (y_hat == y_true))
    ...
    if q_hat > 0: break     # early-stopping

where the per-example ``break`` is the slot refill -- a puzzle leaves its
batch slot the moment its halt logit clears the threshold (or at the
``n_sup`` cap) and a fresh puzzle takes its place, so the device always
steps a full batch.

Two departures from the paper's letter, both in the direction of
protecting accuracy, and both what the recorded checkpoints were trained
under: the head is **detached**, so the trunk's gradients are identical to
the plain loop's and the halt loss cannot cost accuracy at any weight; and
it is the **soft-minimum per-cell head** rather than a pooled scalar,
which halts on the *least* confident cell and is conservative by
construction (a mean dilutes one wrong cell by 1/81).

**The budget and the evaluation cadence are both counted in puzzles
consumed**, not optimizer steps: one optimizer step consumes a different
number of fresh puzzles depending on the model's mean halting depth, so
pinning either to a step count would make trials with different depths
train on, and be evaluated over, different amounts of data.  A trial
trains for ``--epochs`` full passes over ``--n-train`` puzzles and is
evaluated every ``--eval-every`` puzzles; ``n_sup`` is the halting *cap*,
and the learning-rate schedule spans the *worst-case* optimizer-step count
-- as if no puzzle ever halted early -- so a model that learns to halt
early finishes before the schedule fully decays.

``--epochs`` defaults to 6, ~6.0M puzzles, rather than the 10 the first
campaign ran, and ``--schedule-epochs`` stays at 10: a trial is **cut
short on the full-length schedule** rather than given a shorter one of
its own.  The distinction is the whole point.  Compressing the cosine
into six epochs would anneal faster and land somewhere else -- a new
recipe, whose trials could only be compared with the recorded ones
loosely and would need a study of their own.  Stopping early leaves the
learning rate at every step exactly what a ten-epoch trial had at that
step, so a truncated trial is a *prefix* of a full one: its curve is
comparable to the records check for check, the median pruner has a bar
made of the real thing, and both kinds live in one study.

Six is where the records say the information runs out.  Of the seven
completed trials on this benchmark, six reach their best board rate by
check 27 and all but one by check 30; truncating every one of them at
check 30 would have cost 0.0004 board on average and 0.0025 at worst,
against a wall clock cut by 40%.  The residual bias is that a truncated
trial maximises over 30 checks where a full one had 51, which is that
same 0.0004.

The fixed-compute sweep of the objective is a single depth,
:data:`EVAL_COMPUTE`, since the adaptive protocol is the point of the
search; each evaluation additionally reports the adaptive protocol --
early stopping at inference -- with its mean depth, in the trial's user
attributes.

The seed is drawn at random for every trial and recorded, so the search
ranks configurations rather than lucky seeds.

**Continuing the recorded search.**  ``--seed-from`` copies the completed
trials of an earlier study into this one, with their hyperparameters,
their values and their whole intermediate curves:

    python optuna_act.py --seed-from ../../../optuna_trm_extreme_act.db \\
        --seed-study trm-extreme-act-8k

so that :class:`optuna.pruners.MedianPruner` has a median to compare
against from the first trial rather than after
``--pruner-startup`` fresh ones, and the sampler starts from a posterior
rather than from the prior.  The best imported configuration is enqueued
first, and the fixed-compute anchor :data:`BASELINE` second, so a resumed
campaign begins by re-measuring what it already knows.
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

# so the example imports as a flat set of modules regardless of the
# caller's cwd, e.g. when a worker is spawned by ``--gpus``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset  # noqa: E402
import evaluate as evaluations  # noqa: E402
import model as zoo  # noqa: E402
from config import ARTIFACTS, EXTREME_TRM  # noqa: E402
from train import (  # noqa: E402
    ACTTrainer, EMA, ExampleStream, adamw, cosine_schedule, seed_everything,
    to_device)

#: The widths of the ``extreme`` recipe, ~1.0M parameters: the extreme
#: puzzles need more capacity than the matched-budget models, and part of
#: the extra arithmetic still hides under the kernel-launch overhead that
#: dominates at these sizes.
SEARCH_WIDTHS = EXTREME_TRM["widths"]

#: The fixed-compute depth the objective is read at, and the cap of the
#: adaptive protocol beside it.  One depth rather than a sweep: the
#: adaptive number is what the search is about, and the fixed one is a
#: comparability anchor to the fixed-compute study.
EVAL_COMPUTE = (16, )

#: The whole validation subset scored as one forward-only batch.
EVAL_BATCH = 2000

#: The winning configuration of the fixed-compute study ``trm-extreme-3x``
#: trial 5 -- :data:`config.EXTREME_TRM`, 0.4632 valid boards at trained
#: depth -- enqueued so the search always measures ACT against the
#: identical recipe, with ``n_sup`` now the halting cap.
BASELINE = {
    "lr": EXTREME_TRM["lr"], "n": EXTREME_TRM["rounds"],
    "T": EXTREME_TRM["cycles"], "n_sup": EXTREME_TRM["steps"],
    "use_ema": False, "warmup_frac": EXTREME_TRM["warmup_frac"],
    "weight_decay": EXTREME_TRM["weight_decay"]}

#: The study's storage, beside the checkpoints it selects.
STORAGE = f"sqlite:///{ARTIFACTS / 'optuna-act-extreme.db'}"


def make_storage(spec: str):
    """
    A storage from ``spec``: a URL, a path to a sqlite file, or a path to
    an optuna **journal**, which is what workers on more than one node
    need.

    Sqlite takes a whole-file write lock and reaches for ``fcntl`` to do
    it, which a shared filesystem implements per client rather than per
    cluster: two nodes writing one ``.db`` on Ceph or NFS is a corruption
    waiting to happen, however long it survives.  A journal is
    append-only and takes its lock by creating a symlink -- an atomic
    operation on every network filesystem worth the name -- so workers
    scattered over several allocations share one study safely, which is
    the whole point of pooling them: every trial any of them finishes is
    a curve the others prune against.

    Parameters:
        spec : ``...journal`` for a journal, a path for sqlite, or any
               URL optuna's RDB layer accepts.
    """
    if spec.endswith(".journal"):
        path = str(Path(spec).resolve())
        return optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(
                path, optuna.storages.journal.JournalFileSymlinkLock(path)))
    return spec if "://" in spec else "sqlite:///" + str(Path(spec).resolve())


def available_gpu_ids() -> list[str]:
    """ Physical CUDA device ids visible to this process, as strings. """
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited:
        return [id_.strip() for id_ in inherited.split(",") if id_.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def child_argv(arguments: argparse.Namespace) -> list[str]:
    """ A CLI invocation of this script matching ``arguments``, one GPU. """
    argv = [
        "--trials", str(arguments.trials),
        "--epochs", str(arguments.epochs),
        "--schedule-epochs", str(arguments.schedule_epochs),
        "--eval-every", str(arguments.eval_every),
        "--check-every", str(arguments.check_every),
        "--n-train", str(arguments.n_train),
        "--n-valid", str(arguments.n_valid),
        "--batch-size", str(arguments.batch_size),
        "--halt-threshold", str(arguments.halt_threshold),
        "--variant", arguments.variant,
        "--storage", arguments.storage,
        "--study-name", arguments.study_name,
        "--pruner-startup", str(arguments.pruner_startup),
        "--pruner-warmup", str(arguments.pruner_warmup),
        "--device", "cuda",
        "--gpus", "1", "--workers-per-gpu", "1",
        "--compile" if arguments.compile else "--no-compile",
        "--unroll" if arguments.unroll else "--no-unroll"]
    if arguments.timeout is not None:
        argv += ["--timeout", str(arguments.timeout)]
    if arguments.compile_mode:
        argv += ["--compile-mode", arguments.compile_mode]
    return argv


def run_on_gpus(arguments: argparse.Namespace, ids: list[str]) -> int:
    """
    ``--workers-per-gpu`` worker processes per id in ``ids``, each pinned
    to its device and all sharing one study; the worst exit code.

    More than one worker per device is oversubscription, which is
    worthwhile precisely because these maps are launch-bound: a second
    process fills the gaps the first leaves between its kernels.  It costs
    a second copy of the resident training set per worker.
    """
    argv, script = child_argv(arguments), os.path.abspath(__file__)
    children = [
        subprocess.Popen([sys.executable, script, *argv],
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_id))
        for gpu_id in ids for _ in range(arguments.workers_per_gpu)]
    return max(child.wait() for child in children)


def report(study: optuna.Study) -> None:
    """ Print the best trial's board-solve rate, seed and hyperparameters. """
    try:
        best_value, best_trial = study.best_value, study.best_trial
    except ValueError:
        print("\nno completed trials")
        return
    print(f"\nbest valid boards {best_value:.4f} "
          f"(trial {best_trial.number}, "
          f"seed {best_trial.user_attrs.get('seed')}, "
          f"{best_trial.user_attrs.get('best_puzzles')} puzzles, "
          f"n_sup at eval {best_trial.user_attrs.get('best_compute')}, "
          f"act {best_trial.user_attrs.get('act_board')}"
          f"@{best_trial.user_attrs.get('act_depth')})")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")


def random_seed(trial: optuna.Trial) -> int:
    """ A fresh random seed per trial, recorded so the run is replayable. """
    seed = int.from_bytes(os.urandom(4), "little") % (2 ** 31)
    trial.set_user_attr("seed", seed)
    seed_everything(seed)
    return seed


def save_if_best(trial: optuna.Trial, board: float, state: dict,
                 study_name: str, keep: int = 5, extra: dict = None) -> None:
    """
    Keep the weights of a trial that lands in the study's best ``keep``.

    Not the record alone: the runner-up of a search is what a best-of-k
    protocol, an ensemble or a seed-variance check is made of, and a
    checkpoint costs four megabytes where reproducing one costs hours --
    reproducible in principle, since the seed and the parameters are
    recorded, but only in principle.  Nothing is ever deleted, so workers
    on several nodes never race over the same file.

    Parameters:
        keep : How many of the study's best a trial must reach to be
               written; ``1`` is the record-only rule.
        extra : Evaluation results stored alongside the checkpoint, e.g.
                cell accuracy or adaptive-compute statistics.
    """
    done = sorted(
        (t.value for t in trial.study.get_trials(deepcopy=False)
         if t.state == optuna.trial.TrialState.COMPLETE
         and t.value is not None), reverse=True)
    previous = done[keep - 1] if len(done) >= keep else -1.0
    if state is not None and board > previous:
        torch.save(
            {"state_dict": state, "params": trial.params,
             "widths": SEARCH_WIDTHS.asdict(),
             "seed": trial.user_attrs["seed"], "valid_board": board,
             **(extra or {})},
            ARTIFACTS / f"optuna-{study_name}-trial{trial.number}.pt")


def import_legacy(study: optuna.Study, storage: str, name: str,
                  max_step: int = None) -> int:
    """
    Copy the trials of an earlier study into ``study``, with their
    distributions, values, user attributes and intermediate curves.

    With ``max_step`` a trial is copied *as if it had stopped there*: its
    curve is cut at that check and its value re-read as the maximum over
    what is left, which is the objective this study measures.  That is
    exactly right when the budget is a **truncation** rather than a
    schedule of its own, since a longer trial's first ``max_step`` checks
    are a short trial run under the same learning rate at every step --
    and it is what lets a run interrupted *past* ``max_step`` count as a
    finished short trial rather than be discarded.  Without ``max_step``
    only ``COMPLETE`` trials are copied, with the values they were given.
    Returns how many were copied.

    Parameters:
        storage : The storage of the study to copy from.
        name : Its study name.
        max_step : The check to truncate at, ``None`` to keep trials whole.
    """
    source = optuna.load_study(study_name=name, storage=storage)
    known = {tuple(sorted(trial.params.items())) for trial in study.trials}
    copied = 0
    for trial in source.get_trials(deepcopy=False):
        curve = {step: value
                 for step, value in trial.intermediate_values.items()
                 if max_step is None or step <= max_step}
        if max_step is None:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            value = trial.value
        else:
            if not curve or max(curve) < max_step:
                continue
            value = max(curve.values())
        if tuple(sorted(trial.params.items())) in known:
            continue
        study.add_trial(optuna.trial.create_trial(
            params=trial.params, distributions=trial.distributions,
            value=value, intermediate_values=curve,
            user_attrs={**trial.user_attrs,
                        "imported_from": f"{name}#{trial.number}",
                        "imported_state": trial.state.name},
            state=optuna.trial.TrialState.COMPLETE))
        copied += 1
    return copied


def objective(trial, arguments, train_clues, train_targets,
              valid_split, device) -> float:
    seed = random_seed(trial)
    # a truncated trial and a full-length one belong in the same study --
    # same schedule, so one is a prefix of the other -- but they maximise
    # over a different number of checks, so each says which it is.
    trial.set_user_attr("epochs", arguments.epochs)
    trial.set_user_attr("schedule_epochs", arguments.schedule_epochs)
    lr = trial.suggest_float("lr", 2e-4, 3e-3, log=True)
    rounds = trial.suggest_categorical("n", [8, 10])
    cycles = trial.suggest_categorical("T", [4, 6])
    n_sup = trial.suggest_categorical("n_sup", [12, 16])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    warmup_frac = trial.suggest_float("warmup_frac", 0.0, 0.1)
    use_ema = trial.suggest_categorical("use_ema", [True, False])
    ema_decay = trial.suggest_float(
        "ema_decay", 0.99, 0.9995) if use_ema else None

    model = zoo.act(SEARCH_WIDTHS, rounds=rounds, cycles=cycles, steps=n_sup,
                    halt_detach=True, halt_head="softmin").to(device)
    if arguments.compile and device.type == "cuda":
        # fresh modules per trial: recompile from a clean slate, or the
        # recompile limit would silently fall back to eager.
        torch._dynamo.reset()
        model.map.compile_rounds(
            unroll=arguments.unroll,
            **({"mode": arguments.compile_mode}
               if arguments.compile_mode else {}))
    total_puzzles = arguments.epochs * arguments.n_train
    # the schedule spans ``--schedule-epochs``, which is *not* the budget:
    # a trial stops early on a schedule it does not finish, so its
    # learning-rate trajectory is identical to a full-length run's at
    # every step and its curve is comparable to one check for check.
    schedule_puzzles = arguments.schedule_epochs * arguments.n_train
    # worst-case optimizer-step count for the schedule, as if no puzzle
    # ever halted early -- the analogue of the fixed-compute study's
    # ``iterations * n_sup``, derived from the puzzle budget.
    schedule_steps = math.ceil(schedule_puzzles / arguments.batch_size) * n_sup
    optimizer = adamw(model, lr, weight_decay)
    scheduler = cosine_schedule(
        optimizer, int(warmup_frac * schedule_steps), schedule_steps)
    ema = EMA(model, ema_decay) if use_ema else None
    stream = ExampleStream(train_clues, train_targets,
                           np.random.default_rng(seed))
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
                    compute: evaluations.evaluate(
                        model, valid_split, compute=compute,
                        batch_size=EVAL_BATCH)
                    for compute in EVAL_COMPUTE}
                adaptive = evaluations.evaluate_act(
                    model, valid_split, max_steps=max(EVAL_COMPUTE),
                    batch_size=EVAL_BATCH,
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
                  f"@{adaptive['depth']:.2f}", flush=True)
            trial.report(board, check)
            if trial.should_prune():
                raise optuna.TrialPruned()
        save_if_best(trial, best, best_state, arguments.study_name,
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
    parser.add_argument("--trials", type=int, default=40,
                        help="trials per worker")
    parser.add_argument("--epochs", type=int, default=6,
                        help="full passes over --n-train puzzles per trial "
                             "(6 of the default 1,001,000 is ~6.0M puzzles); "
                             "the schedule spans the worst-case "
                             "optimizer-step count for this many puzzles at "
                             "the trial's own n_sup, as if halting never "
                             "fired")
    parser.add_argument("--schedule-epochs", type=int, default=10,
                        help="epochs the learning-rate schedule spans, "
                             "which is not the budget: leaving this at the "
                             "10 the first campaign ran while --epochs is 6 "
                             "stops a trial early on a schedule it does not "
                             "finish, so its learning rate at every step -- "
                             "and hence its curve -- is identical to a "
                             "full-length trial's and the two are directly "
                             "comparable")
    parser.add_argument("--eval-every", type=int, default=200_000,
                        help="puzzles consumed between evaluations, prints "
                             "and pruner reports, regardless of how many "
                             "optimizer steps that took")
    parser.add_argument("--check-every", type=int, default=50,
                        help="optimizer steps between progress checks of "
                             "the puzzles-consumed count, one host read "
                             "each; small relative to --eval-every so the "
                             "budget is not overshot by much")
    parser.add_argument("--n-train", type=int, default=1_001_000)
    parser.add_argument("--n-valid", type=int, default=2000,
                        help="validation puzzles per periodic evaluation")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="batch slots of the refill loop")
    parser.add_argument("--halt-threshold", type=float, default=0.0,
                        help="margin the halt logit must clear to stop, for "
                             "training and adaptive evaluation alike; 0 is "
                             "the paper's q > 0, which the soft-min head "
                             "already makes conservative")
    parser.add_argument("--variant", default="special_large",
                        choices=("standard", "special", "special_large"))
    parser.add_argument("--timeout", type=float, default=None,
                        help="stop starting new trials after this many s")
    parser.add_argument("--storage", default=STORAGE)
    parser.add_argument("--study-name", default="act-extreme")
    parser.add_argument("--seed-from", default=None,
                        help="storage URL, or path to a sqlite file, whose "
                             "completed trials seed this study")
    parser.add_argument("--seed-study", default="trm-extreme-act-8k",
                        help="the study name to read in --seed-from")
    parser.add_argument("--seed-max-step", type=int, default=None,
                        help="cut imported curves at this check and re-read "
                             "their value as the maximum over what is left, "
                             "so a longer trial is imported as the short one "
                             "it contains; defaults to the number of whole "
                             "checks this run's own trials will report, which "
                             "is what makes an imported curve a like-for-like "
                             "bar. Pass 0 to import whole trials instead")
    parser.add_argument("--pruner-startup", type=int, default=2,
                        help="completed trials before the median pruner "
                             "fires; the default assumes the two imported "
                             "trials of --seed-from")
    parser.add_argument("--pruner-warmup", type=int, default=8,
                        help="evaluations a trial is given before it may be "
                             "pruned; 8 of the default cadence is 1.6M "
                             "puzzles, past which the recorded curves and a "
                             "diverging one separate cleanly")
    parser.add_argument("--compile", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="torch.compile the round step (same numerics "
                             "up to rounding error)")
    parser.add_argument("--compile-mode", default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--unroll", default=False,
                        action=argparse.BooleanOptionalAction,
                        help="compile whole n-round cycles as single CUDA "
                             "graphs (a few %% faster, longer compile)")
    parser.add_argument("--device", default="cuda"
                        if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPUs to use in parallel, sharing one study")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="worker processes per GPU; oversubscribing a "
                             "launch-bound model fills the gaps one worker "
                             "leaves between its kernels")
    arguments = parser.parse_args(argv)
    # ``arguments.storage`` stays the raw spec, since that is what a child
    # is handed on its command line; every use resolves it locally.
    storage = make_storage(arguments.storage)
    # the whole checks this run's trials will report: the point at which an
    # imported curve stops being comparable to one of ours.
    if arguments.seed_max_step is None:
        arguments.seed_max_step = (
            arguments.epochs * arguments.n_train) // arguments.eval_every
    max_step = arguments.seed_max_step or None

    # TF32 matmuls on Ampere+: near-free throughput for the float32 GEMMs
    # these solvers are.  Set before any worker builds a model, and in
    # every worker, since it is process-wide state.
    torch.set_float32_matmul_precision("high")

    if (arguments.device.startswith("cuda") and torch.cuda.is_available()
            and arguments.gpus * arguments.workers_per_gpu > 1):
        ids = available_gpu_ids()[:arguments.gpus]
        # the artifacts are built once here rather than raced by the
        # children, which would each write the same files.
        dataset.load_extreme(arguments.variant, verify=False)
        study = optuna.create_study(
            study_name=arguments.study_name, storage=storage,
            direction="maximize", load_if_exists=True)
        if arguments.seed_from:
            copied = import_legacy(
                study, make_storage(arguments.seed_from), arguments.seed_study,
                max_step=max_step)
            print(f"imported {copied} trials from {arguments.seed_study}")
        code = run_on_gpus(arguments, ids)
        report(optuna.load_study(study_name=arguments.study_name,
                                 storage=storage))
        return code

    device = torch.device(arguments.device)

    splits = dataset.load_extreme(arguments.variant)
    train_split = splits["train"].subsample(arguments.n_train)
    valid_split = splits["valid"].subsample(arguments.n_valid)
    # resident on the device once, shared read-only across every trial.
    train_clues, train_targets = to_device(train_split, device)

    study = optuna.create_study(
        study_name=arguments.study_name, storage=storage,
        direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=arguments.pruner_startup,
            n_warmup_steps=arguments.pruner_warmup))
    if arguments.seed_from:
        copied = import_legacy(
            study, make_storage(arguments.seed_from),
            arguments.seed_study, max_step=max_step)
        print(f"imported {copied} trials from {arguments.seed_study}")
    # the record so far runs first, then the fixed-compute anchor, so a
    # resumed campaign begins by re-measuring what it already knows;
    # skip_if_exists stops restarts and sibling workers re-queueing them.
    with contextlib.suppress(ValueError):
        study.enqueue_trial(study.best_params, skip_if_exists=True)
    study.enqueue_trial(BASELINE, skip_if_exists=True)
    study.optimize(
        lambda trial: objective(
            trial, arguments, train_clues, train_targets, valid_split,
            device),
        n_trials=arguments.trials, timeout=arguments.timeout)

    report(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
