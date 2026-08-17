# -*- coding: utf-8 -*-

"""
Scoring a trained sudoku model.

    python evaluate.py artifacts/best-goi-seed0.pt --model goi
    python evaluate.py artifacts/optuna-trm-extreme-act-8k-trial2.pt \\
        --model act --noise

Four protocols, and they are **not** interchangeable:

* :func:`evaluate` -- fixed compute.  The model runs exactly the depth it
  is asked for and is scored on the answer it ends with.
  :func:`sweep_compute` runs it at several depths, which is where the
  "keeps refining past its trained depth" numbers of ``README.md`` come
  from.
* :func:`evaluate_act` -- adaptive compute.  Every puzzle runs until its
  halt logit clears a threshold or the cap is reached.  Stopping early
  changes the predictions -- that is the point -- so both this and the
  fixed number are worth reporting.
* :func:`evaluate_selected` -- best-of-k.  The halt logit is a learned
  *verifier* of the answer it halts on, so it can pick between independent
  stochastic rollouts.  ``solved_oracle`` is the pass@k a perfect verifier
  would reach, i.e. the honest upper bound on what selection can buy.
* :func:`noise_sweep` -- the latent-noise study: a grid of supervision
  depths against noise levels on the answer trace, with a beam schedule
  that spends depth where the fan-out matters.

Every score is under the fill-in-the-blanks decode rule
(:meth:`~model.Decoder.decode`): the clues are written back over the
predictions, so a model is never scored on a cell it was given.  Cell
accuracy is therefore not comparable across benchmarks with different
numbers of givens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import dataset
import model as zoo
from config import ARTIFACTS, Widths
from model import Decoder

#: The supervision depths (rows) and noise levels (columns) of the study.
CAPS = (16, 32, 64, 128, 256)
SIGMAS = (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)

#: The per-example bookkeeping a trajectory carries besides its state.
BOOK_KEYS = ("halted", "halt_step", "halt_correct", "halt_cell", "halt_q")

#: What is recorded at every cap.
RECORD_KEYS = ("fixed_correct", "fixed_cell", "fixed_q",
               "act_correct", "act_cell", "act_q", "act_depth")


def compute_kwargs(model, compute) -> dict:
    """
    Test-time compute in the unit the model measures it in: rounds for the
    single-run models, supervision steps for the recursion.
    """
    return {"steps": compute} if model.segmented else {"rounds": compute}


# --- fixed compute ---------------------------------------------------------

@torch.no_grad()
def evaluate(model, split, compute: int = None, batch_size: int = 256,
             buckets: bool = False) -> dict:
    """
    Cell and board accuracy on a split, optionally bucketed by givens.

    Parameters:
        model : The trained model.
        split : The split to evaluate on.
        compute : The test-time compute, ``None`` for the trained one.
        batch_size : The evaluation batch size.
        buckets : Whether to also return the per-givens breakdown.
    """
    model.eval()
    device = next(model.parameters()).device
    correct, solved = [], []
    for start in range(0, len(split), batch_size):
        stop = start + batch_size
        clues = torch.as_tensor(
            split.puzzles[start:stop], dtype=torch.long, device=device)
        target = torch.as_tensor(
            split.solutions[start:stop], dtype=torch.long, device=device)
        logits = model(clues, **compute_kwargs(model, compute))
        matches = Decoder.decode(logits, clues) == target
        correct.append(matches.float().mean(1).cpu().numpy())
        solved.append(matches.all(1).cpu().numpy())
    correct, solved = np.concatenate(correct), np.concatenate(solved)
    result = {"cell": float(correct.mean()), "board": float(solved.mean())}
    if buckets:
        givens = split.givens
        result["by_givens"] = [
            {"givens": int(value), "n": int((givens == value).sum()),
             "cell": float(correct[givens == value].mean()),
             "board": float(solved[givens == value].mean())}
            for value in sorted(set(givens.tolist()))]
    return result


def sweep_compute(model, split, values, batch_size: int = 256) -> list:
    """ Accuracy as a function of test-time compute. """
    return [dict(compute=value,
                 **evaluate(model, split, compute=value,
                            batch_size=batch_size))
            for value in values]


# --- adaptive compute ------------------------------------------------------

@torch.no_grad()
def evaluate_act(model, split, max_steps: int = None,
                 batch_size: int = 2000, threshold: float = 0.0) -> dict:
    """
    Inference with the paper's early stopping: every puzzle runs until its
    halt logit clears ``threshold`` or ``max_steps`` supervision steps are
    reached, and is scored on the answer it halted with.

    Returns:
        ``cell``, ``board`` and the mean ``depth`` actually run.
    """
    model.eval()
    max_steps = model.n_sup if max_steps is None else max_steps
    device = next(model.parameters()).device
    correct, solved, depths = [], [], []
    for start in range(0, len(split), batch_size):
        stop = start + batch_size
        clues = torch.as_tensor(
            split.puzzles[start:stop], dtype=torch.long, device=device)
        target = torch.as_tensor(
            split.solutions[start:stop], dtype=torch.long, device=device)
        torch.compiler.cudagraph_mark_step_begin()
        state = model.initial(clues)
        halted = torch.zeros(len(clues), dtype=torch.bool, device=device)
        final = torch.zeros(len(clues), model.n_cells, model.n, device=device)
        depth = torch.full((len(clues), ), max_steps, dtype=torch.long,
                           device=device)
        for t in range(1, max_steps + 1):
            torch.compiler.cudagraph_mark_step_begin()
            state, logits, halt = model.act_step(state, grad=False)
            state = state.detach().clone()
            newly = ~halted & (
                (model.halt_logit(halt) > threshold) | (t == max_steps))
            final = torch.where(
                newly[:, None, None], logits.to(final.dtype), final)
            depth = torch.where(newly, torch.full_like(depth, t), depth)
            halted |= newly
            if bool(halted.all()):
                break
        matches = Decoder.decode(final, clues) == target
        correct.append(matches.float().mean(1).cpu().numpy())
        solved.append(matches.all(1).cpu().numpy())
        depths.append(depth.cpu().numpy())
    return {"cell": float(np.concatenate(correct).mean()),
            "board": float(np.concatenate(solved).mean()),
            "depth": float(np.concatenate(depths).mean())}


def perturb_answer(model, state, sigma: float, generator=None):
    """
    The carried state with ``y <- y + eps`` on the answer trace, sampled
    independently per example, cell and feature and written to *both* ends
    of the loop, which intentionally share one ``y``.

    Sudoku has no padded nodes -- all 81 cells are real -- so no valid-node
    mask is needed; a task with padding would mask ``eps`` here.

    Parameters:
        model : The model whose answer trace to perturb.
        state : The flat carried messages.
        sigma : The standard deviation of the noise.
        generator : The rollout's own ``torch.Generator``.
    """
    if not sigma:
        return state
    interaction = model.interaction
    answer = interaction.read(state, model.answer)
    noise = sigma * torch.randn(
        answer.shape, generator=generator, device=answer.device,
        dtype=answer.dtype)
    return interaction.write(state, model.answer, answer + noise)


@torch.no_grad()
def evaluate_selected(model, split, rollouts: int = 4, sigma: float = 0.1,
                      max_steps: int = None, batch_size: int = 2000,
                      threshold: float = 0.0, seed: int = 0) -> dict:
    """
    Best-of-``rollouts`` inference selected by the halt head.

    Each puzzle runs independent trajectories, each with
    :func:`perturb_answer` noise refreshed at the start of every
    supervision step, each halting by the rule of :func:`evaluate_act`; the
    answer kept is the one whose halt logit at its halting step is largest.
    With ``rollouts=1, sigma=0`` this is exactly :func:`evaluate_act`.

    Returns:
        ``cell``, ``board`` and ``depth`` of the selected rollout, plus
        ``board_mean``, the single-rollout average, and ``board_oracle``,
        the pass@k upper bound a perfect verifier would reach.
    """
    model.eval()
    max_steps = model.n_sup if max_steps is None else max_steps
    device = next(model.parameters()).device
    cells, chosen, means, oracles, depths = [], [], [], [], []
    for start in range(0, len(split), batch_size):
        stop = start + batch_size
        clues = torch.as_tensor(
            split.puzzles[start:stop], dtype=torch.long, device=device)
        target = torch.as_tensor(
            split.solutions[start:stop], dtype=torch.long, device=device)
        best_q = torch.full((len(clues), ), -torch.inf, device=device)
        best = torch.zeros(len(clues), model.n_cells, model.n, device=device)
        best_depth = torch.zeros(len(clues), dtype=torch.long, device=device)
        solved_any = torch.zeros(len(clues), dtype=torch.bool, device=device)
        solved_mean = torch.zeros(len(clues), device=device)
        for rollout in range(rollouts):
            generator = torch.Generator(device=device)
            generator.manual_seed(1_000_003 * seed + 9_176 * start + rollout)
            torch.compiler.cudagraph_mark_step_begin()
            state = model.initial(clues)
            halted = torch.zeros(len(clues), dtype=torch.bool, device=device)
            final, q_final = torch.zeros_like(best), torch.full_like(
                best_q, -torch.inf)
            depth = torch.full((len(clues), ), max_steps, dtype=torch.long,
                               device=device)
            for t in range(1, max_steps + 1):
                torch.compiler.cudagraph_mark_step_begin()
                state = perturb_answer(model, state, sigma, generator)
                state, logits, halt = model.act_step(state, grad=False)
                state = state.detach().clone()
                q = model.halt_logit(halt)
                newly = ~halted & ((q > threshold) | (t == max_steps))
                final = torch.where(
                    newly[:, None, None], logits.to(final.dtype), final)
                q_final = torch.where(newly, q.to(q_final.dtype), q_final)
                depth = torch.where(newly, torch.full_like(depth, t), depth)
                halted |= newly
                if bool(halted.all()):
                    break
            solved = (Decoder.decode(final, clues) == target).all(1)
            solved_mean += solved.float() / rollouts
            solved_any |= solved
            better = q_final > best_q
            best_q = torch.where(better, q_final, best_q)
            best = torch.where(better[:, None, None], final, best)
            best_depth = torch.where(better, depth, best_depth)
        matches = Decoder.decode(best, clues) == target
        cells.append(matches.float().mean(1).cpu().numpy())
        chosen.append(matches.all(1).cpu().numpy())
        means.append(solved_mean.cpu().numpy())
        oracles.append(solved_any.cpu().numpy())
        depths.append(best_depth.cpu().numpy())
    return {"cell": float(np.concatenate(cells).mean()),
            "board": float(np.concatenate(chosen).mean()),
            "board_mean": float(np.concatenate(means).mean()),
            "board_oracle": float(np.concatenate(oracles).mean()),
            "depth": float(np.concatenate(depths).mean())}


# --- the latent-noise study ------------------------------------------------

@torch.no_grad()
def latent_stats(model, clues, steps: int = 16) -> dict:
    """
    Statistics of the deterministic carried states, pooled over ``steps``
    supervision steps: global mean, per-feature standard deviation, root
    mean square and mean per-cell norm, for the answer ``y`` -- the state
    :func:`perturb_answer` perturbs -- and for the latent ``z``.

    ``y`` is written post-``LayerNorm``, so its RMS is about one and
    ``sigma`` reads directly as a relative noise level.  This is the number
    that makes that claim checkable.
    """
    interaction = model.interaction
    keys = {"y": model.answer, "z": ("cell", zoo.STATE)}
    totals = {name: None for name in keys}
    torch.compiler.cudagraph_mark_step_begin()
    state = model.initial(clues)
    for _ in range(steps):
        torch.compiler.cudagraph_mark_step_begin()
        state, _, _ = model.act_step(state, grad=False)
        state = state.detach().clone()
        for name, key in keys.items():
            flat = interaction.read(state, key).reshape(
                -1, interaction.widths[interaction.heads[key][0]]).double()
            if totals[name] is None:
                totals[name] = {
                    "count": 0, "sum": torch.zeros_like(flat[0]),
                    "sumsq": torch.zeros_like(flat[0]),
                    "norm": torch.zeros((), dtype=flat.dtype,
                                        device=flat.device)}
            total = totals[name]
            total["count"] += flat.shape[0]
            total["sum"] += flat.sum(0)
            total["sumsq"] += (flat ** 2).sum(0)
            total["norm"] += flat.norm(dim=-1).sum()

    result = {}
    for name, total in totals.items():
        mean = total["sum"] / total["count"]
        std = (total["sumsq"] / total["count"] - mean ** 2).clamp(min=0).sqrt()
        result[name] = {
            "global_mean": mean.mean().item(),
            "per_feature_std_mean": std.mean().item(),
            "per_feature_std_min": std.min().item(),
            "per_feature_std_max": std.max().item(),
            "rms": (total["sumsq"].sum()
                    / (total["count"] * len(mean))).sqrt().item(),
            "mean_norm": (total["norm"] / total["count"]).item()}
    return result


def fresh_book(model, clues) -> dict:
    """ A trajectory at step zero: initial state, nothing halted yet. """
    torch.compiler.cudagraph_mark_step_begin()
    n, device = len(clues), clues.device
    return {
        "t": 0, "state": model.initial(clues),
        "halted": torch.zeros(n, dtype=torch.bool, device=device),
        "halt_step": torch.zeros(n, dtype=torch.long, device=device),
        "halt_correct": torch.zeros(n, dtype=torch.bool, device=device),
        "halt_cell": torch.zeros(n, device=device),
        "halt_q": torch.zeros(n, device=device)}


def assemble(snapshots: list, choice) -> dict:
    """
    One continuation trajectory gathered per example across the phase-one
    snapshots: example ``i`` resumes the trajectory of rollout
    ``choice[i]``, state and halting bookkeeping alike.
    """
    book = {"t": snapshots[0]["t"], "state": snapshots[0]["state"].clone(),
            **{key: snapshots[0][key].clone() for key in BOOK_KEYS}}
    for k, snapshot in enumerate(snapshots[1:], start=1):
        mask = choice == k
        if not bool(mask.any()):
            continue
        book["state"][mask] = snapshot["state"][mask]
        for key in BOOK_KEYS:
            book[key][mask] = snapshot[key][mask]
    return book


@torch.no_grad()
def run_segment(model, clues, target, sigma: float, caps, generator,
                threshold: float, book: dict, t_stop: int) -> dict:
    """
    Advance one trajectory from its current step to ``t_stop``, injecting
    noise once at the start of every supervision step -- never per
    message-passing round, and never reused across steps or rollouts -- and
    scoring every cap it passes, both at fixed compute and under the
    paper's early stopping.
    """
    caps, out = set(caps), {}
    for t in range(book["t"] + 1, t_stop + 1):
        state = perturb_answer(model, book["state"], sigma, generator)
        torch.compiler.cudagraph_mark_step_begin()
        state, logits, halt = model.act_step(state, grad=False)
        book["state"] = state.detach().clone()
        q = model.halt_logit(halt)
        matches = Decoder.decode(logits, clues) == target
        correct, cell = matches.all(-1), matches.float().mean(-1)
        newly = ~book["halted"] & (q > threshold)
        book["halt_step"] = torch.where(
            newly, torch.full_like(book["halt_step"], t), book["halt_step"])
        book["halt_correct"] = torch.where(newly, correct,
                                           book["halt_correct"])
        book["halt_cell"] = torch.where(newly, cell, book["halt_cell"])
        book["halt_q"] = torch.where(newly, q, book["halt_q"])
        book["halted"] = book["halted"] | newly
        if t in caps:
            out[t] = {
                "fixed_correct": correct, "fixed_cell": cell, "fixed_q": q,
                "act_correct": torch.where(
                    book["halted"], book["halt_correct"], correct),
                "act_cell": torch.where(
                    book["halted"], book["halt_cell"], cell),
                "act_q": torch.where(book["halted"], book["halt_q"], q),
                "act_depth": torch.where(
                    book["halted"], book["halt_step"],
                    torch.full_like(book["halt_step"], t))}
    book["t"] = t_stop
    return out


def sweep_chunk(model, clues, target, sigma: float, index: int, caps,
                rollouts: int, survivors: int, select_at: int,
                threshold: float, seed: int, seeds: dict, log=print) -> dict:
    """
    The beam schedule of one noise level on one batch: ``rollouts``
    trajectories to the selection depth, then per *example* the
    ``survivors`` most confident continue to the deepest cap, drawing fresh
    noise all the way.  The halt logit is the model's own confidence, the
    only signal available at test time.
    """
    caps = sorted(caps)
    max_cap, device = caps[-1], clues.device

    def make_generator(label, k):
        drawn = int(np.random.SeedSequence(
            [seed, index, k if label == "start" else 10 ** 6 + k]
        ).generate_state(1)[0])
        seeds[f"sigma{sigma:g}_{label}{k}"] = drawn
        return torch.Generator(device=device).manual_seed(drawn)

    if not sigma:
        book = fresh_book(model, clues)
        out = run_segment(model, clues, target, sigma, caps, None,
                          threshold, book, max_cap)
        return {cap: {key: out[cap][key][None] for key in RECORD_KEYS}
                for cap in caps}

    shallow = [cap for cap in caps if cap <= select_at]
    deep = [cap for cap in caps if cap > select_at]
    records = {cap: {key: [] for key in RECORD_KEYS} for cap in caps}
    snapshots, confidence, tick = [], [], time.perf_counter()
    for k in range(rollouts):
        book = fresh_book(model, clues)
        out = run_segment(model, clues, target, sigma, shallow,
                          make_generator("start", k), threshold, book,
                          select_at)
        snapshots.append(book)
        confidence.append(out[select_at]["fixed_q"])
        for cap in shallow:
            for key in RECORD_KEYS:
                records[cap][key].append(out[cap][key])
    board = torch.stack(records[select_at]["fixed_correct"]).float().mean()
    log(f"  sigma {sigma:g}: {rollouts} rollouts @ {select_at}  "
        f"board {board:.4f}  ({time.perf_counter() - tick:.0f}s)",
        flush=True)

    ranks = torch.stack(confidence).argsort(0, descending=True)
    for r in range(survivors):
        book = assemble(snapshots, ranks[r])
        out = run_segment(model, clues, target, sigma, deep,
                          make_generator("continue", r), threshold, book,
                          max_cap)
        for cap in deep:
            for key in RECORD_KEYS:
                records[cap][key].append(out[cap][key])
        log(f"  sigma {sigma:g}: survivor {r + 1}/{survivors} @ {max_cap}  "
            f"board {out[max_cap]['fixed_correct'].float().mean():.4f}  "
            f"({time.perf_counter() - tick:.0f}s)", flush=True)
    return {cap: {key: torch.stack(records[cap][key])
                  for key in RECORD_KEYS} for cap in caps}


def summarize(arrays: dict) -> dict:
    """
    The board metrics of one ``(cap, sigma)`` box from its stacked arrays
    of shape ``(rollouts, n_eval)``:

    * ``single`` -- mean single-rollout accuracy;
    * ``best_of_k`` -- per example, the trajectory the model itself is most
      confident in, i.e. test-time information only;
    * ``pass_at_k`` -- any trajectory solved it, an oracle diagnostic.
    """
    def block(prefix):
        correct, q = arrays[f"{prefix}_correct"], arrays[f"{prefix}_q"]
        columns = np.arange(correct.shape[1])
        return {"single": float(correct.mean()),
                "single_cell": float(arrays[f"{prefix}_cell"].mean()),
                "best_of_k": float(correct[q.argmax(0), columns].mean()),
                "pass_at_k": float(correct.any(0).mean())}

    result = {"k": int(arrays["fixed_correct"].shape[0]),
              "fixed": block("fixed"), "act": block("act")}
    result["act"]["depth"] = float(arrays["act_depth"].mean())
    return result


def table(results: dict, caps, sigmas, protocol: str, metric: str) -> str:
    """ One caps-by-sigmas table of ``results`` as aligned text. """
    lines = ["steps".rjust(6) + "".join(f"{sigma:>10g}" for sigma in sigmas)]
    for cap in caps:
        lines.append(f"{cap:>6d}" + "".join(
            f"{results[cap][sigma][protocol][metric]:>10.4f}"
            for sigma in sigmas))
    return "\n".join(lines)


def noise_sweep(model, split, caps=CAPS, sigmas=SIGMAS, rollouts: int = 8,
                survivors: int = 2, select_at: int = None,
                threshold: float = 0.0, batch_size: int = 500,
                seed: int = 0, log=print) -> dict:
    """
    The whole depth-by-noise grid, one noise level at a time.

    Returns the summarised results and the raw per-example arrays, which
    :func:`write_sweep` records beside the provenance of the run.
    """
    caps, sigmas = tuple(sorted(caps)), tuple(sigmas)
    select_at = caps[0] if select_at is None else select_at
    device = next(model.parameters()).device
    results = {cap: {} for cap in caps}
    raw: dict = {}
    seeds: dict = {}
    for sigma in sigmas:
        stacked = {cap: {key: [] for key in RECORD_KEYS} for cap in caps}
        for index, start in enumerate(range(0, len(split), batch_size)):
            stop = start + batch_size
            clues = torch.as_tensor(
                split.puzzles[start:stop], dtype=torch.long, device=device)
            target = torch.as_tensor(
                split.solutions[start:stop], dtype=torch.long, device=device)
            chunk = sweep_chunk(
                model, clues, target, sigma, index, caps, rollouts,
                survivors, select_at, threshold, seed, seeds, log=log)
            for cap in caps:
                for key in RECORD_KEYS:
                    stacked[cap][key].append(chunk[cap][key].cpu().numpy())
        for cap in caps:
            arrays = {key: np.concatenate(stacked[cap][key], axis=1)
                      for key in RECORD_KEYS}
            results[cap][sigma] = summarize(arrays)
            for key, value in arrays.items():
                raw[f"cap{cap}_sigma{sigma:g}_{key}"] = value
    return {"results": results, "raw": raw, "seeds": seeds,
            "caps": caps, "sigmas": sigmas,
            "protocol": {"rollouts": rollouts, "survivors": survivors,
                         "select_at": select_at, "threshold": threshold,
                         "n_eval": len(split), "seed": seed}}


def write_sweep(payload: dict, stem: str, meta: dict = None) -> tuple:
    """
    Record a sweep under ``artifacts/``: the summary and the provenance as
    JSON, the per-example arrays as an ``npz``.

    *How* a sweep was produced belongs beside *what* was asked of it: eager
    and compiled agree only up to the rounding freedom
    :meth:`~discopy.neural.CMap.compile` documents, so the torch version
    and the device are recorded with the numbers.
    """
    summary = {key: value for key, value in payload.items() if key != "raw"}
    summary["results"] = {
        str(cap): {str(sigma): value for sigma, value in row.items()}
        for cap, row in payload["results"].items()}
    summary["environment"] = {
        "torch": torch.__version__,
        "device": str(torch.cuda.get_device_name(0))
        if torch.cuda.is_available() else "cpu",
        "float32_matmul_precision": torch.get_float32_matmul_precision()}
    summary["model"] = meta or {}
    (ARTIFACTS / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(ARTIFACTS / f"{stem}.npz", **payload["raw"])
    return ARTIFACTS / f"{stem}.json", ARTIFACTS / f"{stem}.npz"


# --- loading a trained model ----------------------------------------------

def load(path, name: str = "act", widths: Widths = None, rounds: int = None,
         cycles: int = None, steps: int = None, device=None,
         halt_head: str = "softmin", halt_detach: bool = True):
    """
    Rebuild a model from a checkpoint, taking its shape from the record
    when the checkpoint carries one.

    Search checkpoints store their trial under ``params``, the recipes of
    ``train.py`` under ``hyperparameters``; both also store ``widths``.
    """
    path = Path(path)
    stored = torch.load(path, map_location="cpu", weights_only=False)
    params = stored.get("params") or stored.get("hyperparameters") or {}
    shape = {"rounds": rounds or params.get("n") or params.get("rounds"),
             "cycles": cycles or params.get("T") or params.get("cycles"),
             "steps": steps or params.get("n_sup") or params.get("steps")}
    if widths is None:
        widths = Widths(**stored["widths"]) if "widths" in stored else None
    kwargs = {key: value for key, value in shape.items() if value}
    if name == "act":
        kwargs.update(halt_head=halt_head, halt_detach=halt_detach)
    model = zoo.BUILDERS[name](widths=widths, **kwargs)
    zoo.load_checkpoint(model, path)
    meta = {key: value for key, value in stored.items()
            if key not in ("state_dict", "history")}
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model.to(device or (
        "cuda" if torch.cuda.is_available() else "cpu")).eval(), meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model", default="act",
                        choices=("goi", "rrn", "trm", "act"))
    parser.add_argument("--extreme", action="store_true",
                        help="score on sudoku-extreme rather than Palm-2018")
    parser.add_argument("--variant", default="special_large")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--sweep", type=int, nargs="*", default=None,
                        help="test-time compute values to sweep")
    parser.add_argument("--noise", action="store_true",
                        help="run the depth-by-noise study")
    parser.add_argument("--caps", type=int, nargs="+", default=list(CAPS))
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=list(SIGMAS))
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--survivors", type=int, default=2)
    parser.add_argument("--select-at", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stem", default=None)
    arguments = parser.parse_args(argv)

    torch.set_float32_matmul_precision("high")
    model, meta = load(arguments.checkpoint, arguments.model)
    splits = dataset.load_extreme(arguments.variant) if arguments.extreme \
        else dataset.load()
    split = splits[arguments.split].subsample(arguments.n_eval)
    print(f"{arguments.model} on {arguments.split} "
          f"({len(split):,} puzzles), sha {meta['sha256'][:12]}")

    scores = evaluate(model, split)
    print(f"fixed   cell {scores['cell']:.4f}  board {scores['board']:.4f}")
    if arguments.sweep:
        for row in sweep_compute(model, split, arguments.sweep):
            print(f"  at {row['compute']:4d}: board {row['board']:.4f}")
    if arguments.model == "act":
        adaptive = evaluate_act(model, split)
        print(f"act     cell {adaptive['cell']:.4f}  "
              f"board {adaptive['board']:.4f}  depth {adaptive['depth']:.2f}")
        print("latent  " + ", ".join(
            f"{name} rms {stat['rms']:.3f}"
            for name, stat in latent_stats(
                model, torch.as_tensor(
                    split.puzzles[:256], dtype=torch.long,
                    device=next(model.parameters()).device)).items()))
    if arguments.noise:
        payload = noise_sweep(
            model, split, caps=arguments.caps, sigmas=arguments.sigmas,
            rollouts=arguments.rollouts, survivors=arguments.survivors,
            select_at=arguments.select_at, seed=arguments.seed)
        stem = arguments.stem or f"noise-{arguments.checkpoint.stem}"
        print("\nboard, fixed compute, best-of-k selected by the halt logit:")
        print(table(payload["results"], payload["caps"], payload["sigmas"],
                    "fixed", "best_of_k"))
        print("\nboard, adaptive compute, single rollout:")
        print(table(payload["results"], payload["caps"], payload["sigmas"],
                    "act", "single"))
        print("\nwrote " + ", ".join(
            str(path) for path in write_sweep(payload, stem, meta)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
