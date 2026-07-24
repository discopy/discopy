# -*- coding: utf-8 -*-

"""
Inference-time latent noise injection for the pretrained ACT solver on
sudoku-extreme.  No parameter is modified or retrained: the checkpoint is
evaluated as-is, in eval mode, under ``no_grad``, with Gaussian noise
added to its persistent high-level state at the start of every deep
supervision step.

    CUDA_VISIBLE_DEVICES=2 python eval_noise_trm_act.py

**The perturbed state.**  The solver (:class:`experiments.act.ACTSolver`,
the tiny-recursive-model recursion on the factor-graph map) carries one
flat message tensor between supervision steps; inside it live three
recurrent families:

* the *answer* ``y`` -- one ``y_dim`` vector per cell on the traced
  answer loop, refreshed **once per cycle** by the answer ``GRUCell`` and
  read out by the linear head.  This is the persistent high-level
  reasoning state, the exact analogue of the HRM/TRM high-level ``y``/
  ``z_H``;
* the *latent* ``z`` -- one ``state_dim`` vector per cell on the state
  loop, updated every message-passing round: the low-level state;
* the clue loop and the cell-unit message wires: inputs and temporaries.

The primary experiment perturbs **only** ``y``: at the beginning of each
deep-supervision step, immediately before the step's ``T`` cycles run,

    y <- y + eps,   eps ~ Normal(0, sigma^2 I)

sampled independently for every example, rollout, step, cell and feature.
The noise tensor has the shape, dtype and device of ``y`` itself, i.e.
``(batch, 81, y_dim)``.  Two masking/sharing notes, recorded here as part
of the protocol: sudoku has no padded nodes -- all 81 cells are real --
so the valid-node mask is all-ones (kept explicit in the code); and the
two ports of each cell's answer loop intentionally carry the *same*
``y`` (the model writes it duplicated), so one noise sample per cell is
written to both loop ends, preserving that sharing rather than breaking
the loop's consistency.

**Exact injection point.**  ``y_{t-1}`` is written by the previous step's
answer ``GRUCell`` *after* its ``LayerNorm`` (``answer_norm``): the
architecture defines the persistent state post-normalisation, and the
recurrent block reads it raw (no further normalisation between the
carried state and the first cycle).  Noise is therefore added to the
post-norm carried state, in :func:`perturb_answer`, and nowhere else:
not to the raw clues, the wiring, the targets, the logits, the weights,
or any tensor recomputed within one step.  Because ``answer_norm`` makes
``y`` unit-scale per cell (RMS ~ 1, logged by :func:`latent_stats`),
``sigma`` reads directly as a relative noise level.

**Protocol.**  Fixed evaluation examples (a deterministic prefix of the
held-out validation split) for every noise setting; independent
stochastic rollouts with recorded, independent seeds; one noise draw per
supervision step -- never per message-passing round, and never reused
across steps or rollouts.  A rollout is one continuous trajectory scored
at every cap in ``caps`` as it passes it, both

* *fixed-compute*: the answer at exactly ``cap`` steps, and
* *adaptive (ACT)*: the answer the model would have halted with under
  the paper's early stopping (first step with halt logit > threshold),
  capped at ``cap``, with its mean depth.

**Beam schedule.**  Running every rollout to the deepest cap is
wasteful, so depth is spent where the fan-out matters: per noise level,
``--rollouts`` (K) independent trajectories run to ``--select-at`` steps
(the shallowest cap), then per *example* the ``--survivors`` (C)
trajectories with the largest halt logit -- the model's own confidence,
the only signal available at test time -- continue to ``max(caps)``,
drawing fresh noise all the way.  Shallow caps are therefore scored over
all K rollouts, deep caps over the C selected survivors, i.e. deep-cap
numbers are best-of-K-start, best-of-C-continuation.  ``sigma = 0`` is
deterministic and runs once.

Reported per (cap, sigma): the mean single-rollout board accuracy, the
best-of-k accuracy with the rollout **selected by the halt logit**, and,
as a diagnostic upper bound only, the oracle pass@k, where k is the
number of trajectories alive at that cap (K shallow, C deep).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# the search protocol runs under TF32; evaluation matches it.
torch.set_float32_matmul_precision("high")

from experiments import sudoku_extreme                        # noqa: E402
from experiments.act import ACTSolver                         # noqa: E402
from experiments.config import ARTIFACTS, FIGURES, Widths     # noqa: E402
from experiments.train import decode                          # noqa: E402

#: The widths of the extreme searches (optuna_trm_extreme.SEARCH_WIDTHS).
SEARCH_WIDTHS = Widths(dim=72, state_dim=192, hidden=384, y_dim=96)

#: The deep-supervision depths (rows) and noise levels (columns).
CAPS = (16, 32, 64, 128, 256)
SIGMAS = (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)

#: The checkpoint of the first model the ACT search produced: study
#: ``trm-extreme-act-8k``, trial 2, valid board 0.592 (act 0.592 @ 8.4).
CHECKPOINT = ARTIFACTS / "optuna-trm-extreme-act-8k-trial2.pt"

#: The per-example bookkeeping a trajectory carries besides its state.
BOOK_KEYS = ("halted", "halt_step", "halt_correct", "halt_cell", "halt_q")

#: What is recorded at every cap.
RECORD_KEYS = ("fixed_correct", "fixed_cell", "fixed_q",
               "act_correct", "act_cell", "act_q", "act_depth")


def load_model(path: Path, device) -> tuple[ACTSolver, dict]:
    """ The pretrained :class:`ACTSolver` rebuilt from a search checkpoint. """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    params = stored["params"]
    model = ACTSolver(
        SEARCH_WIDTHS, rounds=params["n"], cycles=params["T"],
        n_sup=params["n_sup"], halt_detach=True, halt_head="softmin")
    model.load_state_dict(stored["state_dict"])
    model.to(device).eval()
    meta = {key: value for key, value in stored.items()
            if key != "state_dict"}
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, meta


def valid_node_mask(shape, device, dtype) -> torch.Tensor:
    """
    The mask zeroing noise on padded or invalid nodes.  Sudoku has none:
    all 81 cells are real nodes, so this is all-ones -- kept explicit so
    the protocol's masking step is visible in the code.
    """
    return torch.ones(shape, device=device, dtype=dtype)


def perturb_answer(model: ACTSolver, state, sigma: float,
                   generator) -> torch.Tensor:
    """
    The carried state with ``y <- y + eps`` on the answer loop, the noise
    sampled independently per example, cell and feature with ``generator``
    and written to both loop ends, which intentionally share one ``y``.

    Parameters:
        model : The solver, for its router and answer ports.
        state : The flat carried messages, ``(batch, router.total)``.
        sigma : The noise standard deviation.
        generator : The rollout's own ``torch.Generator``.
    """
    y = model.router.read(state, model.answer_ports)[:, ::2]
    eps = sigma * torch.randn(
        y.shape, generator=generator, device=y.device, dtype=y.dtype)
    eps = eps * valid_node_mask(y.shape, y.device, y.dtype)
    return model.router.write(
        state, model.answer_ports, (y + eps).repeat_interleave(2, dim=1))


@torch.no_grad()
def latent_stats(model: ACTSolver, clues, steps: int = 16) -> dict:
    """
    Statistics of the deterministic carried states over the evaluation
    examples, pooled over ``steps`` supervision steps: global mean,
    per-feature standard deviation (its mean/min/max), root-mean-square
    magnitude and mean per-cell L2 norm, for the answer ``y`` (the
    perturbed state) and, for reference, the latent ``z``.
    """
    def accumulate(totals, tensor):
        flat = tensor.reshape(-1, tensor.shape[-1]).double()
        totals["count"] += flat.shape[0]
        totals["sum"] += flat.sum(0)
        totals["sumsq"] += (flat ** 2).sum(0)
        totals["norm"] += flat.norm(dim=-1).sum()

    def zeros(width):
        return {"count": 0, "sum": torch.zeros(width, dtype=torch.double,
                                               device=clues.device),
                "sumsq": torch.zeros(width, dtype=torch.double,
                                     device=clues.device),
                "norm": torch.zeros((), dtype=torch.double,
                                    device=clues.device)}

    totals = {"y": zeros(model.widths.y_dim),
              "z": zeros(model.widths.state_dim)}
    torch.compiler.cudagraph_mark_step_begin()
    state = model.initial(clues)
    for _ in range(steps):
        torch.compiler.cudagraph_mark_step_begin()
        state, _, _ = model.act_step(state, grad=False)
        state = state.detach().clone()
        accumulate(totals["y"],
                   model.router.read(state, model.answer_ports)[:, ::2])
        accumulate(totals["z"],
                   model.router.read(state, model.state_ports))

    result = {}
    for name, total in totals.items():
        mean = total["sum"] / total["count"]
        var = total["sumsq"] / total["count"] - mean ** 2
        std = var.clamp(min=0).sqrt()
        rms = (total["sumsq"].sum() / (total["count"] * len(mean))).sqrt()
        result[name] = {
            "global_mean": mean.mean().item(),
            "per_feature_std_mean": std.mean().item(),
            "per_feature_std_min": std.min().item(),
            "per_feature_std_max": std.max().item(),
            "rms": rms.item(),
            "mean_norm": (total["norm"] / total["count"]).item(),
            "per_feature_std": std.cpu().numpy()}
    return result


def fresh_book(model: ACTSolver, clues) -> dict:
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


def assemble(snapshots: list[dict], choice) -> dict:
    """
    One continuation trajectory gathered per example across phase-one
    snapshots: example ``i`` resumes the trajectory of rollout
    ``choice[i]``, state and halting bookkeeping alike.

    Parameters:
        snapshots : The books of the phase-one rollouts, all at one step.
        choice : The rollout index each example continues, ``(batch, )``.
    """
    book = {"t": snapshots[0]["t"],
            "state": snapshots[0]["state"].clone(), **{
                key: snapshots[0][key].clone() for key in BOOK_KEYS}}
    for k, snap in enumerate(snapshots[1:], start=1):
        mask = choice == k
        if not bool(mask.any()):
            continue
        book["state"][mask] = snap["state"][mask]
        for key in BOOK_KEYS:
            book[key][mask] = snap[key][mask]
    return book


@torch.no_grad()
def run_segment(model: ACTSolver, clues, target, sigma: float, caps,
                generator, threshold: float, book: dict,
                t_stop: int) -> dict:
    """
    Advance one trajectory from its current step to ``t_stop``, injecting
    noise once at the start of every supervision step and scoring every
    cap it passes.

    Returns, per cap: ``fixed_correct``/``fixed_cell``/``fixed_q`` (the
    answer at exactly that many steps, its per-board and per-cell
    accuracy and its halt logit) and ``act_correct``/``act_cell``/
    ``act_q``/``act_depth`` (the answer under the paper's early stopping
    capped there), all as device tensors.
    """
    caps = set(caps)
    out = {}
    for t in range(book["t"] + 1, t_stop + 1):
        state = book["state"]
        if sigma > 0:
            state = perturb_answer(model, state, sigma, generator)
        torch.compiler.cudagraph_mark_step_begin()
        state, logits, halt = model.act_step(state, grad=False)
        book["state"] = state.detach().clone()
        q = model.halt_logit(halt)
        matches = decode(logits, clues) == target
        correct, cell = matches.all(-1), matches.float().mean(-1)
        newly = ~book["halted"] & (q > threshold)
        book["halt_step"] = torch.where(
            newly, torch.full_like(book["halt_step"], t), book["halt_step"])
        book["halt_correct"] = torch.where(
            newly, correct, book["halt_correct"])
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


def sweep_chunk(model: ACTSolver, clues, target, sigma: float, si: int,
                arguments, seeds: dict, log=print) -> dict:
    """
    The whole beam schedule of one noise level on one batch of examples:
    K phase-one rollouts to the selection depth, per-example selection of
    the C most confident, their continuation to the deepest cap.

    Returns ``{cap: {key: tensor of shape (k_alive, batch)}}``.
    """
    caps = sorted(arguments.caps)
    select_at, max_cap = arguments.select_at, caps[-1]
    threshold = arguments.halt_threshold
    device = clues.device

    def make_generator(label, index):
        seed = int(np.random.SeedSequence(
            [arguments.seed, si, index if label == "start" else 10 ** 6 + index
             ]).generate_state(1)[0])
        seeds[f"sigma{sigma:g}_{label}{index}"] = seed
        return torch.Generator(device=device).manual_seed(seed)

    if sigma == 0:
        book = fresh_book(model, clues)
        out = run_segment(model, clues, target, sigma, caps, None,
                          threshold, book, max_cap)
        return {cap: {key: out[cap][key][None] for key in RECORD_KEYS}
                for cap in caps}

    shallow = [cap for cap in caps if cap <= select_at]
    deep = [cap for cap in caps if cap > select_at]
    records: dict = {cap: {key: [] for key in RECORD_KEYS} for cap in caps}
    snapshots, confidence = [], []
    tick = time.perf_counter()
    for k in range(arguments.rollouts):
        generator = make_generator("start", k)
        book = fresh_book(model, clues)
        out = run_segment(model, clues, target, sigma, shallow, generator,
                          threshold, book, select_at)
        snapshots.append(book)
        confidence.append(out[select_at]["fixed_q"])
        for cap in shallow:
            for key in RECORD_KEYS:
                records[cap][key].append(out[cap][key])
    shallow_board = torch.stack(
        records[select_at]["fixed_correct"]).float().mean()
    log(f"  sigma {sigma:g}: {arguments.rollouts} rollouts @ {select_at}  "
        f"board {shallow_board:.4f}  "
        f"({time.perf_counter() - tick:.0f}s)", flush=True)

    # per example, the C most confident trajectories continue; the halt
    # logit is the model's own confidence, available at test time.
    ranks = torch.stack(confidence).argsort(0, descending=True)
    for r in range(arguments.survivors):
        book = assemble(snapshots, ranks[r])
        generator = make_generator("continue", r)
        out = run_segment(model, clues, target, sigma, deep, generator,
                          threshold, book, max_cap)
        for cap in deep:
            for key in RECORD_KEYS:
                records[cap][key].append(out[cap][key])
        deep_board = out[max_cap]["fixed_correct"].float().mean()
        log(f"  sigma {sigma:g}: survivor {r + 1}/{arguments.survivors} "
            f"@ {max_cap}  board {deep_board:.4f}  "
            f"({time.perf_counter() - tick:.0f}s)", flush=True)
    del snapshots
    return {cap: {key: torch.stack(records[cap][key])
                  for key in RECORD_KEYS} for cap in caps}


def summarize(arrays: dict) -> dict:
    """
    The board metrics of one (cap, sigma) box from its stacked arrays of
    shape ``(k_alive, n_eval)``:

    * ``single`` : mean single-rollout accuracy over the k trajectories;
    * ``best_of_k`` : per example, the trajectory the model itself is most
      confident in (largest halt logit -- test-time information only);
    * ``pass_at_k`` : any trajectory solved the board (oracle diagnostic).
    """
    def block(prefix):
        correct = arrays[f"{prefix}_correct"]
        q = arrays[f"{prefix}_q"]
        chosen = q.argmax(0)
        columns = np.arange(correct.shape[1])
        return {
            "single": float(correct.mean()),
            "single_cell": float(arrays[f"{prefix}_cell"].mean()),
            "best_of_k": float(correct[chosen, columns].mean()),
            "pass_at_k": float(correct.any(0).mean())}
    result = {"k": int(arrays["fixed_correct"].shape[0]),
              "fixed": block("fixed"), "act": block("act")}
    result["act"]["depth"] = float(arrays["act_depth"].mean())
    return result


def table(results: dict, caps, sigmas, protocol: str, metric: str) -> str:
    """ One caps-by-sigmas table of ``results`` as aligned text. """
    header = "steps".rjust(6) + "".join(
        f"{sigma:>10g}" for sigma in sigmas)
    lines = [header]
    for cap in caps:
        lines.append(f"{cap:>6d}" + "".join(
            f"{results[cap][sigma][protocol][metric]:>10.4f}"
            for sigma in sigmas))
    return "\n".join(lines)


# --- figures ---------------------------------------------------------------

#: Chart chrome: ink, hairlines and surface of the validated palette.
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "baseline": "#c3c2b7", "surface": "#fcfcfb"}

#: One-hue ordinal ramp for the supervision depths, light -> dark
#: (validated: monotone lightness, visible gaps, light end >= 2:1).
DEPTH_RAMP = ("#86b6ef", "#75a0d3", "#648ab8", "#51729b", "#395478",
              "#041c40")

#: Categorical slots for the three evaluation protocols (validated
#: adjacent-pair CVD >= 8; the aqua slot is sub-3:1 on the light surface,
#: so the lines carry visible direct labels -- the relief rule).
PROTOCOL_HUES = {"best_of_k": "#2a78d6", "single": "#eb6834",
                 "pass_at_k": "#1baf7a"}

#: Diverging blue <- gray -> red for the gain heatmap (blue = noise wins).
DIVERGING = ("#d03b3b", "#f0efec", "#104281")


def _axes(ax):
    """ Recessive chart chrome: hairline grid, no top or right spine. """
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["baseline"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK["muted"], labelsize=8, width=0.8)
    ax.grid(axis="y", color=INK["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK["secondary"])
    ax.yaxis.label.set_color(INK["secondary"])


def _sigma_axis(ax, sigmas):
    """ Noise on a symlog axis, so sigma = 0 keeps an honest position. """
    positive = [sigma for sigma in sigmas if sigma > 0]
    ax.set_xscale("symlog", linthresh=min(positive), linscale=0.4)
    ax.set_xticks([0.0] + positive)
    ax.set_xticklabels(["0"] + [f"{sigma:g}" for sigma in positive])
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("noise standard deviation $\\sigma$")


def _save(figure, stem: str, name: str) -> list:
    paths = []
    for extension in ("pdf", "png"):
        path = FIGURES / f"{stem}-{name}.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight",
                       facecolor=INK["surface"])
        paths.append(path)
    return paths


def plot_results(payload: dict, stem: str) -> list:
    """
    The three summary figures, from the results dictionary the sweep saves:

    1. board accuracy against noise level, one line per depth
       (fixed-compute, mean single-rollout);
    2. what selection buys at the deepest cap: mean single-rollout,
       best-of-k by halt logit, and the oracle pass@k;
    3. the noise gain over the deterministic solver, per depth and noise
       level (best-of-k minus the sigma = 0 baseline of the same depth).

    Parameters:
        payload : The saved results, as in the sweep's JSON.
        stem : The filename stem of the figures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="white", font="sans-serif", rc={
        "text.color": INK["primary"], "axes.titlesize": 10,
        "axes.titleweight": "semibold", "axes.labelsize": 9,
        "figure.facecolor": INK["surface"]})
    caps = [int(cap) for cap in payload["caps"]]
    sigmas = [float(sigma) for sigma in payload["sigmas"]]
    noisy = [sigma for sigma in sigmas if sigma > 0]

    def value(cap, sigma, protocol, metric):
        return payload["results"][str(cap)][f"{sigma:g}"][protocol][metric]

    note = (f"{payload['n_eval']} fixed {payload.get('split', 'valid')} "
            f"puzzles; K = {payload['rollouts']} "
            f"rollouts to step {payload['select_at']}, the "
            f"{payload['survivors']} most confident (halt logit) continue")
    paths = []

    # 1. accuracy against noise, one line per depth.
    figure, ax = plt.subplots(figsize=(4.4, 3.2))
    _axes(ax)
    _sigma_axis(ax, sigmas)
    for cap, hue in zip(caps, DEPTH_RAMP):
        ax.plot(sigmas, [value(cap, s, "fixed", "single") for s in sigmas],
                color=hue, linewidth=2, marker="o", markersize=4.5,
                markeredgecolor=INK["surface"], markeredgewidth=0.8,
                label=str(cap), zorder=3)
    ax.set_ylabel("board accuracy")
    ax.set_title("Latent noise against depth, mean single rollout")
    legend = ax.legend(title="supervision steps", frameon=False, fontsize=8,
                       title_fontsize=8, labelcolor=INK["secondary"],
                       handlelength=1.4, borderaxespad=0.2)
    legend.get_title().set_color(INK["secondary"])
    figure.text(0.01, -0.04, note, fontsize=6.5, color=INK["muted"])
    paths += _save(figure, stem, "accuracy")
    plt.close(figure)

    # 2. selection at the deepest cap.
    deepest = caps[-1]
    labels = {"single": "mean single rollout",
              "best_of_k": "best-of-k, halt-logit selection",
              "pass_at_k": "pass@k (oracle)"}
    figure, ax = plt.subplots(figsize=(4.4, 3.2))
    _axes(ax)
    _sigma_axis(ax, sigmas)
    curves = {metric: [value(deepest, s, "fixed", metric) for s in sigmas]
              for metric in PROTOCOL_HUES}
    for metric, hue in PROTOCOL_HUES.items():
        ax.plot(sigmas, curves[metric], color=hue, linewidth=2,
                linestyle="--" if metric == "pass_at_k" else "-",
                marker="o", markersize=4.5,
                markeredgecolor=INK["surface"], markeredgewidth=0.8,
                label=labels[metric], zorder=3)
    # direct labels at the line ends, dodged apart when curves coincide.
    span = max(max(c) for c in curves.values()) \
        - min(min(c) for c in curves.values()) or 1.0
    ends = sorted(((c[-1], metric) for metric, c in curves.items()))
    slot = None
    for y, metric in ends:
        slot = y if slot is None else max(y, slot + 0.06 * span)
        ax.annotate(labels[metric], (sigmas[-1], slot),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, color=INK["secondary"], va="center",
                    annotation_clip=False)
    ax.set_ylabel("board accuracy")
    ax.set_title(f"Selection among noisy rollouts at {deepest} steps")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK["secondary"],
              handlelength=1.6, loc="lower left")
    figure.text(0.01, -0.04, note, fontsize=6.5, color=INK["muted"])
    paths += _save(figure, stem, "selection")
    plt.close(figure)

    # 3. the gain heatmap: best-of-k against the deterministic baseline.
    gain = pd.DataFrame(
        [[value(cap, sigma, "fixed", "best_of_k")
          - value(cap, 0.0, "fixed", "single") for sigma in noisy]
         for cap in caps],
        index=caps, columns=[f"{sigma:g}" for sigma in noisy])
    bound = float(gain.abs().max().max()) or 1e-3
    figure, ax = plt.subplots(figsize=(5.0, 3.0))
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "gain", DIVERGING)
    sns.heatmap(
        gain, ax=ax, cmap=colormap, vmin=-bound, vmax=bound, center=0,
        annot=True, fmt="+.3f", annot_kws={"fontsize": 7},
        linewidths=2, linecolor=INK["surface"],
        cbar_kws={"label": "$\\Delta$ board accuracy", "pad": 0.02})
    for text in ax.texts:
        text.set_color(INK["surface"]
                       if abs(float(text.get_text())) > 0.6 * bound
                       else INK["primary"])
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(colors=INK["muted"], labelsize=7)
    colorbar.ax.yaxis.label.set_color(INK["secondary"])
    colorbar.outline.set_visible(False)
    ax.tick_params(colors=INK["muted"], labelsize=8, length=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("noise standard deviation $\\sigma$",
                  color=INK["secondary"], fontsize=9)
    ax.set_ylabel("supervision steps", color=INK["secondary"], fontsize=9)
    ax.set_title("Noise gain over the deterministic solver",
                 fontsize=10, fontweight="semibold")
    figure.text(0.01, -0.04, note + "; baseline: $\\sigma = 0$, same depth",
                fontsize=6.5, color=INK["muted"])
    paths += _save(figure, stem, "gain")
    plt.close(figure)

    # 4. test-time scaling: accuracy against depth at the strongest noise.
    if noisy:
        top = noisy[-1]
        figure, ax = plt.subplots(figsize=(4.4, 3.2))
        _axes(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(caps)
        ax.set_xticklabels([str(cap) for cap in caps])
        from matplotlib.ticker import NullLocator
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("deep-supervision steps")
        curves = {label: ([value(cap, sigma, "fixed", metric)
                           for cap in caps], hue, style)
                  for label, sigma, metric, hue, style in (
                      ("best-of-k, halt-logit selection", top, "best_of_k",
                       PROTOCOL_HUES["best_of_k"], "-"),
                      ("pass@k (oracle)", top, "pass_at_k",
                       PROTOCOL_HUES["pass_at_k"], "--"),
                      ("mean single rollout", top, "single",
                       PROTOCOL_HUES["single"], "-"))}
        if 0.0 in sigmas:
            curves["deterministic ($\\sigma = 0$)"] = (
                [value(cap, 0.0, "fixed", "single") for cap in caps],
                INK["muted"], "--")
        for label, (series, hue, style) in curves.items():
            ax.plot(caps, series, color=hue, linewidth=2, linestyle=style,
                    marker="o", markersize=4.5,
                    markeredgecolor=INK["surface"], markeredgewidth=0.8,
                    label=label, zorder=3)
        span = max(max(c) for c, _, _ in curves.values()) \
            - min(min(c) for c, _, _ in curves.values()) or 1.0
        slot = None
        for y, label in sorted((c[-1], label)
                               for label, (c, _, _) in curves.items()):
            slot = y if slot is None else max(y, slot + 0.06 * span)
            ax.annotate(label, (caps[-1], slot),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=7.5, color=INK["secondary"], va="center",
                        annotation_clip=False)
        ax.set_ylabel("board accuracy")
        ax.set_title(f"Test-time scaling with latent noise "
                     f"($\\sigma = {top:g}$)")
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK["secondary"],
                  handlelength=1.6, loc="lower right")
        figure.text(0.01, -0.04, note, fontsize=6.5, color=INK["muted"])
        paths += _save(figure, stem, "depth")
        plt.close(figure)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--split", default="valid",
                        choices=["valid", "test"],
                        help="the split to evaluate on; the test split is "
                             "never touched by the searches")
    parser.add_argument("--n-eval", type=int, default=2000,
                        help="fixed evaluation examples, a deterministic "
                             "prefix of the chosen split, identical "
                             "for every noise setting")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--rollouts", type=int, default=32,
                        help="independent stochastic rollouts K per noise "
                             "level, run to --select-at steps (sigma = 0 "
                             "is deterministic, one)")
    parser.add_argument("--survivors", type=int, default=4,
                        help="trajectories per example continued to the "
                             "deepest cap, selected by halt logit")
    parser.add_argument("--select-at", type=int, default=16,
                        help="the depth of the phase-one fan-out and of "
                             "the survivor selection; must be a cap")
    parser.add_argument("--caps", type=int, nargs="+", default=list(CAPS))
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=list(SIGMAS))
    parser.add_argument("--seed", type=int, default=0,
                        help="master seed; each (sigma, rollout) derives "
                             "its own recorded, independent seed from it")
    parser.add_argument("--halt-threshold", type=float, default=0.0,
                        help="the margin the halt logit must clear, as in "
                             "training (the paper's q > 0)")
    parser.add_argument("--stats-steps", type=int, default=16,
                        help="supervision steps pooled by the deterministic "
                             "latent-state statistics")
    parser.add_argument("--compile", default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--device", default="cuda"
                        if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None,
                        help="output stem, default artifacts/noise-eval-"
                             "<checkpoint stem>")
    parser.add_argument("--plot-only", action="store_true",
                        help="regenerate the figures from the saved JSON "
                             "of a finished (or partial) sweep and exit")
    arguments = parser.parse_args(argv)
    assert arguments.select_at in arguments.caps, \
        "--select-at must be one of --caps"
    assert arguments.survivors <= arguments.rollouts
    device = torch.device(arguments.device)
    out = arguments.out or ARTIFACTS / (
        "noise-eval-" + arguments.checkpoint.stem.replace("optuna-", ""))

    if arguments.plot_only:
        payload = json.loads(out.with_suffix(".json").read_text())
        for path in plot_results(payload, out.stem):
            print(f"saved {path}")
        return 0

    model, meta = load_model(arguments.checkpoint, device)
    print(f"checkpoint {arguments.checkpoint.name}: "
          + ", ".join(f"{k}={v}" for k, v in meta.items()
                      if k not in ("sha256",)))
    if arguments.compile:
        model.compile_cells(mode=arguments.compile_mode)

    splits = sudoku_extreme.load("special_large")
    split = splits[arguments.split].subsample(arguments.n_eval)
    batches = [
        (torch.as_tensor(split.puzzles[start:start + arguments.batch_size],
                         dtype=torch.long, device=device),
         torch.as_tensor(split.solutions[start:start + arguments.batch_size],
                         dtype=torch.long, device=device))
        for start in range(0, len(split), arguments.batch_size)]

    stats = latent_stats(model, batches[0][0], steps=arguments.stats_steps)
    for name, stat in stats.items():
        print(f"deterministic {name}: mean {stat['global_mean']:+.4f}  "
              f"rms {stat['rms']:.4f}  mean_norm {stat['mean_norm']:.3f}  "
              f"feature_std {stat['per_feature_std_mean']:.4f} "
              f"[{stat['per_feature_std_min']:.4f}, "
              f"{stat['per_feature_std_max']:.4f}]")

    caps, sigmas = sorted(arguments.caps), list(arguments.sigmas)
    results: dict = {cap: {} for cap in caps}
    raw: dict = {}
    seeds: dict = {}
    for si, sigma in enumerate(sigmas):
        chunks = [
            sweep_chunk(model, clues, target, sigma, si, arguments, seeds)
            for clues, target in batches]
        for cap in caps:
            arrays = {
                key: np.concatenate([
                    chunk[cap][key].cpu().numpy() for chunk in chunks],
                    axis=1)
                for key in RECORD_KEYS}
            results[cap][sigma] = summarize(arrays)
            for key in RECORD_KEYS:
                raw[f"cap{cap}_sigma{sigma:g}_{key}"] = arrays[key]
        # checkpoint the sweep after every sigma, so a kill loses little.
        np.savez_compressed(out.with_suffix(".npz"), **raw)
        payload = {
            "checkpoint": str(arguments.checkpoint), "meta": {
                k: v for k, v in meta.items() if k != "state_dict"},
            "split": arguments.split,
            "n_eval": len(split), "rollouts": arguments.rollouts,
            "survivors": arguments.survivors,
            "select_at": arguments.select_at,
            "caps": caps, "sigmas": sigmas[:si + 1],
            "master_seed": arguments.seed, "seeds": seeds,
            "halt_threshold": arguments.halt_threshold,
            "injection_point": "answer loop y (post answer_norm), once "
                               "per supervision step, before the step's "
                               "first cycle",
            "latent_stats": {
                name: {k: v for k, v in stat.items()
                       if k != "per_feature_std"}
                for name, stat in stats.items()},
            "results": {
                str(cap): {f"{s:g}": results[cap][s]
                           for s in results[cap]} for cap in caps}}
        out.with_suffix(".json").write_text(json.dumps(payload, indent=2))

    for protocol, metric, title in (
            ("fixed", "single", "fixed-compute, mean single-rollout"),
            ("fixed", "best_of_k", "fixed-compute, best-of-k by halt "
                                   "logit (test-time selection)"),
            ("fixed", "pass_at_k", "fixed-compute, oracle pass@k"),
            ("act", "single", "adaptive ACT, mean single-rollout"),
            ("act", "best_of_k", "adaptive ACT, best-of-k by halt logit"),
            ("act", "pass_at_k", "adaptive ACT, oracle pass@k"),
            ("act", "depth", "adaptive ACT, mean halting depth")):
        print(f"\nboard accuracy -- {title}"
              if metric != "depth" else f"\n{title}")
        print(table(results, caps, sigmas, protocol, metric))
    print(f"\nsaved {out.with_suffix('.json')} and "
          f"{out.with_suffix('.npz')}")
    for path in plot_results(payload, out.stem):
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
