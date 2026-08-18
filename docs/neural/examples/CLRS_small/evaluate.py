# -*- coding: utf-8 -*-

"""
The evaluation protocol: in-distribution and out-of-distribution scores,
the test-time depth sweep, and the residual of every promise the diagram
makes.

    python evaluate.py --quick
    python evaluate.py --algorithms bfs

Three things are measured, and they are different questions:

* **the score** -- CLRS's own, per output probe and averaged over them,
  pooled over a whole split rather than averaged over its batches, because
  an F1 is not linear;
* **the depth sweep** -- the same weights run for more rounds than they
  were trained at, which is the cheapest test-time compute there is and
  the one Part 3 turns into a claim;
* **the promises** -- :func:`~discopy.neural.check_equivariant` on every
  cell and :meth:`~discopy.neural.map.Interaction.residual` on the state a
  run ends at.  A permutation-equivariant cell is a *measured* residual,
  not an assumption, and whether the learned transition settles is a
  property of the weights that the category never supplies.

Every number lands in a JSON artefact beside its provenance -- the device,
the torch version, the widths, the depth -- because a score of a message
passer is only reproducible up to the rounding freedom its fused forward
documents.
"""

from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from dataclasses import replace
from itertools import combinations, permutations

import numpy as np
import torch

import dataset
import model as zoo
import train as training
from config import (
    ALGORITHMS, ANCHORS, ANCHOR_SOURCE, ARTIFACTS, FULL, H2_ARMS,
    ORDER_DEPENDENT, ORDER_FREE, QUICK, SETTLE, WIDTHS, Budget)
from dataset import DIRECTED, POS, kind, probes
from discopy.neural import check_equivariant
from discopy.neural.cells import POOL
from model import POINTERS, SOLVERS


# --- the test-time depth sweep ---------------------------------------------

def sweep(model, batches, budget: Budget) -> dict:
    """
    The score at every point of the test-time depth sweep, keyed by the
    multiple of the trained depth.

    A multiple rather than a round count: under the trajectory rule the
    trained depth is the length of the sample's own execution, so "three
    times as deep" is the only form of the sweep that asks the same
    question at ``n = 16`` and at ``n = 64``.

    Parameters:
        model : The trained model.
        batches : The :class:`~model.Batches` to score.
        budget : The budget whose ``sweep`` says which multiples to run.
    """
    return {f"x{factor:g}": zoo.evaluate_split(model, batches, factor=factor)
            for factor in budget.sweep}


def ladder(model, batches, factors) -> dict:
    """
    The out-of-distribution score as a function of **how many rounds the
    model is asked to run**, from the depth it trained at up to the depth
    the sample's own trajectory asks for.

    :func:`sweep` asks what running *deeper* than the trajectory buys; this
    asks the question underneath it, which only exists out of
    distribution: a task whose trajectory grows with ``n`` -- ``dijkstra``
    is 17 steps at ``n = 16`` and 65 at ``n = 64`` -- makes a model iterate
    four times further than it ever did in training, and a task whose
    trajectory does not is spared that. Scoring the same model on the same
    split at several depths separates the two: if the score falls
    monotonically in the depth, the lesion is the iteration and not the
    size, and no amount of reading the edge decoders will find it.

    The shallow end is deliberately *wrong* as an imitation -- stopping
    ``dijkstra`` after 17 of its 65 extractions cannot produce the right
    answer -- which is what makes the comparison informative: a model that
    scores better having run a quarter of the algorithm is a model whose
    state degrades with every round.

    Parameters:
        model : The trained model.
        batches : The :class:`~model.Batches` to score.
        factors : The multiples of the sample's own depth to run at.

    Returns:
        One score per factor, keyed by the factor and by the algorithm
        steps it corresponds to.
    """
    steps = max(int(batch.steps) for batch in batches)
    return {f"x{factor:.2f}": {"steps": round(factor * steps),
                               **zoo.evaluate_split(model, batches,
                                                    factor=factor)}
            for factor in factors}


# --- where in the trajectory the model diverges ----------------------------

@torch.no_grad()
def hint_curve(model, batches, factor: float = 1.0) -> dict:
    """
    Per hint probe, its CLRS score at every step of the trajectory: the
    diagnostic Part 2 asks for, i.e. *where* the imitation comes apart.

    ``clrs._src.evaluation.evaluate_hints``' own shape -- the ``k``-th
    checkpoint scored against ``hints[k + 1]`` over the trajectories still
    running -- so a curve that is high at the start and falls is a model
    that tracks the algorithm and then loses it, and one that is flat and
    mediocre everywhere is a model that learned the *average* hint.  A
    uniform mediocrity across every probe at once is what a misaligned
    checkpoint-to-step mapping would look like, which is why this is
    recorded per probe rather than pooled.

    Decoded one checkpoint at a time on purpose: an edge pointer of
    ``floyd_warshall`` is an ``n x n x n`` tensor per step, and a whole
    trajectory of them at ``n = 64`` is not something to hold.

    Parameters:
        model : The trained model.
        batches : The :class:`~model.Batches` to score.
        factor : The multiple of the trained depth to run at.
    """
    model.eval()
    names = probes(model.algorithm, "hint")
    found: dict = {name: {} for name in names}
    for batch in batches:
        for step, state in enumerate(
                model.run(batch, deep=True, factor=factor)):
            targets = model.hint_targets(batch, step, settle=False)
            if not targets:
                break
            prediction = model.decode(batch, state, names=list(targets))
            for name, (truth, alive) in targets.items():
                found[name].setdefault(step, []).append((
                    prediction[name][alive].cpu().numpy(),
                    truth.cpu().numpy()))
    return {name: [
        zoo.DECODERS[kind(model.algorithm, name)].score(
            np.concatenate([one for one, _ in pairs]),
            np.concatenate([other for _, other in pairs]))
        for step, pairs in sorted(steps.items())]
        for name, steps in found.items()}


# --- what the diagram promises ---------------------------------------------

def equivariance(model, widths=None, atol: float = 1e-3) -> dict:
    """
    The :func:`~discopy.neural.check_equivariant` residual of every cell of
    a trained model, in float64.

    A :class:`~discopy.neural.Site` pools its message orbit symmetrically,
    so the equation holds up to the reordering of a floating-point
    reduction and the residual reported is that rounding error.  It is
    measured rather than asserted.

    It is **not** the quantity H4 correlates against the generalization
    drop, and this docstring said it was.  Measured on every trained
    model of Part 2: under the primary ``max`` campaign the residual is
    ``0.0`` exactly, on every cell of every task and seed, because a max
    is order-invariant in floating point -- the law is *strict* here, not
    lax, and a strict law has no residual to vary.  Under ``mean`` it is
    ``8.9e-16`` on the node cell and ``1.8e-15`` on the readout, which is
    machine epsilon over the width of a reduction: a fact about orbit
    sizes, not about learned weights.  H4 asked whether measured symmetry
    covaries with generalization and the answer is that in this formalism
    the symmetry is exact by construction, so it covaries with nothing.
    That is a negative verdict under a controlled protocol, which is a
    result; see ``PART3.md`` and :func:`h4_table`.

    The one cell that owes nothing is the directed edge of
    ``dag_shortest_paths``: its signature declares ``Sym.NONE``, so its
    group has no generators and the residual is the empty dictionary
    rather than a zero.  That is the honest reading -- a model that
    answers its source and its target differently is not equivariant and
    does not claim to be -- and H4 will have to correlate seven cells
    against eight tasks and say so.

    Parameters:
        model : The trained model.
        widths : The widths the cells were built at.
        atol : The residual above which a cell is rejected.
    """
    ob = zoo.graph_ob(widths or WIDTHS["mpnn"],
                      None if sum(model.map.ob[zoo.ESTATE].inside) else 0)
    widths = zoo.widths_of(ob)
    shapes = {"node": zoo.node(4),
              "edge": zoo.edge(model.algorithm in DIRECTED),
              "readout": zoo.readout(6)}
    found: dict = {}
    for name, cell in model.map.ar.items():
        # a copy, on the cpu and in float64: the check owns its input, the
        # model keeps its device and its precision.
        residual = check_equivariant(
            deepcopy(cell).cpu().double(), shapes[name], widths, atol=atol)
        found[name] = {str(role): float(value)
                       for role, value in residual.items()}
    return found


@torch.no_grad()
def residuals(model, batches, factor: float = 1.0) -> dict:
    """
    :math:`\\|T(s) - s\\|_\\infty` at the state a run ends on, over a
    split.

    Nothing makes this go to zero: contractivity is an analytic property of
    the learned weights.  Reporting it in Part 1 is what makes H2 a
    measurement in Part 3 rather than a hope.

    Parameters:
        model : The trained model.
        batches : The :class:`~model.Batches` to run.
        factor : The multiple of the trained depth to run at.
    """
    model.eval()
    found = []
    for batch in batches:
        interaction = model.map.compile(batch.diagram)
        state = model.run(batch, factor=factor)[-1]
        found.append(float(interaction.residual(state).max()))
    return {"max": max(found), "mean": float(np.mean(found))}


@torch.no_grad()
def residual_curve(model, batches, factor: float = 1.0) -> list:
    """
    :math:`\\|T(s_r) - s_r\\|_\\infty` after *every* round, averaged over
    the batches of a split.

    The scalar :func:`residuals` reports says whether a run stopped
    somewhere flat; the curve says what it was doing on the way, and the
    two algorithms answer differently.  ``bellman_ford`` *is* a fixed-point
    iteration -- its own relaxation converges and stays -- so a model that
    aligns with it should show a falling curve; ``minimum`` is a sequential
    scan with nothing to settle to, and a model of it need not.  Recorded
    now, in Part 1, because it is the instrument H2 will read in Part 3 and
    an instrument is worth more with a baseline behind it.

    The curve is run past the trained depth on purpose: whether a state
    stays put after the rounds it was supervised at is exactly the question
    that a residual measured *at* that depth cannot answer.

    Parameters:
        model : The trained model.
        batches : The :class:`~model.Batches` to run.
        factor : The multiple of the trained depth to run at.

    Returns:
        One residual per *round*, so its length is ``HOPS`` times the
        checkpoints; the period-two oscillation Part 1 recorded is a node
        round and a readout round being different distances from a fixed
        point, and it is only readable at this resolution.
    """
    model.eval()
    curves = []
    for batch in batches:
        interaction = model.map.compile(batch.diagram)
        every = model.map(batch.diagram, model.initial(batch), deep=True,
                          rounds=model.rounds_for(batch, factor))
        curves.append([float(interaction.residual(state).max())
                       for state in every])
    return [float(np.mean(round_)) for round_ in zip(*curves)]


@torch.no_grad()
def settling(batches) -> dict:
    """
    The round at which **the algorithm** stops moving, read off the hints
    of a split and not off any model.

    A residual curve says where the learned map settles.  On its own that
    is a fact about a dynamical system; it becomes a fact about *algorithmic
    alignment* only beside where the thing being imitated settles, which
    the benchmark states outright: a trajectory's hints are the algorithm's
    own state at every step, so the last index at which any of them changes
    is the step after which the algorithm is doing nothing.  H2 is the
    sentence "the learned map settles where the algorithm does", and this
    is the second half of it.

    A hint index ``j`` is read at round ``HOPS * j`` -- the mapping of
    :func:`~model.alignment`, since checkpoint ``step`` is round
    ``HOPS * (step + 1)`` and carries hint ``step + 1`` -- so the rounds
    reported here are on the same axis as :func:`residual_curve`.

    Padding is excluded: a trajectory of ``length`` steps defines
    ``hints[0] ... hints[length - 1]`` and the benchmark repeats the last
    state beyond that, which would otherwise read as convergence at the
    padding boundary for every sample at once.

    Parameters:
        batches : The :class:`~model.Batches` to read.

    Returns:
        The per-trajectory settling step and round, their quantiles, and
        the same per hint probe: a trajectory settles when its *last*
        probe does, so the overall number is a maximum, and which probe
        attains it is the difference between "the algorithm has finished"
        and "one counter is still counting".
    """
    steps: dict = {}
    for batch in batches:
        lengths = batch.lengths.cpu().numpy().astype(int)
        for name in probes(batch.algorithm, "hint"):
            values = batch.hints[name].detach().cpu().numpy()
            moved = np.zeros((values.shape[0], len(batch)), bool)
            moved[1:] = (values[1:] != values[:-1]).reshape(
                values.shape[0] - 1, len(batch), -1).any(axis=-1)
            steps.setdefault(name, []).extend(
                int(np.max(np.flatnonzero(alive))) if alive.any() else 0
                for alive in (moved[:length, index]
                              for index, length in enumerate(lengths)))
    overall = [max(one) for one in zip(*steps.values())]
    found = quantiles(overall)
    found["steps"] = overall
    found["per_probe"] = {name: quantiles(one) for name, one in steps.items()}
    return found


def quantiles(steps: list) -> dict:
    """
    An algorithm step per trajectory, summarised on the *round* axis of
    :func:`residual_curve` -- a hint index ``j`` is read at round
    ``HOPS * j``, so this is where a curve and a trajectory can be
    compared.

    Example
    -------
    >>> found = quantiles([1, 2, 3])
    >>> found["median"], found["low"], found["high"]
    (4.0, 2.4, 5.6)

    Parameters:
        steps : One settling step per trajectory.
    """
    rounds = [zoo.HOPS * step for step in steps]
    return {"rounds": rounds, "mean": float(np.mean(rounds)),
            "median": float(np.median(rounds)),
            "low": float(np.quantile(rounds, 0.1)),
            "high": float(np.quantile(rounds, 0.9))}


# --- the report ------------------------------------------------------------

def report(algorithm: str, budget: Budget = FULL, seeds=None, device=None,
           log=print) -> dict:
    """
    Score every trained seed of one algorithm and write the artefact.

    Parameters:
        algorithm : The algorithm.
        budget : The budget the models were trained under.
        seeds : The seeds to score, the budget's by default.
        device : Where to run, the GPU by default.
        log : Where to print progress.
    """
    device = training.default_device() if device is None else device
    splits = dataset.load_all(algorithm)
    splits["wide"] = splits["wide"].subsample(budget.n_wide)
    batches = {name: zoo.Batches(splits[name], budget.eval_batch_size, device)
               for name in ("val", "test", "wide")}
    deepest = max(budget.sweep)
    rows: list = []
    for seed in (seeds or budget.seeds):
        model, record = training.train_model(
            algorithm, budget, seed, device=device, splits=splits)
        # scoring draws three more splits than training did, so the cache
        # is resized again here or the sweep recompiles what it just ran.
        zoo.fit_cache(model, *batches.values())
        model.map.cache_stats(reset=True)
        row = {
            "seed": seed,
            "parameters": record.get("parameters"),
            "calls_per_round": record.get("calls_per_round"),
            "seconds_per_epoch": record.get("seconds_per_epoch"),
            "rounds": record.get("rounds"),
            "in_distribution": zoo.evaluate_split(model, batches["val"]),
            "ood": zoo.evaluate_split(model, batches["test"]),
            "ood_wide": zoo.evaluate_split(model, batches["wide"]),
            "ood_wide_interval": interval(model, batches["wide"]),
            "sweep_ood": sweep(model, batches["test"], budget),
            "equivariance": equivariance(model, WIDTHS[budget.widths]),
            "residual": residuals(model, batches["test"]),
            "residual_curve": {
                "factor": deepest,
                "trained_at": record.get("depth"),
                "val": residual_curve(model, batches["val"], deepest),
                "ood": residual_curve(model, batches["test"], deepest),
            },
            "hints": {"val": hint_curve(model, batches["val"]),
                      "ood": hint_curve(model, batches["test"])},
            "compile_cache": {"training": record.get("compile_cache"),
                              "scoring": model.map.cache_stats()},
        }
        rows.append(row)
        log(f"  {algorithm}/seed{seed}:"
            f"  id {row['in_distribution']['score']:.4f}"
            f"  ood {row['ood']['score']:.4f}"
            f"  ood({len(splits['wide'])}) {row['ood_wide']['score']:.4f}"
            f" +- {row['ood_wide_interval']['half_width']:.4f}")
    found = {
        "algorithm": algorithm,
        # both stages: a hint curve is read against the probe's *type*,
        # since a scalar probe is scored by CLRS with a mean squared error
        # and every other type with an F1 or an accuracy -- one of the
        # four curves in a panel is the only one where lower is better.
        "types": {name: "/".join(kind(algorithm, name))
                  for stage in ("output", "hint")
                  for name in probes(algorithm, stage)},
        "protocol": {"train": "n <= 16, CLRS-30 seeds", "val": "n = 16",
                     "ood": "n = 64, 32 trajectories (canonical)",
                     "ood_wide": f"n = 64, {len(splits['wide'])} trajectories",
                     "depth": training.depth_policy(budget),
                     "hops_per_step": zoo.HOPS},
        "anchors": ANCHORS[algorithm],
        "anchor_source": ANCHOR_SOURCE,
        "settles": {"val": settling(batches["val"]),
                    "ood": settling(batches["test"])},
        "budget": {"name": budget.name, "epochs": budget.epochs,
                   "batch_size": budget.batch_size, "lr": budget.lr,
                   "widths": budget.widths, "pool": budget.pool,
                   "edge_state": budget.edge_state,
                   "hint_weight": budget.hint_weight},
        "provenance": {"device": str(device), "torch": torch.__version__,
                       "python": platform.python_version()},
        "seeds": rows,
        "summary": summarise(rows),
    }
    path = ARTIFACTS / f"{budget.tag}-{algorithm}-report.json"
    path.write_text(json.dumps(found, indent=2))
    log(f"  {algorithm}: -> {path.name}")
    return found


@torch.no_grad()
def interval(model, batches, confidence: float = 1.96) -> dict:
    """
    The score of the larger out-of-distribution split with a confidence
    interval **over trajectories**.

    The primary number of every table, and the reason the split exists.
    A single-seed score on 32 trajectories can only take so many values --
    for a ``mask_one`` output it is one decision per trajectory, so 33 of
    them -- and two rows that differ by a tenth there differ by nothing.
    The interval is the normal approximation to the mean of the per
    -trajectory scores, which is what a per-sample score allows and what
    a pooled F1 does not: a ``mask`` probe is therefore scored per
    trajectory here and pooled in :func:`~model.evaluate_split`, and the
    two agree only up to the non-linearity of an F1, which is why both are
    reported.

    Parameters:
        model : The trained model.
        batches : The :class:`~model.Batches` to score.
        confidence : The number of standard errors, 1.96 for 95%.
    """
    model.eval()
    scores: list = []
    for batch in batches:
        found = model.predict(batch)
        for index in range(len(batch)):
            scores.append(zoo.score(model.algorithm, {
                name: (prediction[index:index + 1], truth[index:index + 1])
                for name, (prediction, truth) in found.items()})["score"])
    error = float(np.std(scores, ddof=1) / np.sqrt(len(scores))) \
        if len(scores) > 1 else 0.0
    return {"mean": float(np.mean(scores)), "std_error": error,
            "half_width": confidence * error, "trajectories": len(scores)}


def ladder_report(algorithm: str, budget: Budget = FULL, seeds=None,
                  device=None, log=print) -> dict:
    """
    Write the depth ladder of one algorithm: the out-of-distribution score
    at the depth the model trained at, at half the sample's own depth, and
    at the sample's own depth.

    Its own artefact rather than a column of :func:`report`, because it is
    a diagnostic run after a table has raised a question and because
    re-scoring eight tasks to add a column is hours the answer does not
    need.

    Parameters:
        algorithm : The algorithm.
        budget : The budget the models were trained under.
        seeds : The seeds to score, the budget's by default.
        device : Where to run, the GPU by default.
        log : Where to print progress.
    """
    device = training.default_device() if device is None else device
    splits = dataset.load_all(algorithm)
    trained = int(splits["val"].lengths.max())
    asked = int(splits["test"].lengths.max())
    factors = sorted({round(trained / asked, 2), 0.5, 1.0})
    rows: list = []
    for seed in (seeds or budget.seeds):
        model, _ = training.train_model(
            algorithm, budget, seed, device=device, splits=splits)
        batches = zoo.Batches(splits["test"], budget.eval_batch_size, device)
        zoo.fit_cache(model, batches)
        rows.append({"seed": seed, "ladder": ladder(model, batches, factors)})
        log(f"  {algorithm}/seed{seed}: " + "  ".join(
            f"x{key[1:]} ({one['steps']} steps) {one['score']:.4f}"
            for key, one in rows[-1]["ladder"].items()))
    found = {
        "algorithm": algorithm, "trained_steps": trained,
        "ood_steps": asked, "factors": factors,
        "summary": {key: {"mean": float(np.mean(
            [row["ladder"][key]["score"] for row in rows]))}
            for key in rows[0]["ladder"]},
        "seeds": rows,
    }
    path = ARTIFACTS / f"{budget.tag}-{algorithm}-ladder.json"
    path.write_text(json.dumps(found, indent=2))
    log(f"  {algorithm}: -> {path.name}")
    return found


def summarise(rows: list) -> dict:
    """
    Each score over the seeds, with **both** of the two spreads a row of
    this study has and a name for each.

    ``std`` is the sample standard deviation over the seeds and ``sem`` is
    its standard error, ``std / sqrt(seeds)``: the second is what
    :data:`config.ANCHORS` prints, on the paper's own statement that its
    error bars are "standard error of the mean across seeds", so it is
    what a column beside them has to print.  ``half_width`` is the other
    question entirely -- 1.96 standard errors of the mean *over the 128
    trajectories of one run*, averaged over the runs -- and it answers how
    finely this split can resolve a difference at all, which no number of
    seeds improves.

    Example
    -------
    >>> row = lambda score: {
    ...     "in_distribution": {"score": score}, "ood": {"score": score},
    ...     "ood_wide": {"score": score}, "sweep_ood": {},
    ...     "ood_wide_interval": {"mean": score, "half_width": 0.02,
    ...                           "trajectories": 128}}
    >>> found = summarise([row(0.5), row(0.6), row(0.7)])["ood_wide"]
    >>> f"{found['std']:.4f} {found['sem']:.4f} {found['seeds']}"
    '0.1000 0.0577 3'

    Parameters:
        rows : One record per seed, as :func:`report` builds them.
    """
    found: dict = {}
    for key in ("in_distribution", "ood", "ood_wide"):
        values = [row[key]["score"] for row in rows]
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        found[key] = {"mean": float(np.mean(values)), "std": std,
                      "sem": std / np.sqrt(len(values)), "seeds": len(values)}
    found["ood_wide_interval"] = {
        "mean": float(np.mean(
            [row["ood_wide_interval"]["mean"] for row in rows])),
        "half_width": float(np.mean(
            [row["ood_wide_interval"]["half_width"] for row in rows])),
        "trajectories": rows[0]["ood_wide_interval"]["trajectories"],
    }
    depths = {depth for row in rows for depth in row["sweep_ood"]}
    found["sweep_ood"] = {
        depth: {"mean": float(np.mean(
            [row["sweep_ood"][depth]["score"] for row in rows]))}
        for depth in sorted(depths, key=lambda one: float(one[1:]))}
    return found


def tabulate(reports: dict, ladders: dict = None, log=print) -> None:
    """
    The eight-task table, as markdown: ours against the two anchors of
    ``project.md``.

    The anchor columns print a number only where one has been transcribed
    from Ibarz et al. (2022); :data:`config.ANCHORS` holds ``None`` until
    then, so a remembered figure can never be mistaken for a published
    one.

    The two spreads are separate columns and neither is called ``±``
    without saying over what.  ``± s.e.m.`` is over the seeds, which is
    the anchors' own convention and therefore the one comparable across
    the row; ``± CI`` is the 95% interval over the 128 trajectories of a
    single run, which says how finely the *split* resolves a difference
    and does not shrink when a seed is added.

    Every column of ours is at ``n = 64``, which is the size the anchors
    are published at; the parenthesis in a header is a **count of
    trajectories** and never a size.  It is spelled out in all three
    because ``OOD (128)`` on its own reads as a size to anyone who has
    not opened :data:`config.WIDE`, and an anchor comparison that looks
    like it crosses two sizes is worth no more than one that does.

    Parameters:
        reports : Per algorithm, what :func:`report` returned.
        ladders : Per algorithm, what :func:`ladder_report` wrote, for
                  the trained-depth column.  Out of distribution a model
                  is asked for a depth it never trained at *and* a size
                  it never trained at, and this is the column that holds
                  the second fixed while the primary one varies both.
        log : Where to print.
    """
    ladders = ladders or {}
    log("| algorithm | seeds | ID `n = 16` | OOD `n = 64` (32 traj.) "
        "| OOD `n = 64` (128 traj.) ± s.e.m. | ± 95% CI (traj.) "
        "| at trained depth | floor (MPNN) | ceiling (Triplet-GMPNN) |")
    log("|---|---|---|---|---|---|---|---|---|")
    for algorithm, found in reports.items():
        summary, anchors = found["summary"], found["anchors"]
        wide, over = summary["ood_wide_interval"], summary["ood_wide"]
        against = [anchor_of(anchors[key]) for key in ("floor", "ceiling")]
        log(f"| `{algorithm}` | {over['seeds']} "
            f"| {summary['in_distribution']['mean']:.4f} "
            f"| {summary['ood']['mean']:.4f} "
            f"| {wide['mean']:.4f} ± {over['sem']:.4f} "
            f"| ± {wide['half_width']:.4f} "
            f"| {at_trained_depth(ladders.get(algorithm))} "
            f"| {against[0]} | {against[1]} |")
    log("")
    log("Every column is at `n = 64` out of distribution; a parenthesis "
        "in a header counts **trajectories**, not nodes. "
        "`± s.e.m.` is the standard error over seeds, the anchors' own "
        "convention (theirs: 3 seeds for the floor, 10 for the ceiling). "
        "`± 95% CI` is 1.96 standard errors over the 128 trajectories "
        "within a run, averaged over the seeds. `at trained depth` is "
        "the same models on the same split run for the number of rounds "
        "they trained at rather than the number the sample's trajectory "
        "asks for.")


def at_trained_depth(ladder) -> str:
    """
    The rung of a depth ladder that means "the depth this model trained
    at", looked up by what it *is* and not by where it sits.

    The factor that names it is a different number for every task and on
    ``bfs`` it is the deepest rung rather than the shallowest, since a
    bigger `bfs` graph has a *shorter* trajectory.

    Example
    -------
    >>> at_trained_depth({"trained_steps": 7, "ood_steps": 4,
    ...                   "summary": {"x1.75": {"mean": 0.8501}}})
    '0.8501'
    >>> at_trained_depth(None)
    '—'
    """
    if ladder is None:
        return "—"
    rung = ladder["summary"].get(
        f"x{ladder['trained_steps'] / ladder['ood_steps']:.2f}")
    return "—" if rung is None else f"{rung['mean']:.4f}"


# --- the heads, split ------------------------------------------------------

def heads(algorithm: str, stages=("hint", "output")) -> dict:
    """
    The scored probes of an algorithm, grouped by what their decoder
    selects over.  Part 3's second rule: no table pools these.

    * ``free`` -- a ``mask`` or a ``categorical``, i.e. one answer per
      element out of a candidate set that does not grow with the graph.
    * ``dependent`` -- a ``pointer`` or a ``mask_one``, an ``argmax``
      whose candidate set *is* the node set, 16 in training and 64 out of
      it; and the reference algorithms iterate in index order, so their
      targets are tie-broken by an order the processor can only see
      through one size-dependent scalar.  M2 is this class failing while
      the other does not.
    * ``unpooled`` -- a ``scalar``, scored by a mean squared error.  It
      is unbounded and lower is better, so it is averaged with nothing:
      pooling it with an F1 is what made ``floyd_warshall``'s order-free
      drop come out at **-0.046**, which is not an improvement, it is two
      scales in one mean.

    Parameters:
        algorithm : The algorithm.
        stages : The stages to include, hints and outputs by default.

    Example
    -------
    >>> heads("bfs")
    {'free': ('reach_h',), 'dependent': ('pi_h', 'pi'), 'unpooled': ()}
    >>> heads("bfs", ("output", ))
    {'free': (), 'dependent': ('pi',), 'unpooled': ()}
    >>> heads("minimum")["free"]
    ()
    """
    found = {"free": [], "dependent": [], "unpooled": []}
    for stage in stages:
        for name in probes(algorithm, stage):
            type_ = kind(algorithm, name)[1]
            found["free" if type_ in ORDER_FREE else "dependent"
                  if type_ in ORDER_DEPENDENT else "unpooled"].append(name)
    return {key: tuple(value) for key, value in found.items()}


def head_mass(algorithm: str) -> dict:
    """
    The share of an algorithm's scored probes in each class, over the
    **output** alone and over the hints and the output together.

    The output column is the finding that forces H4's amendment and it is
    the same on all eight tasks: every algorithm of this study has
    exactly one output probe and it is a ``pointer`` or a ``mask_one``,
    so the benchmark's own micro-F1 is 100 % order-dependent mass
    everywhere.  An amendment that asks for the *order-free output* drop
    therefore asks for an empty column, and one that asks to partial out
    the order-dependent mass asks to partial out a constant.  Both are
    repaired the same way, by scoring over the hints as well; see
    ``PART3.md``.

    Parameters:
        algorithm : The algorithm.

    Example
    -------
    >>> head_mass("bfs")["output"]["dependent"]
    1.0
    >>> sorted({head_mass(one)["output"]["dependent"]
    ...         for one in ALGORITHMS})
    [1.0]
    >>> round(head_mass("bfs")["scored"]["dependent"], 4)
    0.6667
    """
    def share(stages=("hint", "output")):
        found = heads(algorithm, stages)
        total = sum(map(len, found.values()))
        return {key: len(value) / total for key, value in found.items()}
    return {"output": share(("output", )), "scored": share()}


def head_split(report: dict, wide: bool = True) -> dict:
    """
    Per head class, the in-distribution score, the out-of-distribution
    score and the drop between them, over the seeds of a report.

    A hint probe contributes the mean of its per-step curve and an output
    probe contributes its recorded score, and the two are averaged inside
    a class and never across one.  The caveat :func:`hint_curve` carries
    applies to the hint half and is not hidden: a curve's last point is
    computed on the deepest trajectories alone, so a mean over steps
    weights the tail by few samples.  It is the same convention on both
    splits, which is what a *drop* needs.

    Parameters:
        report : One algorithm's report, as :func:`report` returned it.
        wide : Whether the out-of-distribution column is the
               128-trajectory split rather than the canonical 32.

    Example
    -------
    >>> found = head_split({"algorithm": "bfs", "seeds": [{
    ...     "hints": {"val": {"reach_h": [0.8, 1.0], "pi_h": [0.9, 0.9]},
    ...               "ood": {"reach_h": [0.6, 0.8], "pi_h": [0.5, 0.5]}},
    ...     "in_distribution": {"pi": 0.99}, "ood_wide": {"pi": 0.85}}]})
    >>> found["free"]["id"], found["free"]["ood"]
    (0.9, 0.7)
    >>> round(found["free"]["drop"], 4), round(found["dependent"]["drop"], 4)
    (0.2, 0.27)
    """
    algorithm = report["algorithm"]
    group = {name: key for key, names in heads(algorithm).items()
             for name in names}
    rows: dict = {key: [] for key in ("free", "dependent", "unpooled")}
    for seed in report["seeds"]:
        found = {key: ([], []) for key in rows}
        hints = seed.get("hints", {})
        for column, split in enumerate(("val", "ood")):
            for name, curve in hints.get(split, {}).items():
                found[group[name]][column].append(float(np.mean(curve)))
        out = seed.get("ood_wide" if wide else "ood", {})
        for name in probes(algorithm, "output"):
            if name in seed.get("in_distribution", {}) and name in out:
                found[group[name]][0].append(seed["in_distribution"][name])
                found[group[name]][1].append(out[name])
        for key, (here, there) in found.items():
            if here and there:
                rows[key].append((float(np.mean(here)), float(np.mean(there))))

    def summary(pairs):
        if not pairs:
            return None
        here = [one for one, _ in pairs]
        there = [one for _, one in pairs]
        drops = [one - other for one, other in pairs]
        return {"id": float(np.mean(here)), "ood": float(np.mean(there)),
                "drop": float(np.mean(drops)), "seeds": len(pairs),
                "sem": float(np.std(drops, ddof=1) / np.sqrt(len(drops)))
                if len(drops) > 1 else None}
    return {**{key: summary(value) for key, value in rows.items()},
            "mass": head_mass(algorithm)}


def head_table(algorithms=ALGORITHMS, budget: Budget = FULL,
               log=print) -> dict:
    """
    :func:`head_split` for several algorithms as one markdown table:
    Part 3's second rule, in the form every one of its tables owes.

    A row's headline score is the mean over its **output** probes, and
    every algorithm here has exactly one output probe which is an
    ``argmax`` over the node set.  So the headline column and the
    order-dependent column are answering nearly the same question, and
    the order-free column is answering the other one -- whether the
    processor computed the order-free part of the algorithm at all.  A
    solver that improves the second and not the first has done something
    real that the benchmark's own metric cannot see; a solver credited
    with the first alone has been credited with pointer points.

    Parameters:
        algorithms : The algorithms to read.
        budget : The budget they were scored under.
        log : Where to print the markdown.
    """
    rows = {}
    for algorithm in algorithms:
        path = ARTIFACTS / f"{budget.tag}-{algorithm}-report.json"
        if path.exists():
            rows[algorithm] = head_split(json.loads(path.read_text()))
    log("| algorithm | order-free ID | order-free OOD | order-free drop "
        "| order-dep. ID | order-dep. OOD | order-dep. drop | scalar "
        "(MSE) ID → OOD |")
    log("|---|---|---|---|---|---|---|---|")
    for algorithm, row in rows.items():
        cells = []
        for key in ("free", "dependent"):
            one = row[key]
            cells += ["—"] * 3 if one is None else [
                f"{one['id']:.3f}", f"{one['ood']:.3f}", f"{one['drop']:.3f}"]
        scalar = row["unpooled"]
        cells.append("—" if scalar is None
                     else f"{scalar['id']:.3f} → {scalar['ood']:.3f}")
        log(f"| `{algorithm}` | " + " | ".join(cells) + " |")
    log("")
    log("Order-free is a `mask` or a `categorical`, order-dependent is a "
        "`pointer` or a `mask_one` — an `argmax` whose candidate set is "
        "the node set. Both are scored over the hints and the output "
        "together. The `scalar` column is a mean squared error, lower is "
        "better, and it is pooled with neither: it is printed as two "
        "numbers rather than a drop for that reason.")
    return rows


def correlate(x: list, y: list) -> dict:
    """
    Pearson's ``r`` and its **exact** two-sided permutation ``p``, over
    every relabelling of the pairs.

    An exact test rather than a table lookup, for the reason H1's
    significance section gives: a design has a floor and printing it is
    what stops a reader importing a threshold that does not apply.  Here
    the floor is ``2 / n!`` -- with seven tasks that is 0.0004, so unlike
    the three-versus-three seed comparison this test *can* resolve
    something, and the small-sample caveat is about the seven points
    being tasks rather than about the arithmetic.

    Parameters:
        x : The independent variable, one number per task.
        y : The dependent variable, one number per task.

    Example
    -------
    >>> found = correlate([1, 2, 3, 4], [1, 2, 3, 4])
    >>> found["r"], found["p"], found["floor"]
    (1.0, 0.0833, 0.083)
    >>> correlate([1, 1, 1], [1, 2, 3])["why"]
    'a variable with no variance has no correlation'
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or not x.std() or not y.std():
        return {"r": None, "p": None, "n": len(x), "floor": None,
                "why": "a variable with no variance has no correlation"}
    found = float(np.corrcoef(x, y)[0, 1])
    every = [abs(float(np.corrcoef(x, y[list(order)])[0, 1]))
             for order in permutations(range(len(y)))]
    return {"r": round(found, 4), "n": len(x),
            "p": round(sum(one >= abs(found) - 1e-12 for one in every)
                       / len(every), 4),
            "floor": float(f"{2 / len(every):.2g}")}


def tracking(report: dict) -> dict:
    """
    Executor or shortcut: how closely a model follows the hints it was
    supervised on, out of distribution, at its **best** step rather than
    at its last one.

    The distinction the number exists for is the one Part 3 rests on.  A
    model that has learned to execute tracks the trajectory while it is
    short enough and comes apart later, so its out-of-distribution curve
    starts near its in-distribution one and falls: the failure is in the
    iteration, and stabilizing the iteration is a thing one can do.  A
    model that has learned a shortcut calibrated to ``n = 16`` never
    tracks the trajectory at any step of a bigger one, however well it
    scores on the output -- and then there is no fixed point to find,
    because the rounds were never approximating the steps.  Reading the
    *last* step cannot tell those apart, since both end low; reading the
    best step over the whole trajectory can.

    ``reached`` is the ratio of the two, and it is deliberately a ratio
    and not a verdict: where a probe is scored by an error rather than by
    an accuracy -- CLRS scores a ``scalar`` with a mean squared error --
    there is no ratio to take and the field is ``None``, because a
    threshold that silently flips direction on one probe type is how a
    classification becomes an artefact of its coding.

    Parameters:
        report : One algorithm's report, as :func:`report` returned it.

    Example
    -------
    >>> found = tracking({"algorithm": "bfs", "seeds": [{"hints": {
    ...     "val": {"reach_h": [0.8, 1.0]}, "ood": {"reach_h": [0.5, 0.6]}}}]})
    >>> found["reach_h"]["ood_best"], found["reach_h"]["reached"]
    (0.6, 0.6667)
    """
    found = {}
    for name, curve in report["seeds"][0]["hints"]["val"].items():
        out = report["seeds"][0]["hints"]["ood"][name]
        # the probe's type comes off the benchmark's own specification and
        # not off the report, whose `types` field five of the eight
        # campaigns were written before it covered the hint stage.
        location, type_ = kind(report["algorithm"], name)
        best = min(out) if type_ == "scalar" else max(out)
        found[name] = {
            "kind": f"{location}/{type_}",
            "id": float(np.mean(curve)), "ood_first": float(out[0]),
            "ood_best": float(best), "ood_last": float(out[-1]),
            "reached": None if type_ == "scalar"
            else round(float(best / np.mean(curve)), 4)}
    return found


def tracking_table(algorithms, budget: Budget = FULL, log=print) -> dict:
    """
    :func:`tracking` for several algorithms as one markdown table, over
    the probes whose decoder is an ``argmax`` over the nodes -- a
    ``pointer`` or a ``mask_one``.

    Those are the probes the restriction is about: every one of the eight
    tasks is scored on one, and they are the heads whose candidate set is
    the graph, so they are the heads a change of size reaches.  A ``mask``
    is a sigmoid per node and its number of candidates is one however big
    the graph is; the tables show it generalizing, which is the contrast
    that makes the restriction informative rather than convenient.

    Parameters:
        algorithms : The algorithms to read.
        budget : The budget they were scored under.
        log : Where to print the markdown.
    """
    found = {}
    for algorithm in algorithms:
        path = ARTIFACTS / f"{budget.tag}-{algorithm}-report.json"
        if path.exists():
            found[algorithm] = tracking(json.loads(path.read_text()))
    log("| algorithm | probe | kind | ID (mean over steps) "
        "| OOD first | OOD best | OOD last | reached |")
    log("|---|---|---|---|---|---|---|---|")
    for algorithm, one in found.items():
        for name, row in one.items():
            if not row["kind"].endswith(("pointer", "mask_one")):
                continue
            log(f"| `{algorithm}` | `{name}` | {row['kind']} "
                f"| {row['id']:.3f} | {row['ood_first']:.3f} "
                f"| {row['ood_best']:.3f} | {row['ood_last']:.3f} "
                f"| {row['reached']:.2f} |")
    log("")
    log("`reached` is the best out-of-distribution step over the mean "
        "in-distribution one: near 1 the model tracks the algorithm "
        "somewhere and the failure is in the iteration, near 0 it never "
        "tracks it at any step and the rounds are not approximating the "
        "steps at all.")
    return found


def h4_table(algorithms=ALGORITHMS, budget: Budget = FULL, log=print) -> dict:
    """
    H4's table and its correlations, in the amended form ``PART3.md``
    argues for.

    Three things are printed and they answer three different questions.

    **The equivariance column closes H4 as it was written.** The
    hypothesis was that the ``check_equivariant`` residual of the trained
    cells covaries with the generalization drop.  It is measured here and
    it is *identically zero* on the primary campaign -- a
    :data:`~discopy.neural.cells.POOL` of ``"max"`` is order-invariant in
    floating point, so the law is strict rather than lax and the residual
    is not small, it is absent.  Under ``"mean"`` it is 9e-16, i.e.
    machine epsilon over the width of a reduction, which is a fact about
    orbit sizes and not about learned weights.  A correlate with no
    variance is not a weak correlate; :func:`correlate` says so rather
    than returning a number.

    **The two drop columns are the amendment.** They are computed over
    the hints as well as the output, because :func:`head_mass` measures
    the output mass of all eight tasks to be 100 % order-dependent -- so
    an order-free *output* drop is an empty column on every row, and
    there is no order-dependent mass to partial out at the output level
    because it is a constant.

    **The mass correlation is printed as a rejected candidate**, not as a
    result.  It was proposed as H4's replacement -- order-dependent mass
    as an independent variable that, unlike the equivariance residual,
    varies -- and the measurement kills it: ``r = -0.65`` at
    ``p = 0.046``, which is *significant with the wrong sign*.  More
    order-dependent mass goes with a **smaller** drop, because this
    quantity is a ratio of probe *counts* and the tasks with few probes
    are the easy ones: ``minimum`` is 100 % order-dependent mass and
    drops 0.195, ``dijkstra`` is 50 % and drops 0.908.  It is an inverse
    proxy for how hard a task is, wearing a mechanism's name -- the same
    failure the amendment to the dependent variable was written to avoid,
    one level down.  It is kept in the output so that nobody derives it
    again and reports it; the independent variable H4 needs is the
    measured **tie rate**, which is a property of the labels rather than
    of how many probes a spec happens to declare, and ``PART3.md`` says
    what it costs to build.

    Parameters:
        algorithms : The algorithms to read.
        budget : The budget they were scored under.
        log : Where to print the markdown.
    """
    rows: dict = {}
    for algorithm in algorithms:
        path = ARTIFACTS / f"{budget.tag}-{algorithm}-report.json"
        if not path.exists():
            continue
        found = json.loads(path.read_text())
        split = head_split(found)
        residual = [value for seed in found["seeds"]
                    for cell in seed.get("equivariance", {}).values()
                    for value in cell.values()]
        drops = [seed["in_distribution"]["score"] - seed["ood_wide"]["score"]
                 for seed in found["seeds"] if "ood_wide" in seed]
        rows[algorithm] = {
            "equivariance": max(residual) if residual else None,
            "mass": split["mass"]["scored"]["dependent"],
            "output_mass": split["mass"]["output"]["dependent"],
            "free": split["free"], "dependent": split["dependent"],
            "drop": float(np.mean(drops)) if drops else None}
    log("| algorithm | equivariance residual | order-dependent mass "
        "(output) | order-dependent mass (scored) | order-free drop "
        "| order-dependent drop | headline drop |")
    log("|---|---|---|---|---|---|---|")
    for algorithm, row in rows.items():
        free = "—" if row["free"] is None else f"{row['free']['drop']:.3f}"
        log(f"| `{algorithm}` | {row['equivariance']:.1e} "
            f"| {row['output_mass']:.0%} | {row['mass']:.0%} | {free} "
            f"| {row['dependent']['drop']:.3f} | {row['drop']:.3f} |")
    log("")
    log("The equivariance column is the measurement that closes H4 as "
        "written: a `max` pooling is order-invariant in floating point, "
        "so the residual is absent rather than small and there is "
        "nothing for a drop to covary with. `—` in the order-free "
        "column is a task with no order-free probe of any kind. The mass "
        "columns count probes, and `mass_vs_drop` below is a **rejected "
        "candidate** kept on the record, not a finding: see the "
        "docstring, and `PART3.md` for the tie rate that replaces it.")
    keys = [one for one in rows if rows[one]["free"] is not None]
    found = {
        "rows": rows,
        "equivariance_vs_drop": correlate(
            [rows[one]["equivariance"] for one in rows],
            [rows[one]["drop"] for one in rows]),
        "mass_vs_drop": correlate([rows[one]["mass"] for one in rows],
                                  [rows[one]["drop"] for one in rows]),
        "mass_vs_free_drop": correlate(
            [rows[one]["mass"] for one in keys],
            [rows[one]["free"]["drop"] for one in keys])}
    for name, one in found.items():
        if name == "rows":
            continue
        log(f"* `{name}`: " + ("no correlation — " + one["why"]
                               if one["r"] is None else
                               f"r = {one['r']}, exact permutation "
                               f"p = {one['p']} over n = {one['n']} tasks "
                               f"(floor {one['floor']})"))
    return found


def anchor_of(anchor) -> str:
    """
    One published number, or a dash where none has been transcribed.

    Example
    -------
    >>> anchor_of({"mean": 0.4852, "sem": 0.0104}), anchor_of(None)
    ('0.4852 ± 0.0104', '—')
    """
    return "—" if anchor is None \
        else f"{anchor['mean']:.4f} ± {anchor['sem']:.4f}"


def ladder_table(algorithms, budget: Budget = FULL, log=print) -> dict:
    """
    The depth ladders of several algorithms as one markdown table, read
    off the artefacts :func:`ladder_report` wrote.

    One row per algorithm, one column per rung, and the rungs are named by
    what they *are* -- the depth the model trained at, half the sample's
    trajectory, all of it -- since the factor that means "the trained
    depth" is a different number for every task.

    Parameters:
        algorithms : The algorithms to read.
        budget : The budget they were scored under.
        log : Where to print the markdown.
    """
    found = {}
    for algorithm in algorithms:
        path = ARTIFACTS / f"{budget.tag}-{algorithm}-ladder.json"
        if path.exists():
            found[algorithm] = json.loads(path.read_text())
    log("| algorithm | steps: trained → out of distribution "
        "| at the trained depth | at half | at its own depth |")
    log("|---|---|---|---|---|")
    for algorithm, one in found.items():
        # by name and not by position: the rung that means "the depth this
        # model trained at" is a different factor for every task, and on
        # `bfs` it is the *deepest* one, since a bigger graph is shallower.
        summary = one["summary"]
        trained = f"x{one['trained_steps'] / one['ood_steps']:.2f}"
        rungs = [summary.get(key) for key in (trained, "x0.50", "x1.00")]
        log(f"| `{algorithm}` "
            f"| {one['trained_steps']} → {one['ood_steps']} "
            + "".join(f"| {rung['mean']:.4f} " if rung else "| — "
                      for rung in rungs) + "|")
    return found


def h1_table(algorithm: str, budget: Budget = FULL, log=print) -> dict:
    """
    H1's two columns, read off the two reports of one algorithm: the same
    diagram with an edge state and without one.

    The last row is the difference of the two, which is the number H1 is
    about, with the standard error the difference of two independent means
    has: ``sqrt(sem_edge**2 + sem_node**2)`` over the seeds.  A delta
    without one is a claim about an architecture that might be a claim
    about an initialization.

    Parameters:
        algorithm : The showcase algorithm.
        budget : The budget both arms were run under.
        log : Where to print the markdown.
    """
    arms = {"edge state": budget,
            "node only": replace(budget, edge_state=False, widths="paired")}
    found = {}
    for name, arm in arms.items():
        path = ARTIFACTS / f"{arm.tag}-{algorithm}-report.json"
        if path.exists():
            found[name] = json.loads(path.read_text())
    log(f"| `{algorithm}` | seeds | parameters | ID `n = 16` "
        f"| OOD `n = 64` (32 traj.) "
        f"| OOD `n = 64` (128 traj.) ± s.e.m. | ± 95% CI (traj.) |")
    log("|---|---|---|---|---|---|---|")
    for name, one in found.items():
        summary = one["summary"]
        wide, over = summary["ood_wide_interval"], summary["ood_wide"]
        log(f"| {name} | {over['seeds']} | {one['seeds'][0]['parameters']} "
            f"| {summary['in_distribution']['mean']:.4f} "
            f"| {summary['ood']['mean']:.4f} "
            f"| {wide['mean']:.4f} ± {over['sem']:.4f} "
            f"| ± {wide['half_width']:.4f} |")
    if len(found) == 2:
        arms = [[one["ood_wide"]["score"] for one in found[name]["seeds"]]
                for name in ("edge state", "node only")]
        test = significance(*arms)
        log(f"| **difference** | | | | | "
            f"**{test['difference']:+.4f} ± {test['error']:.4f}** | |")
        log("")
        log(f"Welch `t = {test['t']:.2f}` on `df = {test['df']:.1f}`, and "
            f"the exact two-sided permutation test over the "
            f"{test['partitions']} relabellings of the seeds gives "
            f"`p = {test['p']:.2f}`. With three seeds an arm the "
            f"permutation floor is `p = {2 / test['partitions']:.2f}` "
            f"however cleanly the arms separate, so a delta here is "
            f"suggestive and cannot be more than that: a standard-error "
            f"count is not a significance, and at these degrees of "
            f"freedom the two-sided 95% threshold is near `|t| = 3.2` "
            f"rather than `1.96`.")
    return found


def significance(edge, node) -> dict:
    """
    The difference of two arms with the two statements one can honestly
    make about it at three seeds each.

    ``t`` and ``df`` are Welch's, i.e. the difference in units of its own
    standard error and the degrees of freedom that error has -- which is
    near three here and not infinity, so the threshold that goes with it
    is near ``3.2`` and not ``1.96``.  ``p`` is exact and assumes
    nothing: it enumerates every way of dealing the seeds into two arms
    and counts the fraction whose gap is at least the observed one.  With
    three seeds an arm there are twenty such deals, so ``p`` cannot fall
    below ``0.1`` however far apart the arms are -- which is the honest
    ceiling on what this table can claim, and the reason it is printed
    beside the delta rather than left for a reader to work out.

    Example
    -------
    >>> found = significance([1.0, 1.1, 1.2], [2.0, 2.1, 2.2])
    >>> round(found["difference"], 4), found["p"], found["partitions"]
    (-1.0, 0.1, 20)
    """
    edge, node = np.asarray(edge, float), np.asarray(node, float)
    errors = [one.std(ddof=1) / np.sqrt(len(one)) for one in (edge, node)]
    error = float(np.sqrt(sum(one ** 2 for one in errors)))
    difference = float(edge.mean() - node.mean())
    pooled, deals = np.concatenate([edge, node]), []
    for pick in combinations(range(len(pooled)), len(edge)):
        rest = [one for one in range(len(pooled)) if one not in pick]
        deals.append(abs(pooled[list(pick)].mean() - pooled[rest].mean()))
    return {
        "difference": difference, "error": error,
        "t": difference / error,
        "df": float(sum(one ** 2 for one in errors) ** 2 / sum(
            one ** 4 / (len(arm) - 1)
            for one, arm in zip(errors, (edge, node)))),
        "p": float(np.mean(np.asarray(deals) >= abs(difference) - 1e-12)),
        "partitions": len(deals)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="*", default=ALGORITHMS)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--pool", choices=sorted(POOL), default=None)
    parser.add_argument("--widths", choices=sorted(WIDTHS), default=None)
    parser.add_argument("--hint-weight", dest="hint_weight", type=float,
                        default=None)
    parser.add_argument("--node-only", action="store_true")
    parser.add_argument("--h1", action="store_true",
                        help="the two arms of H1, from written reports")
    parser.add_argument("--table", action="store_true",
                        help="the eight-task table, from written reports")
    parser.add_argument("--ladder", action="store_true",
                        help="the out-of-distribution score against depth")
    parser.add_argument("--tracking", action="store_true",
                        help="executor or shortcut, from written reports")
    parser.add_argument("--heads", action="store_true",
                        help="the order-free / order-dependent split every "
                             "Part 3 table owes, from written reports")
    parser.add_argument("--h4", action="store_true",
                        help="H4 as amended, from written reports")
    parser.add_argument("--pointer", choices=sorted(POINTERS),
                        default=None,
                        help="which node-pointer head to build")
    parser.add_argument("--n-train", dest="n_train", type=int, default=None,
                        help="training trajectories, all of them by default")
    parser.add_argument("--pos", choices=POS, default=None,
                        help="what to do to the pos input")
    parser.add_argument("--mixed", action="store_true",
                        help="train on config.MIXED sizes")
    parser.add_argument("--settle", nargs="?", const="interior",
                        choices=[one for one in SETTLE if one], default=None,
                        help="hold a finished trajectory's last hint")
    parser.add_argument("--probe", action="store_true",
                        help="fit the hint heads on a detached state")
    parser.add_argument("--solver", choices=sorted(SOLVERS), default=None,
                        help="the execution policy")
    parser.add_argument("--backward", choices=("full", "last"), default=None,
                        help="a fixed point's differentiation policy")
    parser.add_argument("--arm", choices=sorted(H2_ARMS), default=None,
                        help="a Part 3 arm of config.H2_ARMS")
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args(argv)
    training.single_threaded()

    budget = H2_ARMS[arguments.arm] if arguments.arm \
        else QUICK if arguments.quick else FULL
    for key in ("epochs", "rounds", "pool", "widths",
                "hint_weight", "pointer", "pos", "settle",
                "solver", "backward", "n_train"):
        if getattr(arguments, key) is not None:
            budget = replace(budget, **{key: getattr(arguments, key)})
    for key in ("mixed", "probe"):
        if getattr(arguments, key):
            budget = replace(budget, **{key: True})
    if arguments.node_only:
        budget = replace(budget, edge_state=False, widths="paired")
    device = torch.device(arguments.device) if arguments.device else None
    if arguments.h1:
        for algorithm in arguments.algorithms:
            h1_table(algorithm, budget)
        return 0
    if arguments.h4:
        h4_table(arguments.algorithms, budget)
        return 0
    if arguments.heads:
        head_table(arguments.algorithms, budget)
        return 0
    if arguments.tracking:
        tracking_table(arguments.algorithms, budget)
        return 0
    if arguments.ladder and arguments.table:
        ladder_table(arguments.algorithms, budget)
        return 0
    if arguments.ladder:
        for algorithm in arguments.algorithms:
            ladder_report(algorithm, budget, arguments.seeds, device)
        return 0
    if arguments.table:
        written, ladders = {}, {}
        for algorithm in arguments.algorithms:
            path = ARTIFACTS / f"{budget.tag}-{algorithm}-report.json"
            if path.exists():
                written[algorithm] = json.loads(path.read_text())
            path = ARTIFACTS / f"{budget.tag}-{algorithm}-ladder.json"
            if path.exists():
                ladders[algorithm] = json.loads(path.read_text())
        tabulate(written, ladders)
        print(f"  {len(written)} of {len(arguments.algorithms)} rows written")
        return 0
    found = {}
    for algorithm in arguments.algorithms:
        found[algorithm] = report(algorithm, budget, arguments.seeds, device)
    tabulate(found)
    print(json.dumps({name: one["summary"] for name, one in found.items()},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
