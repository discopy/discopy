# -*- coding: utf-8 -*-

"""
Phase 1: the published floor, reproduced locally in the reference
implementation, plus the hints-on / no-hint contrast inside it.

This file runs in the **dm-clrs venv** (`/scratch/tommaso.salvatori/
dm-clrs`), not in the study's own environment: it trains DeepMind's
`clrs` BaselineModel, not a `MapNN`.  It exists for two reasons and
does nothing else:

* **the gate** — if the 2022 recipe cannot reproduce `bellman_ford`
  0.9201 and `bfs` 0.9989 here, within noise, every comparison against
  the anchor column changes meaning and Phase 3 stops;
* **the replication** — Part 3's R-vs-O finding (output-only training
  rescues `dijkstra` and `mst_prim` out of distribution) is either a
  fact about hint supervision or a fact about our pipeline.  Running
  hints-on against no-hint in the *reference* harness decides which,
  and positions the result against Rodionov & Prokhorenkova 2023 and
  Mahdavi et al. 2023, who both report the no-hint side.

The recipe is the 2022 paper's, not the repo's modern defaults, and
every choice is named here so a stranger can check it against the
paper's appendix: MPNN processor (max aggregation, fully connected),
hidden 128, layer norm; noisy teacher forcing 0.5 with *hard* hint
feedback; Adam at 1e-3, batch 32, 10 000 steps, **no** gradient
clipping, default (LeCun) encoder init; the fixed 1000-trajectory
CLRS30 dataset at n = 16, early stopping on the 32-trajectory
validation split; OOD test at n = 64 on the benchmark's own fixed
samples.  The one knob the paper does not pin is the validation
cadence; we evaluate every 100 steps (100 evaluations per run) and
record the choice in the artefact.

The no-hint arm (`O_ref`) sets `encode_hints=False, decode_hints=False`:
no hint loss and no hint feedback, the same model otherwise.  The
number of processor steps still comes from the trajectory's own length,
which the features carry regardless (`nets.py` reads `hints[0].data.
shape[0] - 1` for steps whether or not hints are encoded), so the two
arms run the same depth.  Teacher forcing is meaningless without
feedback, so the arm's `hint_teacher_forcing` is 0 and that is not a
second axis: it is the same axis, hints-off.

Artefacts: ``artifacts/floor-<arm>-<algorithm>-report.json`` with one
row per seed, per-output-probe scores, per-hint scores (hints-on arm),
and the val curve; written after every seed so a killed run keeps what
it measured.

Usage (from this directory, inside the dm-clrs venv)::

    python floor.py --algorithms bellman_ford bfs --arm R
    python floor.py --algorithms dijkstra mst_prim --arm R O
    python floor.py --smoke          # 60 steps, one seed, CPU-friendly
"""

import argparse
import functools
import json
import os
import time
from pathlib import Path

# Be a good citizen on a *shared, bursty* GPU: no BFC pool at all --
# the model is a few MB, and a pool grab races the neighbours' own
# allocation bursts (two runs died that way before this line existed).
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import tensorflow as tf  # noqa: E402
# The tfds reader must never touch the GPU; it will happily reserve it.
tf.config.set_visible_devices([], "GPU")

import jax  # noqa: E402
import numpy as np  # noqa: E402

import clrs  # noqa: E402
from clrs._src import decoders as clrs_decoders  # noqa: E402
from clrs._src import processors  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
#: The published archive itself (CLRS30_v1.0.0.tar.gz from the GCS
#: bucket), not a re-generation: dataset provenance ends here.  Our
#: cached splits were verified sample-for-sample equal to it --
#: see ``artifacts/floor-dataset-provenance.json``.
DATASET = "/scratch/tommaso.salvatori/clrs30-v1.0.0"

#: The 2022 recipe, named once.  `grad_clip 0.0` and `encoder_init
#: "default"` are the floor's absences of the Ibarz-era stabilizers;
#: see artifacts/parity.md rows b and d.
RECIPE = dict(learning_rate=1e-3, batch_size=32, train_steps=10_000,
              eval_every=100, hidden_dim=128, use_ln=True,
              hint_teacher_forcing=0.5, hint_repred_mode="hard",
              grad_clip_max_norm=0.0, encoder_init="default",
              dropout_prob=0.0, use_lstm=False, nb_msg_passing_steps=1)

SEEDS = (0, 1, 2)

#: What each arm means.  One axis: whether hints exist for the model.
ARMS = {
    "R": dict(encode_hints=True, decode_hints=True),
    "O": dict(encode_hints=False, decode_hints=False),
}


def batches_of(algorithm: str, split: str, batch_size: int):
    """The benchmark's own fixed split, as an iterator of feedbacks."""
    ds, num_samples, spec = clrs.create_dataset(
        folder=DATASET, algorithm=algorithm, split=split,
        batch_size=batch_size)
    return ds.as_numpy_iterator(), num_samples, spec


def collect_and_eval(model, algorithm: str, split: str, rng_key,
                     batch_size: int = 32, with_hints: bool = False):
    """
    Pool the whole split's predictions, score once — the reference's
    own protocol (v1.0.0 run.py `collect_and_eval`), which is also this
    study's (`model.evaluate_split`).
    """
    iterator, num_samples, spec = batches_of(algorithm, split, batch_size)
    seen, preds, outs = 0, [], []
    hints, lengths, hint_preds = [], [], []
    while seen < num_samples:
        feedback = next(iterator)
        rng_key, key = jax.random.split(rng_key)
        found, aux = model.predict(key, feedback.features,
                                   return_hints=with_hints)
        preds.append(found)
        outs.append(feedback.outputs)
        if with_hints:
            hints.append(feedback.features.hints)
            lengths.append(feedback.features.lengths)
            # 2.0.3's predict returns the hint predictions directly: a
            # list over time of per-probe dicts (v1.0.0 wrapped it).
            hint_preds.append(aux)
        seen += batch_size
    concat = lambda xs, axis: jax.tree_util.tree_map(
        lambda *x: np.concatenate(x, axis), *xs)
    scores = clrs.evaluate(concat(outs, 0), concat(preds, 0))
    if with_hints:
        stacked = [clrs_decoders.postprocess(
            spec, step, sinkhorn_temperature=0.1, sinkhorn_steps=25,
            hard=True) for step in concat(hint_preds, 0)]
        scores.update(clrs.evaluate_hints(
            concat(hints, 1), concat(lengths, 0), stacked))
    return {name: (value.tolist() if hasattr(value, "tolist") else value)
            for name, value in scores.items()
            if not name.endswith("_along_time")}


def train_one(algorithm: str, arm: str, seed: int, steps: int,
              log=print) -> dict:
    """One seed of one arm: train, keep best-val, restore, score OOD."""
    recipe = dict(RECIPE, **ARMS[arm])
    train_iter, _, spec = batches_of(
        algorithm, "train", recipe["batch_size"])
    dummy_iter, _, _ = batches_of(algorithm, "train", recipe["batch_size"])
    dummy = next(dummy_iter)

    processor_factory = processors.get_processor_factory(
        "mpnn", use_ln=recipe["use_ln"], nb_triplet_fts=0)
    model = clrs.models.BaselineModel(
        spec=[spec], dummy_trajectory=[dummy],
        processor_factory=processor_factory,
        hidden_dim=recipe["hidden_dim"],
        encode_hints=recipe["encode_hints"],
        decode_hints=recipe["decode_hints"],
        encoder_init=recipe["encoder_init"],
        use_lstm=recipe["use_lstm"],
        learning_rate=recipe["learning_rate"],
        grad_clip_max_norm=recipe["grad_clip_max_norm"],
        checkpoint_path="/tmp/clrs_floor",
        dropout_prob=recipe["dropout_prob"],
        hint_teacher_forcing=(recipe["hint_teacher_forcing"]
                              if recipe["decode_hints"] else 0.0),
        hint_repred_mode=recipe["hint_repred_mode"],
        nb_msg_passing_steps=recipe["nb_msg_passing_steps"])
    model.init(dummy.features, seed + 1)

    rng_key = jax.random.PRNGKey(seed)
    best = {"val": -1.0, "step": -1, "params": None}
    curve = []
    started = time.time()
    for step in range(steps):
        feedback = next(train_iter)
        rng_key, key = jax.random.split(rng_key)
        loss = model.feedback(key, feedback)
        if (step + 1) % RECIPE["eval_every"] == 0 or step + 1 == steps:
            rng_key, key = jax.random.split(rng_key)
            val = collect_and_eval(model, algorithm, "val", key)["score"]
            curve.append({"step": step + 1, "loss": float(loss),
                          "val": val})
            if val > best["val"]:
                best = {"val": val, "step": step + 1,
                        "params": jax.tree_util.tree_map(
                            np.copy, model.params)}
    if best["params"] is not None:
        model.params = best["params"]
    rng_key, key = jax.random.split(rng_key)
    ood = collect_and_eval(model, algorithm, "test", key,
                           with_hints=ARMS[arm]["decode_hints"])
    rng_key, key = jax.random.split(rng_key)
    val = collect_and_eval(model, algorithm, "val", key,
                           with_hints=ARMS[arm]["decode_hints"])
    n_params = sum(int(np.prod(np.shape(leaf)))
                   for leaf in jax.tree_util.tree_leaves(model.params))
    log(f"  {algorithm}/{arm}/seed{seed}: best val {best['val']:.4f} at "
        f"step {best['step']}, ood {ood['score']:.4f}, "
        f"{time.time() - started:.0f}s")
    return {"seed": seed, "parameters": n_params,
            "best_val": best["val"], "best_step": best["step"],
            "in_distribution": val, "ood": ood, "val_curve": curve,
            "seconds": time.time() - started}


def report(algorithm: str, arm: str, seeds, steps: int, log=print) -> dict:
    """All seeds of one arm, summarised, written after every seed."""
    path = ARTIFACTS / f"floor-{arm}-{algorithm}-report.json"
    rows = []
    for seed in seeds:
        rows.append(train_one(algorithm, arm, seed, steps, log=log))
        scores = [row["ood"]["score"] for row in rows]
        found = {
            "algorithm": algorithm, "arm": arm,
            "what": ("the 2022 floor recipe in the reference "
                     "implementation" if arm == "R" else
                     "the same recipe with hints off entirely"),
            "recipe": {key: value for key, value in RECIPE.items()},
            "arm_flags": ARMS[arm],
            "dataset": "the published CLRS30 tfds splits, "
                       f"downloaded to {DATASET}",
            "environment": {
                "clrs": getattr(clrs, "__version__", "2.0.3"),
                "jax": jax.__version__,
                "device": str(jax.devices()[0]),
            },
            "seeds": rows,
            "summary": {
                "ood_mean": float(np.mean(scores)),
                "ood_std": float(np.std(scores, ddof=1))
                if len(scores) > 1 else None,
                "ood_sem": float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
                if len(scores) > 1 else None,
            },
        }
        path.write_text(json.dumps(found, indent=2))
    log(f"  {algorithm}/{arm}: -> {path.name}")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+",
                        default=["bellman_ford", "bfs"])
    parser.add_argument("--arm", nargs="+", default=["R"],
                        choices=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--steps", type=int,
                        default=RECIPE["train_steps"])
    parser.add_argument("--smoke", action="store_true",
                        help="60 steps, one seed: exercises every code "
                             "path in minutes")
    arguments = parser.parse_args(argv)
    seeds = [arguments.seeds[0]] if arguments.smoke else arguments.seeds
    steps = 60 if arguments.smoke else arguments.steps
    for algorithm in arguments.algorithms:
        for arm in arguments.arm:
            report(algorithm, arm, seeds, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
