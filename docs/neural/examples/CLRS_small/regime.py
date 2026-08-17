# -*- coding: utf-8 -*-

"""
The probe that decides ``config.REGIME``, and nothing else.

    python regime.py --algorithms bfs           # score one row's two arms
    python regime.py --write                    # freeze the decision

Part 2 adopted mixed training sizes as a blanket protocol and the
``minimum`` control killed that: at 200 trajectories of one size the row
holds at 0.8281 and at 1000 trajectories of five sizes it collapses to
0.1719.  So size mixing is destructive on at least one task, it cannot be
a default, and every row of Part 3 owes its own decision -- made here,
recorded in ``artifacts/regime.json``, and frozen before the campaign
that reads it.

The rule is :data:`~config.REGIME`'s and it is pre-registered rather than
chosen once the numbers are in.  Two clauses, both conservative, because
the probe is **one seed an arm** and Part 2's protocol is the incumbent:

* mixed has to beat fixed on the out-of-distribution score by more than
  :data:`~config.REGIME_MARGIN` times the row's own Part 2 seed standard
  error -- the only run-to-run noise estimate this study owns;
* and it must not cost the **order-free** heads, whatever it does to the
  headline.  A regime that buys pointer points by giving up the part of
  the algorithm the processor demonstrably computes is not a regime this
  study adopts on one seed.

Otherwise the row stays ``"fixed"``.
"""

from __future__ import annotations

import argparse
import json

import torch

import dataset
import evaluate as evaluations
import model as zoo
import train as training
from config import (
    ARTIFACTS, EXECUTORS, H2_ARMS, REGIME_FILE, REGIME_MARGIN, WIDTHS)
from dataclasses import replace


def arms(algorithm: str, seed: int = 0, device=None) -> dict:
    """
    The probe's two arms, scored on the wide out-of-distribution split at
    the depth their own trajectory asks for.

    Parameters:
        algorithm : The row.
        seed : The seed both arms were trained at.
        device : Where to score.
    """
    device = training.default_device() if device is None else device
    splits = dataset.load_all(algorithm)
    wide = zoo.Batches(splits["wide"], H2_ARMS["R"].eval_batch_size, device)
    valid = zoo.Batches(splits["val"], H2_ARMS["R"].eval_batch_size, device)
    found = {}
    for name, mixed in (("fixed", False), ("mixed", True)):
        budget = replace(H2_ARMS["R"], mixed=mixed)
        path = training.artifact_of(algorithm, budget, seed)
        if not path.exists():
            found[name] = None
            continue
        model = zoo.build(
            algorithm, WIDTHS[budget.widths], pool=budget.pool,
            pointer=budget.pointer, settle=budget.settle, probe=budget.probe,
            solver=budget.solver, backward=budget.backward)
        zoo.load_checkpoint(model, path)
        model = model.to(device)
        found[name] = {
            "tag": budget.tag,
            "in_distribution": zoo.evaluate_split(model, valid),
            "ood": zoo.evaluate_split(model, wide),
            "hints": {"val": evaluations.hint_curve(model, valid),
                      "ood": evaluations.hint_curve(model, wide)}}
        del model
        torch.cuda.empty_cache()
    return found


def decide(algorithm: str, found: dict, noise: float) -> dict:
    """
    :data:`~config.REGIME`'s pre-registered rule, applied to one row.

    Parameters:
        algorithm : The row.
        found : Its two arms, as :func:`arms` returned them.
        noise : The row's Part 2 seed standard error.
    """
    if not found.get("fixed") or not found.get("mixed"):
        return {"regime": None, "why": "an arm is missing"}
    split = {name: evaluations.head_split(
        {"algorithm": algorithm, "seeds": [{
            "hints": one["hints"], "in_distribution": one["in_distribution"],
            "ood_wide": one["ood"]}]}) for name, one in found.items()}
    gain = found["mixed"]["ood"]["score"] - found["fixed"]["ood"]["score"]
    free = {name: (one["free"] or {}).get("ood")
            for name, one in split.items()}
    kept = free["mixed"] is None or free["fixed"] is None \
        or free["mixed"] >= free["fixed"]
    regime = "mixed" if gain > REGIME_MARGIN * noise and kept else "fixed"
    return {
        "regime": regime, "gain": round(gain, 4),
        "margin": round(REGIME_MARGIN * noise, 4),
        "order_free_ood": {name: None if one is None else round(one, 4)
                           for name, one in free.items()},
        "order_free_kept": bool(kept),
        "ood": {name: round(one["ood"]["score"], 4)
                for name, one in found.items()},
        "in_distribution": {name: round(one["in_distribution"]["score"], 4)
                            for name, one in found.items()},
        "why": f"mixed {'beats' if gain > REGIME_MARGIN * noise else 'loses'} "
               f"by more than {REGIME_MARGIN} seed s.e.m. "
               f"({round(REGIME_MARGIN * noise, 4)}) and "
               f"{'keeps' if kept else 'costs'} the order-free heads"}


def noise_of(algorithm: str) -> float:
    """
    The row's Part 2 seed standard error on the wide split, read off its
    recorded report -- the only run-to-run noise estimate this study owns.
    """
    path = ARTIFACTS / f"full-max-{algorithm}-report.json"
    if not path.exists():
        return 0.05
    return json.loads(path.read_text())["summary"]["ood_wide"]["sem"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="*", default=EXECUTORS)
    parser.add_argument("--write", action="store_true",
                        help="freeze the decision into artifacts/regime.json")
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args(argv)
    training.single_threaded()
    device = torch.device(arguments.device) if arguments.device else None

    rows = {}
    for algorithm in arguments.algorithms:
        found = arms(algorithm, device=device)
        rows[algorithm] = decide(algorithm, found, noise_of(algorithm))
        print(f"{algorithm:22} {rows[algorithm]['regime'] or '—':6} "
              f"{rows[algorithm].get('why', '')}", flush=True)
    if arguments.write:
        stored = json.loads(REGIME_FILE.read_text()) \
            if REGIME_FILE.exists() else {"rows": {}}
        # a decision is frozen once: a row already recorded is never
        # rewritten, which is what makes "declared before, frozen after"
        # a property of the file rather than of anyone's discipline.
        for name, one in rows.items():
            if one["regime"] and name not in stored["rows"]:
                stored["rows"][name] = one
        stored["rule"] = (
            f"fixed unless mixed beats it on the wide out-of-distribution "
            f"score by more than {REGIME_MARGIN} x the row's Part 2 seed "
            f"s.e.m., and never when it costs the order-free heads; "
            f"pre-registered in config.REGIME before any probe was scored")
        REGIME_FILE.write_text(json.dumps(stored, indent=2))
        frozen = {k: v["regime"] for k, v in stored["rows"].items()}
        print(f"\nfrozen into {REGIME_FILE.name}: {frozen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
