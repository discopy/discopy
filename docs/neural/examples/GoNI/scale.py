# -*- coding: utf-8 -*-

"""
Scale the trained kmp matcher past the benchmark: ``n = 128`` and ``256``.

    python scale.py

The benchmark's out-of-distribution split stops at ``n = 64``, where
``train.py`` reports 100.0 over three seeds.  This script retrains the
same three seeds under the same protocol — the weights were not kept —
and evaluates each selected model on the cached scale splits, the
benchmark's own sampler drawn at ``n = 128`` and ``n = 256``, sizes
where no baseline has numbers.  One evaluator instance owns the
compiled maps, so the one-off circuit build at each size is shared
across the seeds; this time the weights are saved beside the report,
so a later evaluation is a load, not a rerun.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import model as goni
import train
from dataset import SCALE, load

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def main(seeds, epochs, batch_size, lr) -> dict:
    evaluator = goni.Matcher()
    results = {}
    for seed in seeds:
        started = time.time()
        _, matcher = train.run(seed, epochs, batch_size, lr)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        torch.save(matcher.state_dict(), ARTIFACTS / f"kmp-seed{seed}.pt")
        evaluator.load_state_dict(matcher.state_dict())
        entry = {}
        for split in SCALE:
            entry[split] = goni.accuracy(evaluator, load(split))
            print(f"seed {seed} {split}: {entry[split]:.3f}", flush=True)
        entry["minutes"] = (time.time() - started) / 60
        results[seed] = entry
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    arguments = parser.parse_args()
    results = main(arguments.seeds, arguments.epochs,
                   arguments.batch_size, arguments.lr)
    with open(ARTIFACTS / "kmp-scale.json", "w") as stream:
        json.dump(results, stream, indent=2)
