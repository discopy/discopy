# -*- coding: utf-8 -*-

"""
Score the study with the benchmark's own code, not ours.

    python verify.py --predict --seed 0     # needs torch
    python verify.py --score                # needs the `clrs` package

``--predict`` trains a matcher exactly as ``train.py`` does and writes
its predictions as one-hot vectors over **all** the nodes of each
sample — text, pattern and non-alignment positions included, so the
model competes on the benchmark's full output space — beside the cached
ground truth.  ``--score`` reads them back in the other environment and
averages ``clrs._src.evaluation._eval_one``, the function behind every
published ``kmp_matcher`` number, over each split.  The two halves
share nothing but the ``npz``, so a bug in our accuracy could not
survive into the score this prints.
"""

from __future__ import annotations

import argparse

import numpy as np

from dataset import DATA_DIR, SPLITS, load


def predict(seed: int):
    import torch
    from train import run
    _, matcher = run(seed, epochs=15, batch_size=64, lr=1e-3)
    for split in ("val", "test", "wide"):
        cache = load(split)
        keys = torch.as_tensor(cache["keys"], dtype=torch.long)
        text, pattern = int(cache["text"]), int(cache["pattern"])
        nodes = SPLITS[split]["length"]
        with torch.no_grad():
            scores = matcher(keys, text, pattern)
        onehot = np.zeros((len(keys), nodes), dtype=np.float32)
        onehot[np.arange(len(keys)), scores.argmax(-1).numpy()] = 1
        np.savez_compressed(DATA_DIR / f"kmp-{split}-predictions.npz",
                            predictions=onehot)
        print(f"{split}: wrote {onehot.shape} one-hot predictions")


def score():
    from clrs._src.evaluation import _eval_one
    for split in ("val", "test", "wide"):
        cache = load(split)
        with np.load(DATA_DIR / f"kmp-{split}-predictions.npz") as file:
            predictions = file["predictions"]
        nodes = SPLITS[split]["length"]
        truth = np.zeros((len(cache["match"]), nodes), dtype=np.float32)
        truth[np.arange(len(truth)), cache["match"]] = 1
        scores = [_eval_one(prediction, target)
                  for prediction, target in zip(predictions, truth)]
        print(f"{split}: clrs._src.evaluation._eval_one = "
              f"{float(np.mean(scores)):.4f} over {len(scores)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--score", action="store_true")
    arguments = parser.parse_args()
    if arguments.predict:
        predict(arguments.seed)
    if arguments.score:
        score()
