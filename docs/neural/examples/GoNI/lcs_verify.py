# -*- coding: utf-8 -*-

"""
Score the LCS study with the benchmark's own code, not ours.

    python lcs_verify.py --predict --seed 0     # needs torch
    python lcs_verify.py --score                # needs the `clrs` package

``--predict`` trains a grid exactly as ``lcs_train.py`` does and writes
its predicted direction per grid cell beside the cached ground truth.
``--score`` reads them back in the other environment, lays *both* sides
out with ``clrs._src.probing.strings_pair_cat`` — the very builder that
produced every published ``lcs_length`` target, masked blanks included —
and averages ``clrs._src.evaluation._eval_one`` over each split.  The
two halves share nothing but the ``npz``, so a bug in our accuracy
could not survive into the score this prints.
"""

from __future__ import annotations

import argparse

import numpy as np

from lcs_dataset import DATA_DIR, load


def predict(seed: int):
    import torch
    from lcs_train import run
    _, grid = run(seed, epochs=50, batch_size=64, lr=1e-3)
    for split in ("val", "test", "wide"):
        cache = load(split)
        keys = torch.as_tensor(cache["keys"], dtype=torch.long)
        m, n = int(cache["x_len"]), int(cache["y_len"])
        predictions = []
        with torch.no_grad():
            for start in range(0, len(keys), 32):
                logits = grid(keys[start:start + 32], m, n)
                predictions.append(logits.argmax(-1).numpy().astype(np.int8))
        predictions = np.concatenate(predictions)
        np.savez_compressed(DATA_DIR / f"lcs-{split}-predictions.npz",
                            predictions=predictions)
        print(f"{split}: wrote {predictions.shape} predicted directions")


def score():
    from clrs._src.evaluation import _eval_one
    from clrs._src.probing import strings_pair_cat
    for split in ("val", "test", "wide"):
        cache = load(split)
        with np.load(DATA_DIR / f"lcs-{split}-predictions.npz") as file:
            predictions = file["predictions"]
        scores = [
            _eval_one(strings_pair_cat(prediction.astype(float), 3),
                      strings_pair_cat(target.astype(float), 3))
            for prediction, target in zip(predictions, cache["b"])]
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
