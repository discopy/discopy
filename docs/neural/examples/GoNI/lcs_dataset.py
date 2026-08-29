# -*- coding: utf-8 -*-

"""
The benchmark's ``lcs_length`` samples, generated once and cached.

    python lcs_dataset.py --generate      # needs the `clrs` package
    python lcs_dataset.py --check         # needs numpy and nothing else

The splits are the benchmark's own, the same specification as the
``kmp_matcher`` cache of ``dataset.py``.  A sample is two strings over
a four-letter alphabet, each half the node count; the output is not the
length but the traceback: one direction per cell of the
dynamic-programming grid — 0 for a diagonal move on a match, 1 for up,
2 for left — the ``b`` matrix of ``clrs._src.algorithms.lcs_length``,
scored cell by cell.  The benchmark computes it by relaxation to a
fixpoint with its own boundary rules; ``--check`` re-derives every
cached matrix from the keys with the classic zero-boundary scan under
the uniform rule "diagonal on a match, else up when ``up >= left``",
which agrees with the benchmark's everywhere — the check *is* the proof,
run over every cached sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

#: The benchmark's split specification, verbatim from
#: ``clrs._src.samplers.CLRS30``, and the wider out-of-distribution
#: split of ``examples/CLRS_small``.
SPLITS = {
    "train": {"num_samples": 1000, "length": 16, "seed": 1},
    "val": {"num_samples": 32, "length": 16, "seed": 2},
    "test": {"num_samples": 32, "length": 64, "seed": 3},
    "wide": {"num_samples": 128, "length": 64, "seed": 30},
}

#: The alphabet of ``clrs._src.samplers.LCSSampler``.
CHARS = 4


def lengths(length: int) -> tuple[int, int]:
    """ The two string lengths of a sample of ``length`` nodes,
    following ``LCSSampler._sample_data``. """
    return length - length // 2, length // 2


def path(split: str) -> Path:
    return DATA_DIR / f"lcs-{split}.npz"


def generate():
    """ Draw every split with ``clrs`` and write the cache. """
    import clrs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, spec in SPLITS.items():
        sampler, _ = clrs.build_sampler("lcs_length", **spec)
        feedback = sampler.next(spec["num_samples"])
        inputs = {dp.name: dp.data for dp in feedback.features.inputs}
        outputs = {dp.name: dp.data for dp in feedback.outputs}
        x_len, y_len = lengths(spec["length"])
        assert (inputs["string"][:, :x_len] == 0).all()
        assert (inputs["string"][:, x_len:] == 1).all()
        keys = inputs["key"].argmax(-1).astype(np.int8)
        block = outputs["b"][:, :x_len, x_len:, :]
        assert (block[..., :3].sum(-1) == 1).all() and (block[..., 3] == 0).all()
        b = block.argmax(-1).astype(np.int8)
        np.savez_compressed(
            path(split), keys=keys, b=b,
            x_len=np.int16(x_len), y_len=np.int16(y_len))
        print(f"{split}: {keys.shape[0]} samples, "
              f"grid {x_len} x {y_len} -> {path(split)}")


def load(split: str) -> dict:
    """ One cached split, as arrays. """
    with np.load(path(split)) as cache:
        return {name: cache[name] for name in cache.files}


def directions(keys: np.ndarray, x_len: int, y_len: int) -> np.ndarray:
    """ The traceback matrix of each sample, by the classic scan. """
    x, y = keys[:, :x_len], keys[:, x_len:]
    table = np.zeros((len(keys), x_len + 1, y_len + 1), dtype=np.int32)
    b = np.zeros((len(keys), x_len, y_len), dtype=np.int8)
    for i in range(1, x_len + 1):
        for j in range(1, y_len + 1):
            match = x[:, i - 1] == y[:, j - 1]
            diag, up, left = (table[:, i - 1, j - 1], table[:, i - 1, j],
                              table[:, i, j - 1])
            table[:, i, j] = np.where(match, diag + 1, np.maximum(up, left))
            b[:, i - 1, j - 1] = np.where(match, 0, np.where(up >= left, 1, 2))
    return b


def check():
    """ Re-derive every cached traceback from the cached keys. """
    for split, spec in SPLITS.items():
        cache = load(split)
        x_len, y_len = int(cache["x_len"]), int(cache["y_len"])
        assert (x_len, y_len) == lengths(spec["length"])
        derived = directions(cache["keys"], x_len, y_len)
        assert (derived == cache["b"]).all()
        print(f"{split}: {len(cache['b'])} tracebacks check out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.generate:
        generate()
    if arguments.check:
        check()
