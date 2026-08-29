# -*- coding: utf-8 -*-

"""
The benchmark's ``kmp_matcher`` samples, generated once and cached.

    python dataset.py --generate      # needs the `clrs` package
    python dataset.py --check         # needs numpy and nothing else

The splits are the benchmark's own — ``clrs.build_sampler`` with the
counts, lengths and seeds of ``clrs._src.samplers.CLRS30``, plus the
wider out-of-distribution split of the ``CLRS_small`` example — so the
cache holds the benchmark's samples and not a re-draw of its
distribution.  A sample is a string of characters over a four-letter
alphabet, the first ``text`` of them the haystack and the last
``pattern`` the needle, which the sampler always embeds; the answer is
the first position where the needle occurs.  The study trains output
only, the benchmark's no-hint setting, so the trajectories are not
cached: a split is its keys and its answers.

Generation imports ``clrs``, which brings jax and tensorflow with it;
training imports ``torch``.  The ``npz`` cache is the interface between
the two environments, and ``--check`` re-derives every cached answer
from the keys by brute force, so nothing is trusted twice.
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

#: Past the benchmark: the same sampler drawn at sizes where no baseline
#: has numbers, with fresh seeds.  ``n = 256`` costs sixteen ``n = 64``
#: forwards per sample, hence the smaller split.
SCALE = {
    "n128": {"num_samples": 128, "length": 128, "seed": 4},
    "n256": {"num_samples": 32, "length": 256, "seed": 5},
}

#: The alphabet of ``clrs._src.samplers.MatcherSampler``.
CHARS = 4


def lengths(length: int) -> tuple[int, int]:
    """ The text and pattern lengths of a sample of ``length`` nodes,
    following ``MatcherSampler._sample_data``. """
    pattern = 1 if length < 5 else length // 5
    return length - pattern, pattern


def path(split: str) -> Path:
    return DATA_DIR / f"kmp-{split}.npz"


def generate():
    """ Draw every split with ``clrs`` and write the cache. """
    import clrs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split, spec in {**SPLITS, **SCALE}.items():
        sampler, _ = clrs.build_sampler("kmp_matcher", **spec)
        feedback = sampler.next(spec["num_samples"])
        inputs = {dp.name: dp.data for dp in feedback.features.inputs}
        outputs = {dp.name: dp.data for dp in feedback.outputs}
        text, pattern = lengths(spec["length"])
        assert (inputs["string"].sum(-1) == pattern).all()
        assert (inputs["string"][:, :text] == 0).all()
        keys = inputs["key"].argmax(-1).astype(np.int8)
        match = outputs["match"].argmax(-1).astype(np.int16)
        np.savez_compressed(
            path(split), keys=keys, match=match,
            text=np.int16(text), pattern=np.int16(pattern))
        print(f"{split}: {keys.shape[0]} samples, "
              f"text {text}, pattern {pattern} -> {path(split)}")


def load(split: str) -> dict:
    """ One cached split, as arrays. """
    with np.load(path(split)) as cache:
        return {name: cache[name] for name in cache.files}


def first_match(keys: np.ndarray, text: int, pattern: int) -> np.ndarray:
    """ The first occurrence of each sample's needle, by brute force. """
    windows = np.stack(
        [keys[:, i:i + pattern] for i in range(text - pattern + 1)], axis=1)
    hits = (windows == keys[:, None, text:]).all(-1)
    return hits.argmax(-1)


def check():
    """ Re-derive every cached answer from the cached keys. """
    for split, spec in {**SPLITS, **SCALE}.items():
        cache = load(split)
        text, pattern = int(cache["text"]), int(cache["pattern"])
        assert (text, pattern) == lengths(spec["length"])
        derived = first_match(cache["keys"], text, pattern)
        assert (derived == cache["match"]).all()
        print(f"{split}: {len(cache['match'])} answers check out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.generate:
        generate()
    if arguments.check:
        check()
