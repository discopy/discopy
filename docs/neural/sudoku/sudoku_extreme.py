# -*- coding: utf-8 -*-

"""
The sudoku-extreme benchmark of the Hierarchical Reasoning Model,
preprocessed into the same arrays as :mod:`sudoku.data` so that either
dataset can feed the three solvers unchanged.

The source is the dataset distributed by the authors,
`sapientinc/sudoku-extreme
<https://huggingface.co/datasets/sapientinc/sudoku-extreme>`_: ``train.csv``
and ``test.csv``, each row a ``source, puzzle, solution, rating`` line with
``.`` for a blank cell.  :func:`build` downloads the two files, draws a
seeded subsample of :data:`N_BASE` training puzzles and pre-generates three
augmented training sets from them, following the recipe of the authors'
``build_sudoku_dataset.py``:

* ``sudoku_extreme_standard`` -- every puzzle plus :data:`N_AUG` ``["standard"]
  = 1000`` augmentations drawn from the full sudoku symmetry group:
  relabeling the nine digits, permuting the three bands and the rows inside
  each band, permuting the three stacks and the columns inside each stack,
  and transposing the grid with probability one half.  ``1000 * 1001 =
  1,001,000`` training examples.

* ``sudoku_extreme_special`` -- every puzzle plus ``N_AUG["special"] = 100``
  augmentations, each the composition of exactly one transformation from
  each of two generating sets: the transposition, which maps row-units to
  column-units, and a non-identity digit relabeling.  ``1000 * 101 =
  101,000`` training examples.

* ``sudoku_extreme_special_large`` -- the special recipe at the standard
  size: every puzzle plus ``N_AUG["special_large"] = 1000`` transposed,
  relabeled boards.  ``1000 * 1001 = 1,001,000`` training examples.

Every group keeps its untransformed original (augmentation index ``0``), and
its augmented boards are pairwise distinct and distinct from the original --
checked on cell contents, not just on the sampled transformations; on top
of that, :func:`check_artifacts` asserts that each variant's whole training
set is globally duplicate-free, across groups as well as within them.  The
stored order is a seeded shuffle, so that a prefix -- which is what
:meth:`sudoku.data.Split.subsample` takes -- is balanced across the
base puzzles; ``puzzle_id`` and ``aug_id`` columns keep the provenance.

The other two splits are shared by all the variants and are not augmented:
``valid`` is a further :data:`N_VALID` held-out puzzles sampled from
``train.csv`` disjointly from the base subsample, and ``test`` is the
authors' complete test file, shuffled with its row indices recorded.
:func:`load` returns the usual ``{"train", "valid", "test"}`` dictionary of
:class:`sudoku.data.Split` objects::

    from sudoku import sudoku_extreme
    splits = sudoku_extreme.load("standard")    # or "special"

in place of ``sudoku.data.load()``.  As everywhere else in this folder,
boards are ``uint8`` arrays of shape ``(n, 81)`` with ``0`` for a blank;
note that the authors' repository instead shifts digits by one to reserve
``0`` for padding, so their checkpoints are *not* label-compatible.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date

import numpy as np

from sudoku.config import N, N_CELLS, ROOT
from sudoku.data import Split

#: The authors' dataset on the hugging face hub.
REPO = "sapientinc/sudoku-extreme"

#: Everything this module writes lives here, a sibling of the other data.
DIR = ROOT / "sudoku_data" / "sudoku_extreme"
RAW = DIR / "raw"

#: The single seed behind the subsample, the augmentations and the shuffles.
SEED = 0

#: The number of base training puzzles, as in the authors' 1k subsample.
N_BASE = 1000

#: The number of held-out validation puzzles, matching the old benchmark.
N_VALID = 18000

#: Augmentations per puzzle; each group also keeps its original, so the
#: variants hold ``N_BASE * (1 + N_AUG[name])`` training examples.
N_AUG = {"standard": 1000, "special": 100, "special_large": 1000}

FILES = {name: DIR / f"sudoku_extreme_{name}.npz" for name in N_AUG}
COMMON = DIR / "common.npz"


# --- fetching and parsing the source CSVs ---------------------------------

def fetch(force: bool = False) -> dict:
    """
    Download ``train.csv`` and ``test.csv`` into :data:`RAW`, returning
    their paths.

    Parameters:
        force : Whether to download again even if the files are cached.
    """
    from huggingface_hub import hf_hub_download
    RAW.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("train", "test"):
        target = RAW / f"{name}.csv"
        if force or not target.exists():
            hf_hub_download(REPO, f"{name}.csv", repo_type="dataset",
                            local_dir=RAW, force_download=force)
        paths[name] = target
    return paths


def _count_rows(path) -> int:
    """ The number of data rows of a CSV, excluding the header. """
    with open(path, newline="") as stream:
        return sum(1 for _ in stream) - 1


def _boards(strings: list) -> np.ndarray:
    """ Parse 81-character board strings, ``.`` for blank, to ``(n, 81)``. """
    flat = np.frombuffer(
        "".join(strings).replace(".", "0").encode(), np.uint8)
    return (flat - ord("0")).reshape(len(strings), N_CELLS)


def _read_rows(path, wanted=None):
    """
    Stream one CSV, keeping the 0-based data rows in ``wanted`` (all rows
    when ``None``), in file order.

    Returns:
        ``rows, puzzles, solutions, ratings, sources`` where ``rows`` are
        the kept 0-based row indices.
    """
    if wanted is not None:
        wanted = set(int(index) for index in wanted)
    rows, puzzles, solutions, ratings, sources = [], [], [], [], []
    with open(path, newline="") as stream:
        reader = csv.reader(stream)
        next(reader)                                          # header
        for index, row in enumerate(reader):
            if wanted is not None and index not in wanted:
                continue
            source, puzzle, solution, rating = row
            assert len(puzzle) == N_CELLS and len(solution) == N_CELLS, \
                f"{path}:{index}: malformed board"
            rows.append(index)
            puzzles.append(puzzle)
            solutions.append(solution)
            ratings.append(int(rating))
            sources.append(source)
    assert wanted is None or len(rows) == len(wanted), \
        f"{path}: found {len(rows)} of {len(wanted)} wanted rows"
    return (np.array(rows, np.int64), _boards(puzzles), _boards(solutions),
            np.array(ratings, np.int32), np.array(sources))


# --- the two augmentation recipes -----------------------------------------

def _permutations(rng: np.random.Generator, m: int, k: int) -> np.ndarray:
    """ ``m`` independent uniform permutations of ``range(k)``. """
    return rng.permuted(np.tile(np.arange(k), (m, 1)), axis=1)


def _standard_maps(rng: np.random.Generator, m: int):
    """
    ``m`` draws from the full symmetry group, exactly as the authors'
    ``shuffle_sudoku``: shuffle the three bands and the rows inside each,
    likewise stacks and columns, transpose with probability one half, and
    relabel the digits, keeping ``0`` (blank) fixed.

    Returns:
        ``mapping`` of shape ``(m, 81)`` with ``new[i] = old[mapping[i]]``
        on the transposed-or-not board, ``transpose`` booleans of shape
        ``(m, )`` and ``digits`` of shape ``(m, 10)`` with column 0 zero.
    """
    rows = (3 * _permutations(rng, m, 3)[:, :, None]
            + _permutations(rng, 3 * m, 3).reshape(m, 3, 3)).reshape(m, N)
    cols = (3 * _permutations(rng, m, 3)[:, :, None]
            + _permutations(rng, 3 * m, 3).reshape(m, 3, 3)).reshape(m, N)
    mapping = (N * rows[:, :, None] + cols[:, None, :]).reshape(m, N_CELLS)
    transpose = rng.random(m) < 0.5
    digits = np.concatenate([np.zeros((m, 1), np.uint8),
                             (1 + _permutations(rng, m, N)).astype(np.uint8)],
                            axis=1)
    return mapping, transpose, digits


def _special_maps(rng: np.random.Generator, m: int):
    """
    ``m`` draws of the restricted recipe: *always* transpose, composed with
    a uniform non-identity digit relabeling. Same return type as
    :func:`_standard_maps`, with the identity cell mapping.
    """
    identity = np.arange(N)
    perms = np.empty((m, N), np.int64)
    filled = 0
    while filled < m:
        perm = rng.permutation(N)
        if (perm == identity).all():
            continue
        perms[filled] = perm
        filled += 1
    digits = np.concatenate(
        [np.zeros((m, 1), np.uint8), (1 + perms).astype(np.uint8)], axis=1)
    return np.tile(np.arange(N_CELLS), (m, 1)), np.ones(m, bool), digits


def _apply(board: np.ndarray, mapping, transpose, digits) -> np.ndarray:
    """ Apply ``m`` sampled transformations to one ``(81, )`` board. """
    grid = board.reshape(N, N)
    both = np.stack([grid.reshape(-1), grid.T.reshape(-1)])
    base = both[transpose.astype(np.int64)]
    moved = np.take_along_axis(base, mapping, axis=1)
    return np.take_along_axis(digits, moved.astype(np.int64), axis=1)


def _augment_group(puzzle, solution, rng, m, maps_fn):
    """
    One puzzle's group: the original followed by ``m`` augmentations that
    are pairwise distinct and distinct from the original as ``(puzzle,
    solution)`` cell contents, resampling any collision.
    """
    out_p = np.empty((1 + m, N_CELLS), np.uint8)
    out_s = np.empty((1 + m, N_CELLS), np.uint8)
    out_p[0], out_s[0] = puzzle, solution
    seen = {puzzle.tobytes() + solution.tobytes()}
    filled, attempts = 1, 0
    while filled < 1 + m:
        attempts += 1
        if attempts > 64:
            raise RuntimeError(
                "could not draw enough distinct augmentations")
        mapping, transpose, digits = maps_fn(rng, 1 + m - filled)
        for new_p, new_s in zip(_apply(puzzle, mapping, transpose, digits),
                                _apply(solution, mapping, transpose, digits)):
            key = new_p.tobytes() + new_s.tobytes()
            if key not in seen:
                seen.add(key)
                out_p[filled], out_s[filled] = new_p, new_s
                filled += 1
    return out_p, out_s


# --- building the artifacts -----------------------------------------------

def _build_variant(name, base_p, base_s, rng, order_rng, log):
    """ Generate, shuffle and save one variant, returning its arrays. """
    m = N_AUG[name]
    total = N_BASE * (1 + m)
    all_p = np.empty((total, N_CELLS), np.uint8)
    all_s = np.empty((total, N_CELLS), np.uint8)
    maps_fn = _standard_maps if name == "standard" else _special_maps
    for i in range(N_BASE):
        group_p, group_s = _augment_group(base_p[i], base_s[i], rng, m,
                                          maps_fn)
        all_p[i * (1 + m):(i + 1) * (1 + m)] = group_p
        all_s[i * (1 + m):(i + 1) * (1 + m)] = group_s
        if (i + 1) % 200 == 0:
            log(f"  {name}: {i + 1}/{N_BASE} puzzles augmented")
    puzzle_id = np.repeat(np.arange(N_BASE, dtype=np.int32), 1 + m)
    aug_id = np.tile(np.arange(1 + m, dtype=np.int32), N_BASE)
    order = order_rng.permutation(total)
    stored = {"puzzles": all_p[order], "solutions": all_s[order],
              "puzzle_id": puzzle_id[order], "aug_id": aug_id[order]}
    np.savez_compressed(FILES[name], **stored)
    log(f"  {name}: saved {total:,} examples to {FILES[name].name}")
    return stored


def build(force: bool = False, log=print) -> None:
    """
    Download the benchmark, generate the variants and verify everything.

    All randomness -- which puzzles are subsampled, which transformations
    are drawn, and the stored order -- comes from :data:`SEED`, so the
    artifacts are reproducible.  Whatever is already on disk is kept and
    only the missing variants are generated, replaying the shared shuffle
    stream over the cached parts, so that an incremental build and a full
    rebuild produce identical arrays.

    Parameters:
        force : Whether to re-download and rebuild even when cached.
    """
    if not force and COMMON.exists() and all(
            path.exists() for path in FILES.values()):
        _refresh_meta(log)
        return
    # stream 3 is the shared shuffle; 4 was appended when special_large was
    # added, so streams 0-2 -- and with them the first two variants -- are
    # bit-identical to what the first release of this module built.
    (sample_rng, standard_rng, special_rng, order_rng,
     special_large_rng) = map(
        np.random.default_rng, np.random.SeedSequence(SEED).spawn(5))
    variant_rngs = {"standard": standard_rng, "special": special_rng,
                    "special_large": special_large_rng}

    if force or not COMMON.exists():
        paths = fetch(force)
        log("counting train.csv ...")
        count = _count_rows(paths["train"])
        assert count >= N_BASE + N_VALID
        picked = sample_rng.choice(count, N_BASE + N_VALID, replace=False)
        log(f"subsampling {N_BASE} base + {N_VALID} valid "
            f"of {count:,} puzzles")
        rows, puzzles, solutions, ratings, sources = _read_rows(
            paths["train"], picked)
        position = {int(row): i for i, row in enumerate(rows)}

        def take(row_subset):
            index = np.array([position[int(row)] for row in row_subset])
            return (rows[index], puzzles[index], solutions[index],
                    ratings[index], sources[index])

        base = take(picked[:N_BASE])
        valid = take(order_rng.permutation(picked[N_BASE:]))

        log("parsing test.csv ...")
        test = _read_rows(paths["test"])
        shuffle = order_rng.permutation(len(test[0]))
        test = tuple(array[shuffle] for array in test)

        meta = json.dumps({
            "repo": REPO, "seed": SEED, "built": date.today().isoformat(),
            "train_rows": count, "test_rows": len(test[0]),
            "n_base": N_BASE, "n_valid": N_VALID, "n_aug": N_AUG})
        np.savez_compressed(
            COMMON, meta=np.array(meta),
            base_rows=base[0], base_puzzles=base[1], base_solutions=base[2],
            base_ratings=base[3], base_sources=base[4],
            valid_rows=valid[0], valid_puzzles=valid[1],
            valid_solutions=valid[2], valid_ratings=valid[3],
            test_rows=test[0], test_puzzles=test[1], test_solutions=test[2],
            test_ratings=test[3])
        log(f"saved common splits to {COMMON.name}")
        base_p, base_s = base[1], base[2]
    else:
        _refresh_meta(log)
        common = np.load(COMMON)
        base_p, base_s = common["base_puzzles"], common["base_solutions"]
        order_rng.permutation(N_VALID)                    # replay valid
        order_rng.permutation(len(common["test_rows"]))   # replay test

    for name in N_AUG:
        if not force and FILES[name].exists():
            order_rng.permutation(N_BASE * (1 + N_AUG[name]))   # replay
            continue
        log(f"building sudoku_extreme_{name} "
            f"({N_BASE} x (1 + {N_AUG[name]}) examples) ...")
        _build_variant(name, base_p, base_s, variant_rngs[name],
                       order_rng, log)

    check_artifacts(log)


def _refresh_meta(log=print) -> None:
    """
    Rewrite the stored metadata when the variant list has changed since the
    common splits were built, leaving every array untouched.
    """
    common = np.load(COMMON)
    meta = json.loads(common["meta"].item())
    if meta["n_aug"] == N_AUG:
        return
    meta["n_aug"] = N_AUG
    arrays = {key: common[key] for key in common.files}
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(COMMON, **arrays)
    log(f"refreshed {COMMON.name} metadata with the current variant list")


# --- verification ----------------------------------------------------------

def _assert_boards(puzzles, solutions, where: str) -> None:
    """ Dtypes, digit ranges and clue-solution consistency, exhaustively. """
    assert puzzles.dtype == solutions.dtype == np.uint8, where
    assert puzzles.shape == solutions.shape, where
    assert puzzles.shape[1] == N_CELLS, where
    assert puzzles.max() <= N, f"{where}: puzzle digit out of range"
    assert ((solutions >= 1) & (solutions <= N)).all(), \
        f"{where}: solution digit out of range"
    assert ((puzzles == 0) | (puzzles == solutions)).all(), \
        f"{where}: a given disagrees with its solution"


def _assert_valid(solutions, where: str) -> None:
    """ Every row, column and box of every solution holds 1..9 once. """
    grids = solutions.reshape(-1, N, N)
    target = np.arange(1, N + 1, dtype=np.uint8)
    for view in (grids, grids.transpose(0, 2, 1),
                 grids.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4)
                 .reshape(-1, N, N)):
        assert (np.sort(view, axis=2) == target).all(), \
            f"{where}: a solution breaks the rules"


def _content_keys(puzzles, solutions) -> np.ndarray:
    """ One opaque row key per ``(puzzle, solution)`` pair. """
    combined = np.ascontiguousarray(
        np.concatenate([puzzles, solutions], axis=1))
    return combined.view(np.dtype((np.void, combined.shape[1]))).reshape(-1)


def _check_special_group(group_p, group_s) -> None:
    """
    Every augmented row of one group must be exactly the transposed
    original under some non-identity digit relabeling, all distinct.
    """
    transposed_p = group_p[0].reshape(N, N).T.reshape(-1)
    transposed_s = group_s[0].reshape(N, N).T.reshape(-1)
    augmented_p, augmented_s = group_p[1:], group_s[1:]
    m = len(augmented_s)
    first = np.array([np.argmax(transposed_s == d)
                      for d in range(1, N + 1)])
    sigma = augmented_s[:, first]
    assert (np.sort(sigma, axis=1)
            == np.arange(1, N + 1, dtype=np.uint8)).all(), \
        "special: relabeling is not a permutation"
    assert not (sigma == np.arange(1, N + 1, dtype=np.uint8)
                ).all(axis=1).any(), "special: identity relabeling"
    assert len(np.unique(sigma, axis=0)) == m, \
        "special: repeated relabeling"
    padded = np.concatenate([np.zeros((m, 1), np.uint8), sigma], axis=1)
    assert (padded[:, transposed_s] == augmented_s).all(), \
        "special: solution is not transpose-plus-relabel"
    assert (padded[:, transposed_p] == augmented_p).all(), \
        "special: puzzle is not transpose-plus-relabel"


def check_artifacts(log=print) -> None:
    """
    Exhaustively verify the artifacts on disk: exact counts and group
    structure, originals matching the base puzzles, global pairwise
    distinctness of each variant's whole training set, solution validity
    and clue consistency of every example, givens preserved by every
    augmentation, the transpose-plus-relabel structure of the special
    variants, and disjointness of the splits.
    """
    common = np.load(COMMON)
    base_p, base_s = common["base_puzzles"], common["base_solutions"]
    for prefix in ("base", "valid", "test"):
        puzzles = common[f"{prefix}_puzzles"]
        solutions = common[f"{prefix}_solutions"]
        _assert_boards(puzzles, solutions, prefix)
        _assert_valid(solutions, prefix)
    assert len(base_p) == N_BASE and len(common["valid_puzzles"]) == N_VALID
    overlap = set(map(int, common["base_rows"])) \
        & set(map(int, common["valid_rows"]))
    assert not overlap, "base and valid share a train.csv row"
    test_keys = set(
        _content_keys(common["test_puzzles"], common["test_solutions"])
        .tolist())
    for prefix in ("base", "valid"):
        keys = _content_keys(common[f"{prefix}_puzzles"],
                             common[f"{prefix}_solutions"])
        assert not any(key in test_keys for key in keys.tolist()), \
            f"{prefix}: a puzzle also appears in the test split"
    log("common splits: OK "
        f"(base {N_BASE}, valid {N_VALID}, test {len(test_keys):,})")

    base_givens = (base_p > 0).sum(1)
    for name, m in N_AUG.items():
        stored = np.load(FILES[name])
        puzzles, solutions = stored["puzzles"], stored["solutions"]
        puzzle_id, aug_id = stored["puzzle_id"], stored["aug_id"]
        assert len(puzzles) == N_BASE * (1 + m), f"{name}: wrong total"
        assert (np.bincount(puzzle_id, minlength=N_BASE) == 1 + m).all(), \
            f"{name}: wrong group size"
        assert (np.bincount(aug_id, minlength=1 + m) == N_BASE).all(), \
            f"{name}: wrong augmentation indexing"
        _assert_boards(puzzles, solutions, name)
        _assert_valid(solutions, name)
        assert ((puzzles > 0).sum(1) == base_givens[puzzle_id]).all(), \
            f"{name}: an augmentation changed the number of givens"
        unique = len(np.unique(_content_keys(puzzles, solutions)))
        assert unique == len(puzzles), \
            f"{name}: {len(puzzles) - unique} duplicate examples"
        order = np.lexsort((aug_id, puzzle_id))
        grouped_p = puzzles[order].reshape(N_BASE, 1 + m, N_CELLS)
        grouped_s = solutions[order].reshape(N_BASE, 1 + m, N_CELLS)
        assert (grouped_p[:, 0] == base_p).all() \
            and (grouped_s[:, 0] == base_s).all(), \
            f"{name}: original of a group differs from its base puzzle"
        if name.startswith("special"):
            for i in range(N_BASE):
                _check_special_group(grouped_p[i], grouped_s[i])
        log(f"sudoku_extreme_{name}: OK ({len(puzzles):,} examples, "
            f"{N_BASE} groups of {1 + m}, globally distinct)")


# --- loading ---------------------------------------------------------------

def load(variant: str = "standard", verify: bool = True) -> dict:
    """
    The three splits, building the artifacts on first use.

    Parameters:
        variant : ``"standard"``, ``"special"`` or ``"special_large"``
                  (the full names ``"sudoku_extreme_standard"`` etc. are
                  also accepted).
        verify : Whether to check counts, group structure, consistency of
                 every example and validity of a sample, mirroring
                 :func:`sudoku.data.load`.
    """
    name = variant.replace("sudoku_extreme_", "")
    if name not in N_AUG:
        raise ValueError(f"unknown variant {variant!r}")
    build()
    stored, common = np.load(FILES[name]), np.load(COMMON)
    splits = {
        "train": Split("train", stored["puzzles"], stored["solutions"]),
        "valid": Split("valid", common["valid_puzzles"],
                       common["valid_solutions"]),
        "test": Split("test", common["test_puzzles"],
                      common["test_solutions"])}
    if verify:
        m = N_AUG[name]
        assert len(splits["train"]) == N_BASE * (1 + m)
        assert (np.bincount(stored["puzzle_id"], minlength=N_BASE)
                == 1 + m).all()
        for split in splits.values():
            _assert_boards(split.puzzles, split.solutions, split.name)
            sample = slice(0, min(len(split), 2000))
            _assert_valid(split.solutions[sample], split.name)
    return splits


if __name__ == "__main__":
    build(force="--force" in sys.argv)
    if "--check" in sys.argv:
        check_artifacts()
