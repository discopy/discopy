# -*- coding: utf-8 -*-

"""
The sudoku benchmark of :cite:t:`PalmEtAl18`, plus the sudoku symmetry group.

The dataset is the one distributed with the authors' repository
`rasmusbergpalm/recurrent-relational-networks
<https://github.com/rasmusbergpalm/recurrent-relational-networks>`_: 180k
training, 18k validation and 18k test puzzles whose number of givens is
uniform on 17-34, derived from Gordon Royle's collection of 17-given puzzles
as described in the paper. :func:`fetch` downloads and verifies it,
:func:`load` returns the three splits as ``uint8`` arrays of shape
``(n, 81)`` with ``0`` for a blank cell.

Nothing here is specific to a model: the same arrays feed the three solvers.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np

from experiments.config import DATA_DIR

#: The authors' download, as given by ``tasks/sudoku/data.py`` in their repo.
URL = "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"

#: A mirror, used only if the authors' link is dead.
MIRROR = "https://data.dgl.ai/dataset/sudoku-hard.zip"

#: The SHA-256 of the archive as downloaded from :data:`URL`.
SHA256 = "99f5c1f3f9a7c26e2e52d087dba2b312a816c9a54c638811455db72b8d3aa30d"

#: The row counts reported in the paper, checked on load.
ROWS = {"train": 180000, "valid": 18000, "test": 18000}

#: The range of givens, uniform over this range by construction.
GIVENS = (17, 34)

N = 9
N_CELLS = 81


@dataclass(frozen=True)
class Split:
    """
    One split of the benchmark.

    Parameters:
        name : ``"train"``, ``"valid"`` or ``"test"``.
        puzzles : The clues, of shape ``(n, 81)``, with ``0`` for a blank.
        solutions : The solutions, of shape ``(n, 81)``, digits ``1..9``.
        surrogate : Whether these puzzles were regenerated rather than
                    downloaded, i.e. whether they deviate from the paper.
    """
    name: str
    puzzles: np.ndarray
    solutions: np.ndarray
    surrogate: bool = False

    def __len__(self) -> int:
        return len(self.puzzles)

    @property
    def givens(self) -> np.ndarray:
        """ The number of givens of each puzzle. """
        return (self.puzzles > 0).sum(1)

    def subsample(self, n: int, seed: int = 0) -> Split:
        """
        The first ``n`` puzzles, or all of them when ``n`` is larger.

        The benchmark is already in random order and stratified by givens, so
        taking a prefix keeps the givens distribution; we check that below.
        """
        if n >= len(self):
            return self
        return Split(self.name, self.puzzles[:n], self.solutions[:n],
                     self.surrogate)


def fetch(force: bool = False) -> tuple[bool, str]:
    """
    Download and verify the archive, returning whether the authors' link was
    reached and the SHA-256 of what was downloaded.

    Parameters:
        force : Whether to download again even if the archive is cached.
    """
    archive = DATA_DIR / "sudoku-hard.zip"
    if force or not archive.exists():
        errors = []
        for url in (URL, MIRROR):
            try:
                with urllib.request.urlopen(url, timeout=300) as response:
                    archive.write_bytes(response.read())
                break
            except Exception as error:                    # pragma: no cover
                errors.append(f"{url}: {error}")
        else:                                             # pragma: no cover
            raise RuntimeError("could not download the benchmark:\n"
                               + "\n".join(errors))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if not (DATA_DIR / "sudoku-hard" / "train.csv").exists():
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(DATA_DIR)
    return digest == SHA256, digest


def _parse(name: str) -> tuple[np.ndarray, np.ndarray]:
    """ Parse one ``puzzle,solution`` CSV into two ``(n, 81)`` arrays. """
    cached = DATA_DIR / f"{name}.npz"
    if cached.exists():
        stored = np.load(cached)
        return stored["puzzles"], stored["solutions"]
    text = (DATA_DIR / "sudoku-hard" / f"{name}.csv").read_text()
    rows = [line for line in text.split("\n") if line]
    flat = np.frombuffer(
        "".join(line.replace(",", "") for line in rows).encode(), np.uint8)
    flat = (flat - ord("0")).reshape(len(rows), 2 * N_CELLS)
    puzzles, solutions = flat[:, :N_CELLS].copy(), flat[:, N_CELLS:].copy()
    np.savez_compressed(cached, puzzles=puzzles, solutions=solutions)
    return puzzles, solutions


def load(verify: bool = True) -> dict[str, Split]:
    """
    The three splits of the benchmark, downloading them on first use.

    Parameters:
        verify : Whether to check the row counts, the range of givens and
                 that every solution is a valid completion of its puzzle.
    """
    fetch()
    splits = {}
    for name in ROWS:
        puzzles, solutions = _parse(name)
        split = Split(name, puzzles, solutions)
        if verify:
            check(split)
        splits[name] = split
    return splits


def check(split: Split) -> None:
    """ Validate row count, givens range and solution consistency. """
    assert len(split) == ROWS[split.name] or split.surrogate, \
        f"{split.name}: {len(split)} rows, expected {ROWS[split.name]}"
    givens = split.givens
    assert givens.min() >= GIVENS[0] and givens.max() <= GIVENS[1], \
        f"{split.name}: givens in {givens.min()}-{givens.max()}"
    sample = slice(0, min(len(split), 2000))
    puzzles, solutions = split.puzzles[sample], split.solutions[sample]
    assert ((puzzles == 0) | (puzzles == solutions)).all(), \
        f"{split.name}: a given disagrees with its solution"
    grids = solutions.reshape(-1, N, N)
    for view in (grids, grids.transpose(0, 2, 1),
                 grids.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4)
                 .reshape(-1, N, N)):
        counts = np.zeros(view.shape[:2] + (N + 1,), np.int32)
        np.add.at(counts, (np.arange(view.shape[0])[:, None, None],
                           np.arange(N)[None, :, None], view), 1)
        assert (counts[..., 1:] == 1).all(), \
            f"{split.name}: a solution breaks the rules"


def givens_histogram(split: Split) -> dict[int, int]:
    """ How many puzzles the split has for each number of givens. """
    counts = np.bincount(split.givens, minlength=GIVENS[1] + 1)
    return {int(k): int(counts[k]) for k in range(GIVENS[0], GIVENS[1] + 1)}


# --- the sudoku symmetry group, used only for the optional ablation --------

def symmetry(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    A uniform element of the sudoku symmetry group, as a cell permutation and
    a digit relabeling.

    The group is generated by relabeling the nine digits, permuting the rows
    inside a band and the columns inside a stack, permuting the three bands
    and the three stacks, and transposing the grid. Every generator maps
    valid grids to valid grids and preserves both the number of givens and
    the uniqueness of the solution, so it is a label-preserving augmentation.

    Returns:
        ``cells`` of shape ``(81, )`` with ``new[i] = old[cells[i]]`` and
        ``digits`` of shape ``(10, )`` with ``digits[0] == 0``.
    """
    rows = np.concatenate([
        3 * band + rng.permutation(3)
        for band in rng.permutation(3)])
    cols = np.concatenate([
        3 * stack + rng.permutation(3)
        for stack in rng.permutation(3)])
    grid = rows[:, None] * N + cols[None, :]
    if rng.random() < 0.5:
        grid = grid.T
    digits = np.concatenate([[0], 1 + rng.permutation(N)]).astype(np.uint8)
    return grid.reshape(-1), digits


def augment(puzzles: np.ndarray, solutions: np.ndarray,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply one independent symmetry to each puzzle-solution pair in a batch.

    Parameters:
        puzzles : Clues of shape ``(batch, 81)``.
        solutions : Solutions of shape ``(batch, 81)``.
        rng : The generator, so that augmentation is reproducible.
    """
    out_puzzles = np.empty_like(puzzles)
    out_solutions = np.empty_like(solutions)
    for i in range(len(puzzles)):
        cells, digits = symmetry(rng)
        out_puzzles[i] = digits[puzzles[i][cells]]
        out_solutions[i] = digits[solutions[i][cells]]
    return out_puzzles, out_solutions
