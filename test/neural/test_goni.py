# -*- coding: utf-8 -*-

"""
The GoNI circuits are diagrams, and the diagrams are the algorithms.

The previous run of this study concluded that the LCS grid could not be a
diagram because it needs swaps, and wrote the wiring by hand instead.
These tests state the opposite: :func:`circuits.lcs` builds the grid as a
symmetric diagram whose crossings are permutation layers, ``to_map``
absorbs every one of them into wiring — the map holds one box per cell
and nothing else — and message passing over that map, with an exact cell,
computes the longest common subsequence.
"""

import importlib
import sys
from pathlib import Path

from pytest import importorskip, raises

from discopy.neural import Dim, Network

torch = importorskip("torch")

GONI = Path(__file__).resolve().parents[2] / "docs" / "neural" \
    / "examples" / "GoNI"

sys.path.insert(0, str(GONI))
circuits = importlib.import_module("circuits")
sys.path.remove(str(GONI))


def exact_cell() -> Network:
    """ The LCS transition on scalar wires, as an exact network. """
    class Cell(torch.nn.Module):
        def forward(self, x):
            a, b, diag, up, left = (x[..., i] for i in range(5))
            value = torch.where(a == b, diag + 1, torch.maximum(up, left))
            zero = torch.zeros_like(value)
            return torch.stack(
                [zero] * 5 + [b, value, value, a, value], dim=-1)

    return Network('cell', Dim(1) @ Dim(1) @ Dim(1) ** 3,
                   Dim(1) @ Dim(1) ** 2 @ Dim(1) @ Dim(1), module=Cell())


def reference(xs, ys) -> int:
    table = [[0] * (len(ys) + 1) for _ in range(len(xs) + 1)]
    for i, x in enumerate(xs, 1):
        for j, y in enumerate(ys, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if x == y\
                else max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def test_lcs_shape():
    cell = exact_cell()
    for m, n in ((1, 1), (2, 3), (4, 4)):
        grid = circuits.lcs(m, n, cell)
        assert grid.dom == Dim(1) ** (3 * m) @ Dim(1) ** (3 * n)
        assert grid.cod == Dim(1) ** (3 * n + 3 * m)
    with raises(ValueError):
        circuits.lcs(1, 1, Network('cell', Dim(1), Dim(1)))


def test_lcs_permutations_are_wiring():
    cell = exact_cell()
    for m, n in ((1, 2), (3, 3)):
        boxes = circuits.lcs(m, n, cell).to_map().boxes
        assert boxes == m * n * (cell, )


def test_lcs_computes():
    cell = exact_cell()
    generator = torch.Generator().manual_seed(0)
    for m, n in ((1, 1), (2, 3), (3, 2), (4, 4)):
        grid = circuits.lcs(m, n, cell).to_map()
        words = torch.randint(0, 3, (8, m + n), generator=generator).float()
        x = torch.zeros(8, 3 * (m + n))
        for i in range(m):
            x[:, 3 * (m - 1 - i) + 1] = words[:, i]
        for j in range(n):
            x[:, 3 * (m + j) + 1] = words[:, m + j]
        outputs = grid(x, n_rounds=m + n + 2)
        expected = torch.tensor([reference(
            word[:m].tolist(), word[m:].tolist()) for word in words])
        assert (outputs[:, circuits.answer(n)] == expected).all()
