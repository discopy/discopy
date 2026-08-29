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

from pytest import importorskip, raises, skip

from discopy.neural import Dim, Network

torch = importorskip("torch")

GONI = Path(__file__).resolve().parents[2] / "docs" / "neural" \
    / "examples" / "GoNI"

#: The module names the examples of ``docs/neural`` share: import them
#: under a saved ``sys.modules`` and put back whatever was there, so two
#: examples in one pytest session don't shadow each other's ``model``.
SHARED = ("circuits", "dataset", "model", "lcs_dataset", "lcs_model")


def _example(directory: Path) -> dict:
    saved = {name: sys.modules.pop(name, None) for name in SHARED}
    sys.path.insert(0, str(directory))
    try:
        return {name: importlib.import_module(name) for name in SHARED}
    finally:
        sys.path.remove(str(directory))
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


EXAMPLE = _example(GONI)
circuits, goni_dataset, goni_model, lcs_dataset, lcs_model = (
    EXAMPLE[name] for name in SHARED)


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


def exact_matcher() -> tuple[Network, Network]:
    """ The matcher's step and select on scalar wires, as exact networks. """
    class Step(torch.nn.Module):
        def forward(self, x):
            s, q, c = (x[..., i] for i in range(3))
            zero = torch.zeros_like(s)
            return torch.stack(
                [zero] * 3 + [c, q, s * (q == c)], dim=-1)

    class Select(torch.nn.Module):
        def forward(self, x):
            m, f = x[..., 0], x[..., 1]
            zero = torch.zeros_like(m)
            return torch.stack(
                [zero] * 2 + [torch.maximum(f, m), m * (1 - f)], dim=-1)

    return Network('step', Dim(1) ** 3, Dim(1) ** 3, module=Step()), \
        Network('select', Dim(1) ** 2, Dim(1) ** 2, module=Select())


def test_match_permutations_are_wiring():
    step, select = exact_matcher()
    for text, pattern in ((3, 1), (5, 2), (13, 3)):
        alignments = text - pattern + 1
        boxes = circuits.match(text, pattern, step, select).to_map().boxes
        assert boxes == alignments * pattern * (step, ) \
            + alignments * (select, )


def test_match_computes():
    step, select = exact_matcher()
    generator = torch.Generator().manual_seed(1)
    for text, pattern in ((3, 1), (5, 2), (13, 3), (12, 5)):
        alignments = text - pattern + 1
        grid = circuits.match(text, pattern, step, select).to_map()
        keys = torch.randint(0, 3, (16, text + pattern),
                             generator=generator).float()
        embed = torch.randint(0, alignments, (16, ), generator=generator)
        for sample in range(16):
            position = int(embed[sample])
            keys[sample, position:position + pattern] = \
                keys[sample, text:]
        x = torch.ones(16, alignments)
        for j in range(pattern):
            x = torch.cat([x, keys[:, text + j:text + j + 1],
                           keys[:, j:j + 1]], dim=-1)
        x = torch.cat([x, keys[:, pattern:text],
                       torch.zeros(16, 1)], dim=-1)
        outputs = grid(x, n_rounds=text + pattern + 2)
        first = torch.tensor([next(
            i for i in range(alignments)
            if (keys[sample, i:i + pattern] == keys[sample, text:]).all())
            for sample in range(16)])
        scores = torch.stack([
            outputs[:, circuits.match_answer(text, pattern, i)]
            for i in range(alignments)], dim=-1)
        assert (scores.argmax(-1) == first).all()
        assert (scores.sum(-1) == 1).all()


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


def test_lcs_readout_directions():
    """ The per-cell readout of ``lcs_model.Grid`` reads each cell's true
    inputs: with the exact cell and the exact direction rule, it
    reproduces the benchmark's traceback on random words. """
    import numpy as np
    cell = exact_cell()
    generator = torch.Generator().manual_seed(7)
    for m, n in ((3, 4), (5, 5), (8, 8)):
        cmap = circuits.lcs(m, n, cell).to_map()
        widths = cmap.port_widths
        offsets, total = [], 0
        for width in widths:
            offsets.append(total)
            total += width
        index = torch.tensor([
            [k for port in cmap.box_ports(box)[:5]
             for k in range(offsets[port], offsets[port] + widths[port])]
            for box in range(m * n)])
        words = torch.randint(0, 4, (16, m + n), generator=generator).float()
        x = torch.zeros(16, 3 * (m + n))
        for i in range(m):
            x[:, 3 * (m - 1 - i) + 1] = words[:, i]
        for j in range(n):
            x[:, 3 * (m + j) + 1] = words[:, m + j]
        flat = cmap(x, n_rounds=m + n + 2, return_flat=True)
        gathered = flat[:, index.reshape(-1)].reshape(16, m * n, 5)
        a, b, _, up, left = (gathered[..., i] for i in range(5))
        directions = torch.where(
            a == b, torch.zeros_like(a),
            torch.where(up >= left, torch.ones_like(a),
                        2 * torch.ones_like(a)))
        expected = lcs_dataset.directions(
            words.numpy().astype(np.int8), m, n)
        assert (directions.reshape(16, m, n).numpy() == expected).all()


def test_grid_learns():
    if not (GONI / "data" / "lcs-train.npz").exists():
        skip("the lcs cache is not built; see lcs_dataset.py --generate")
    torch.manual_seed(0)
    grid = lcs_model.Grid(state=8, char=4, hidden=16)
    split = lcs_dataset.load("train")
    keys = torch.as_tensor(split["keys"][:128], dtype=torch.long)
    b = torch.as_tensor(split["b"][:128], dtype=torch.long)
    m, n = int(split["x_len"]), int(split["y_len"])
    optimizer = torch.optim.Adam(grid.parameters(), lr=1e-2)
    losses = []
    for _ in range(50):
        loss = torch.nn.functional.cross_entropy(
            grid(keys, m, n).reshape(-1, 3), b.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 2


def test_matcher_learns():
    if not (GONI / "data" / "kmp-train.npz").exists():
        skip("the kmp cache is not built; see dataset.py --generate")
    torch.manual_seed(0)
    matcher = goni_model.Matcher(state=8, char=4, hidden=16)
    split = goni_dataset.load("train")
    keys = torch.as_tensor(split["keys"][:128], dtype=torch.long)
    match = torch.as_tensor(split["match"][:128], dtype=torch.long)
    text, pattern = int(split["text"]), int(split["pattern"])
    optimizer = torch.optim.Adam(matcher.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        loss = torch.nn.functional.cross_entropy(
            matcher(keys, text, pattern), match)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 2
