# -*- coding: utf-8 -*-

"""
The GoNI grid: the circuit of :func:`circuits.lcs` with one learned cell,
and a per-cell readout for the benchmark's traceback output.

The characters pass through the cell unchanged and the learned part is
the value ``v = g(a, b, diag, up, left)`` alone, written on all three
value outputs the way the exact cell writes ``L[i][j]``.  The benchmark
does not score the corner value but the traceback: one direction per
grid cell, a pure function of that cell's *inputs* — diagonal on a
match, else up against left.  So the readout asks the map for its flat
port state (``return_flat``) and reads each cell's incoming messages
off its domain ports, one gather indexed at compile time; a direction
head turns them into three logits per cell.  The circuit is untouched:
the readout is an address into the state, not a wire.

One size is one compiled map: the grid for ``(m, n)`` is built and
converted once, and every sample of a split rides the batch axis of the
same message passing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

from discopy.neural import Dim, Network

sys.path.insert(0, str(Path(__file__).resolve().parent))
circuits = importlib.import_module("circuits")
lcs_dataset = importlib.import_module("lcs_dataset")
sys.path.remove(str(Path(__file__).resolve().parent))


class Cell(torch.nn.Module):
    """ ``(a, b, diag, up, left) -> (b, v, v, a, v)`` with ``v`` learned. """
    def __init__(self, state: int, char: int, hidden: int):
        super().__init__()
        self.state, self.char = state, char
        self.value = torch.nn.Sequential(
            torch.nn.Linear(2 * char + 3 * state, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, state))

    def forward(self, x):
        state, char = self.state, self.char
        incoming = x[..., :2 * char + 3 * state]
        a, b = (incoming[..., i * char:(i + 1) * char] for i in range(2))
        value = self.value(incoming)
        return torch.cat(
            [torch.zeros_like(incoming), b, value, value, a, value], dim=-1)


class Grid(torch.nn.Module):
    """
    The trainable study model: encoders, the cell, the direction head.

    Calling it on a batch of keys compiles (or reuses) the map for the
    split's grid and returns three logits per cell.
    """
    def __init__(self, state: int = 32, char: int = 16, hidden: int = 64):
        super().__init__()
        self.state, self.char = state, char
        self.embedding = torch.nn.Embedding(lcs_dataset.CHARS, char)
        self.boundary = torch.nn.Parameter(torch.zeros(state))
        self.cell = Cell(state, char, hidden)
        self.direction = torch.nn.Sequential(
            torch.nn.Linear(2 * char + 3 * state, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 3))
        self.maps = {}

    def compile(self, m: int, n: int):
        """ The map of the grid at one size and the flat indices of each
        cell's domain ports, both built once. """
        if (m, n) not in self.maps:
            s, c = Dim(self.state), Dim(self.char)
            cell = Network('cell', c @ c @ s @ s @ s, c @ s @ s @ c @ s,
                           module=self.cell)
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
            self.maps[m, n] = cmap, index
        return self.maps[m, n]

    def forward(self, keys: torch.Tensor, m: int, n: int):
        cmap, index = self.compile(m, n)
        batch_size = keys.shape[0]
        chars = self.embedding(keys)
        boundary = self.boundary.expand(batch_size, self.state)
        blocks = []
        for k in range(m):
            blocks += [boundary, chars[:, m - 1 - k], boundary]
        for j in range(n):
            blocks += [boundary, chars[:, m + j], boundary]
        flat = cmap(torch.cat(blocks, dim=-1), n_rounds=m + n + 2,
                    return_flat=True)
        gathered = flat[:, index.reshape(-1)].reshape(
            batch_size, m * n, 2 * self.char + 3 * self.state)
        return self.direction(gathered).reshape(batch_size, m, n, 3)


def accuracy(model: Grid, split: dict, batch_size: int = 32) -> float:
    """ The benchmark's score for the ``b`` output: the fraction of grid
    cells whose predicted direction is exact, as in
    ``clrs._src.evaluation._eval_one`` over the scored block. """
    keys = torch.as_tensor(split["keys"], dtype=torch.long)
    b = torch.as_tensor(split["b"], dtype=torch.long)
    m, n = int(split["x_len"]), int(split["y_len"])
    hits = []
    with torch.no_grad():
        for start in range(0, len(keys), batch_size):
            logits = model(keys[start:start + batch_size], m, n)
            hits.append(
                (logits.argmax(-1) == b[start:start + batch_size]).flatten())
    return torch.cat(hits).float().mean().item()
