# -*- coding: utf-8 -*-

"""
The GoNI matcher: the circuit of :func:`circuits.match` with two learned
cells, and the encoders around it.

The characters pass through a ``step`` unchanged — they are carried, the
way an edge cell of ``examples/CLRS_small`` carries its per-edge inputs —
and the learned part of a step is the fold ``s' = g(s, q, c)`` alone, so
the depth of an alignment only ever iterates ``g``.  A ``select`` learns
both its heads, the flag ``f' = h_f(m, f)`` and the answer wire
``out = h_o(m, f)``.  The encoders are a character embedding and two
learned initial vectors; the decoder is one linear head reading a score
off each alignment's ``out``.

One size is one compiled map: the circuit for ``(text, pattern)`` is
built and converted once, and every sample of a split rides the batch
axis of the same message passing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

from discopy.neural import Dim, Network

sys.path.insert(0, str(Path(__file__).resolve().parent))
circuits = importlib.import_module("circuits")
dataset = importlib.import_module("dataset")
sys.path.remove(str(Path(__file__).resolve().parent))


class Step(torch.nn.Module):
    """ ``(s, q, c) -> (c, q, s')`` with ``s' = g(s, q, c)`` learned. """
    def __init__(self, state: int, char: int, hidden: int):
        super().__init__()
        self.state, self.char = state, char
        self.fold = torch.nn.Sequential(
            torch.nn.Linear(state + 2 * char, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, state))

    def forward(self, x):
        state, char = self.state, self.char
        incoming = x[..., :state + 2 * char]
        s, q, c = torch.split(incoming, [state, char, char], dim=-1)
        return torch.cat(
            [torch.zeros_like(incoming), c, q, self.fold(incoming)], dim=-1)


class Select(torch.nn.Module):
    """ ``(m, f) -> (f', out)`` with both heads learned. """
    def __init__(self, state: int, flag: int, hidden: int):
        super().__init__()
        self.state, self.flag = state, flag
        self.heads = torch.nn.Sequential(
            torch.nn.Linear(state + flag, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, flag + state))

    def forward(self, x):
        incoming = x[..., :self.state + self.flag]
        return torch.cat(
            [torch.zeros_like(incoming), self.heads(incoming)], dim=-1)


class Matcher(torch.nn.Module):
    """
    The trainable study model: encoders, the two cells, the score head.

    Calling it on a batch of keys compiles (or reuses) the map for the
    split's ``(text, pattern)`` and returns one score per alignment.
    """
    def __init__(self, state: int = 32, char: int = 16, hidden: int = 64):
        super().__init__()
        self.state, self.char = state, char
        self.embedding = torch.nn.Embedding(dataset.CHARS, char)
        self.initial = torch.nn.Parameter(torch.zeros(state))
        self.flag = torch.nn.Parameter(torch.zeros(state))
        self.step = Step(state, char, hidden)
        self.select = Select(state, state, hidden)
        self.score = torch.nn.Linear(state, 1)
        self.maps = {}

    def compile(self, text: int, pattern: int):
        """ The map of the circuit at one size, built once. """
        if (text, pattern) not in self.maps:
            s, c = Dim(self.state), Dim(self.char)
            step = Network('step', s @ c @ c, c @ c @ s, module=self.step)
            select = Network('select', s @ s, s @ s, module=self.select)
            self.maps[text, pattern] = circuits.match(
                text, pattern, step, select).to_map()
        return self.maps[text, pattern]

    def forward(self, keys: torch.Tensor, text: int, pattern: int):
        cmap = self.compile(text, pattern)
        batch_size, alignments = keys.shape[0], text - pattern + 1
        chars = self.embedding(keys)
        x = [self.initial.repeat(alignments).expand(batch_size, -1)]
        for j in range(pattern):
            x += [chars[:, text + j], chars[:, j]]
        x += [chars[:, pattern:text].flatten(1),
              self.flag.expand(batch_size, self.state)]
        outputs = cmap(torch.cat(x, dim=-1), n_rounds=text + pattern + 2)
        first = (2 * pattern - 1) * self.char + self.state
        outs = torch.stack([
            outputs[:, first + (alignments - 1 - i) * self.state:
                    first + (alignments - i) * self.state]
            for i in range(alignments)], dim=1)
        return self.score(outs)[..., 0]


def accuracy(model: Matcher, split: dict, batch_size: int = 32) -> float:
    """ The benchmark's score for ``mask_one``: exact argmax match. """
    keys = torch.as_tensor(split["keys"], dtype=torch.long)
    match = torch.as_tensor(split["match"], dtype=torch.long)
    text, pattern = int(split["text"]), int(split["pattern"])
    hits = []
    with torch.no_grad():
        for start in range(0, len(keys), batch_size):
            scores = model(keys[start:start + batch_size], text, pattern)
            hits.append(scores.argmax(-1) == match[start:start + batch_size])
    return torch.cat(hits).float().mean().item()
