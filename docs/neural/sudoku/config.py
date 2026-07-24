# -*- coding: utf-8 -*-

"""
Paths, seeds, matched widths and the experiment budgets.

A :class:`core.study.Budget` bundles everything a training run is allowed
to spend: data, epochs, batch size and depth. :data:`QUICK` runs end to end in a few
minutes on one GPU and is a faithful miniature of :data:`FULL`, the budget
behind the recorded baseline results -- everything about the two is
identical except the amounts, so the small run exercises exactly the code
paths of the large one.
"""

from __future__ import annotations

from pathlib import Path

from core.study import Budget, Widths  # noqa: F401 -- re-exported

#: The size of the sudoku grid and its number of cells.
N = 9
N_CELLS = 81

#: The task packages live under ``docs/neural``; everything they read and
#: write -- the benchmarks, the checkpoints, the figures -- lives beside
#: them, so the whole study is one relocatable directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "sudoku_data" / "rrn"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"

for _directory in (DATA_DIR, ARTIFACTS, FIGURES):
    _directory.mkdir(parents=True, exist_ok=True)

#: The seeds every model is trained with, in order.
SEEDS = (0, 1, 2)

#: The learning rates of the per-model grid search on the validation split.
LR_GRID = (3e-4, 1e-3, 3e-3)

MODELS = ("goi", "rrn", "trm")


QUICK = Budget(
    name="quick",
    n_train=5000, n_valid=1500, n_test=3000,
    epochs=4, batch_size=64, rounds=8,
    trm_n=4, trm_T=2, trm_n_sup=4)

FULL = Budget(
    name="full",
    n_train=50000, n_valid=6000, n_test=18000,
    epochs=8, batch_size=128, rounds=20,
    trm_n=6, trm_T=3, trm_n_sup=8,
    lr_n_train=12000)


#: Widths chosen by :func:`sudoku.models.match_widths` so that the three
#: models have trainable parameter counts within 10% of each other.
WIDTHS = {
    "goi": Widths(dim=24, state_dim=96, hidden=192),
    "rrn": Widths(dim=24, state_dim=96, hidden=172),
    "trm": Widths(dim=24, state_dim=88, hidden=172, y_dim=48),
}

#: The parameter count the three models are tuned to match.
PARAM_TARGET = 205_000

#: Learning rates used when the grid search has not been run.
DEFAULT_LR = {"goi": 1e-3, "rrn": 1e-3, "trm": 1e-3}

#: Gradient-norm clipping, identical for the three models.
GRAD_CLIP = 1.0
