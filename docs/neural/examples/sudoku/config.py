# -*- coding: utf-8 -*-

"""
Paths, widths, budgets and the recorded hyperparameters of every run in
`README.md <README.md>`_.

Everything here is a plain value: no torch, no diagrams, nothing that has
to be built.  A :class:`Widths` fixes the parameter count of a model, a
:class:`Budget` fixes what a training run is allowed to spend, and the two
recipe dictionaries at the end are the winning configurations copied
verbatim from the searches that found them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: The size of the sudoku grid and its number of cells.
N = 9
N_CELLS = 81

#: Everything this example reads and writes -- the benchmarks, the
#: checkpoints, the figures -- lives beside ``docs/neural``, so the whole
#: study is one relocatable directory.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "sudoku_data" / "rrn"
EXTREME_DIR = ROOT / "sudoku_data" / "sudoku_extreme"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"

for _directory in (DATA_DIR, ARTIFACTS, FIGURES):
    _directory.mkdir(parents=True, exist_ok=True)

#: The seeds every model is trained with, in order.
SEEDS = (0, 1, 2)

#: The models this example builds; see ``model.py``.
MODELS = ("goi", "rrn", "trm", "act")

#: Gradient-norm clipping, identical for every model.
GRAD_CLIP = 1.0

#: The learning rates of the per-model grid search on the validation split.
LR_GRID = (3e-4, 1e-3, 3e-3)

#: Learning rates used when the grid search has not been run.
DEFAULT_LR = {"goi": 1e-3, "rrn": 1e-3, "trm": 1e-3, "act": 1e-3}


@dataclass(frozen=True)
class Widths:
    """
    The widths of one model, the knobs that set its parameter count.

    Parameters:
        dim : The width of a clue embedding and of a unit message.
        state_dim : The width of a cell's recurrent state.
        hidden : The width of the hidden layers inside a cell.
        y_dim : The width of a recursion model's answer embedding, unused
                by the single-run models.
    """
    dim: int = 24
    state_dim: int = 96
    hidden: int = 192
    y_dim: int = 48

    def asdict(self) -> dict:
        return {"dim": self.dim, "state_dim": self.state_dim,
                "hidden": self.hidden, "y_dim": self.y_dim}


@dataclass(frozen=True)
class Budget:
    """
    One experiment budget: how much data, how long, and how deep.

    Parameters:
        name : The name of the budget, used for artifact filenames.
        n_train, n_valid, n_test : The size of each split.
        epochs : The passes over the training subsample.
        batch_size : The batch size of every model.
        rounds : The message-passing rounds of the single-run models, and
                 the rounds per cycle of the recursion models.
        cycles : The cycles per supervision step of a recursion model.
        steps : The supervision steps of a recursion model.
        lr_epochs, lr_n_train : The size of the learning-rate grid search.
        seeds : The seeds to train.
        augment : Whether to apply the sudoku symmetry group on the fly.
    """
    name: str
    n_train: int
    n_valid: int
    n_test: int
    epochs: int
    batch_size: int
    rounds: int = 16
    cycles: int = 3
    steps: int = 8
    lr_epochs: int = 1
    lr_n_train: int = 2000
    seeds: tuple[int, ...] = SEEDS
    augment: bool = False


#: A few-minute miniature of :data:`FULL`, faithful in everything but the
#: amounts, so it exercises exactly the code paths of the large run.
QUICK = Budget(
    name="quick", n_train=5000, n_valid=1500, n_test=3000,
    epochs=4, batch_size=64, rounds=8, cycles=2, steps=4)

#: The budget behind the recorded baseline results.
FULL = Budget(
    name="full", n_train=50000, n_valid=6000, n_test=18000,
    epochs=8, batch_size=128, rounds=20, cycles=3, steps=8,
    lr_n_train=12000)

#: Widths chosen so that the three models have trainable parameter counts
#: within 10% of each other; ``model.match_widths`` checks it.
WIDTHS = {
    "goi": Widths(dim=24, state_dim=96, hidden=192),
    "rrn": Widths(dim=24, state_dim=96, hidden=172),
    "trm": Widths(dim=24, state_dim=88, hidden=172, y_dim=48),
}
WIDTHS["act"] = WIDTHS["trm"]

#: The parameter count the three models are tuned to match.
PARAM_TARGET = 205_000

#: The depth each recorded baseline was trained at; see ``README.md``.
BEST_ROUNDS = {"goi": 64, "rrn": 16, "trm": 6}

#: Test-time compute swept after training, per model: rounds for the
#: single-run models, supervision steps for the recursion.
SWEEP = {"goi": (64, 96, 144), "rrn": (16, 32, 288), "trm": (8, 12, 16)}


#: The winning trial of the ``c-trm-v2`` optuna study, verbatim: 0.9933
#: validation boards on the full Palm et al. (2018) benchmark.
SIMPLE_TRM = {
    "widths": Widths(dim=24, state_dim=64, hidden=128, y_dim=32),
    "rounds": 8, "cycles": 4, "steps": 6,
    "lr": 1.514758938606213e-3,
    "weight_decay": 2.8161126429784275e-4,
    "warmup_frac": 0.0587365178457409,
    "ema_decay": 0.9941523779564319,
    "epochs": 15, "batch_size": 256,
    "n_train": 180_000, "n_valid": 18_000, "n_test": 18_000,
    "checkpoint": "simple-sudoku-trm.pt"}

#: The winning trial of the ``trm-extreme-3x`` study, verbatim: 0.4632
#: validation boards at trained depth on sudoku-extreme, 0.4801 at 32
#: supervision steps.  Measured in iterations of 512, not epochs.
EXTREME_TRM = {
    "widths": Widths(dim=72, state_dim=192, hidden=384, y_dim=96),
    "rounds": 10, "cycles": 4, "steps": 12,
    "lr": 8.97846346433278e-4,
    "weight_decay": 6.271288420139338e-5,
    "warmup_frac": 0.060740911088609954,
    "iterations": 6000, "eval_every": 200, "batch_size": 512,
    "variant": "special_large",
    "n_train": 1_001_000, "n_valid": 2000,
    "eval_compute": (8, 16, 32), "final_compute": (8, 16, 32, 64, 128),
    "checkpoint": "extreme-sudoku-trm.pt"}
