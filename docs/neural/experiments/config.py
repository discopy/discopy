# -*- coding: utf-8 -*-

"""
Paths, seeds and the two experiment budgets.

``QUICK`` is the budget the committed notebook runs end to end in a few
minutes on one GPU; ``FULL`` is the larger budget whose artifacts are cached
to disk and read back by the notebook when they exist. Everything else about
the two is identical, so that the QUICK run is a faithful miniature.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

#: This package lives at ``docs/neural/experiments``; everything it reads and
#: writes -- the benchmark, the checkpoints, the figures -- is a sibling of
#: it, so the whole study is one relocatable directory.
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

#: Display names and the colours used consistently by every figure.
LABELS = {
    "goi": "A · GoI factor graph",
    "rrn": "B · RRN clique",
    "trm": "C · TRM-inspired",
}
PALETTE = {"goi": "#0b7285", "rrn": "#d9480f", "trm": "#5f3dc4"}


@dataclass(frozen=True)
class Budget:
    """
    One experiment budget: how much data, how long, and how deep.

    Parameters:
        name : The name of the budget, used for artifact filenames.
        n_train : The number of training puzzles subsampled from the 180k.
        n_valid : The number of validation puzzles.
        n_test : The number of test puzzles.
        epochs : The number of passes over the training subsample.
        batch_size : The batch size of every model.
        rounds : The message-passing rounds of models A and B.
        trm_n : The rounds per macro-step (cycle) of model C.
        trm_T : The cycles per supervision step of model C.
        trm_n_sup : The supervision steps of model C.
        lr_epochs : The epochs of the learning-rate grid search.
        lr_n_train : The training puzzles of the learning-rate grid search.
        sweep_rounds : The test-time round counts swept for A and B.
        sweep_sup : The test-time supervision steps swept for C.
        seeds : The seeds to train.
    """
    name: str
    n_train: int
    n_valid: int
    n_test: int
    epochs: int
    batch_size: int
    rounds: int
    trm_n: int
    trm_T: int
    trm_n_sup: int
    lr_epochs: int
    lr_n_train: int
    sweep_rounds: tuple[int, ...]
    sweep_sup: tuple[int, ...]
    seeds: tuple[int, ...] = SEEDS
    augment: bool = False
    light: bool = False

    @property
    def effective_rounds(self) -> int:
        """ Message-passing rounds a training example receives in total. """
        return self.trm_n * self.trm_T * self.trm_n_sup

    def with_augmentation(self) -> Budget:
        """ The same budget with symmetry augmentation of the train split. """
        return replace(self, name=self.name + "-aug", augment=True)


QUICK = Budget(
    name="quick",
    n_train=5000, n_valid=1500, n_test=3000,
    epochs=4, batch_size=64, rounds=8,
    trm_n=4, trm_T=2, trm_n_sup=4,
    lr_epochs=1, lr_n_train=2000,
    sweep_rounds=(2, 4, 6, 8, 12, 16, 24, 32),
    sweep_sup=(1, 2, 3, 4, 6, 8))

FULL = Budget(
    name="full",
    n_train=50000, n_valid=6000, n_test=18000,
    epochs=8, batch_size=128, rounds=20,
    trm_n=6, trm_T=3, trm_n_sup=8,
    # one epoch: cheap, and deliberately declared up front rather than tuned
    # after the fact. It is a *proxy* -- a step size that descends fastest in
    # one epoch need not be the one that trains best in eight -- so we also
    # ran the same grid for three epochs and report the disagreement as a
    # tuning-sensitivity check in the notebook rather than quietly adopting
    # whichever grid flattered the results.
    lr_epochs=1, lr_n_train=12000,
    # chosen so that A/B rounds and C's N_sup * T * n meet on one
    # axis: a supervision step of C is worth 18 rounds, and 144, 216
    # and 288 appear in both grids, which is what makes "the same
    # test-time compute" a comparison rather than a figure of speech.
    sweep_rounds=(
        2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 144, 216, 288),
    sweep_sup=(1, 2, 3, 4, 6, 8, 12, 16))

BUDGETS = {budget.name: budget for budget in (QUICK, FULL)}


@dataclass(frozen=True)
class Widths:
    """
    The widths of one model, tuned so that the three parameter counts match.

    Parameters:
        dim : The width of a clue embedding and, for A and C, of a message.
        state_dim : The width of a cell's recurrent state.
        hidden : The width of the hidden layers inside a cell.
        y_dim : The width of model C's answer embedding, unused elsewhere.
    """
    dim: int = 24
    state_dim: int = 96
    hidden: int = 192
    y_dim: int = 48

    def asdict(self) -> dict:
        return {"dim": self.dim, "state_dim": self.state_dim,
                "hidden": self.hidden, "y_dim": self.y_dim}


#: Widths chosen by :func:`experiments.models.match_widths` so that the three
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


def effective_rounds(name: str, budget: Budget, compute: int = None) -> int:
    """
    The number of message-passing rounds one puzzle actually receives.

    This is the common axis on which the test-time compute of the three
    models can be compared: for A and B a unit of compute is one round, for
    C it is one supervision step, which is ``T`` cycles of ``n`` rounds.

    Parameters:
        name : The model.
        budget : The budget it was trained under.
        compute : Rounds for A and B, supervision steps for C; the trained
                  value by default.
    """
    if name == "trm":
        compute = budget.trm_n_sup if compute is None else compute
        return compute * budget.trm_T * budget.trm_n
    return budget.rounds if compute is None else compute


def backprop_depth(name: str, budget: Budget) -> int:
    """
    The rounds one gradient is backpropagated through: the whole run for A
    and B, one cycle for C, whose earlier cycles are detached.
    """
    return budget.trm_n if name == "trm" else budget.rounds
