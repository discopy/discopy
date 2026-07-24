# -*- coding: utf-8 -*-

"""
The dataclasses a study is made of: the :class:`Widths` of a model, the
:class:`Budget` of a run and the :class:`Split` of a dataset.

They are deliberately torch-free, like everything a task needs at
configuration time: a task package defines its budgets and widths as plain
instances, and only the training harness (:mod:`core.train`) turns them
into tensors and optimizer steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Widths:
    """
    The widths of one solver, the knobs that set its parameter count.

    Parameters:
        dim : The width of an input embedding and of a message.
        state_dim : The width of a variable's recurrent state.
        hidden : The width of the hidden layers inside a cell.
        y_dim : The width of a recursion solver's answer embedding,
                unused by the other solvers.
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
        n_train : The number of training examples.
        n_valid : The number of validation examples.
        n_test : The number of test examples.
        epochs : The number of passes over the training subsample.
        batch_size : The batch size of every model.
        rounds : The message-passing rounds of the single-run solvers.
        trm_n : The rounds per macro-step (cycle) of a recursion solver.
        trm_T : The cycles per supervision step of a recursion solver.
        trm_n_sup : The supervision steps of a recursion solver.
        lr_epochs : The epochs of the learning-rate grid search.
        lr_n_train : The training examples of the learning-rate search.
        seeds : The seeds to train.
        augment : Whether to apply the task's augmentation on the fly.
    """
    name: str
    n_train: int
    n_valid: int
    n_test: int
    epochs: int
    batch_size: int
    rounds: int = 16
    trm_n: int = 6
    trm_T: int = 3
    trm_n_sup: int = 8
    lr_epochs: int = 1
    lr_n_train: int = 2000
    seeds: tuple[int, ...] = (0, 1, 2)
    augment: bool = False


@dataclass(frozen=True)
class Split:
    """
    One split of a fill-in-the-blanks benchmark.

    Parameters:
        name : ``"train"``, ``"valid"`` or ``"test"``.
        puzzles : The clues, of shape ``(n, cells)``, with ``0`` for a
                  blank.
        solutions : The solutions, of shape ``(n, cells)``, positive
                    digits.
        surrogate : Whether these puzzles were regenerated rather than
                    downloaded, i.e. whether they deviate from the source.
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

        The benchmarks are stored in random order, so taking a prefix
        keeps the distribution.
        """
        if n >= len(self):
            return self
        return Split(self.name, self.puzzles[:n], self.solutions[:n],
                     self.surrogate)
