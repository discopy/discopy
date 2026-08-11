# -*- coding: utf-8 -*-

"""
The sudoku registry: the study protocol of :mod:`core.registry`, bound to
this task.

Everything procedural moved to core -- checkpointing, the re-load-rather-
than-retrain entry point, the learning-rate grid live in
:mod:`core.registry`, and the epoch harness in :mod:`core.train`.  What
is declared here is the :class:`~core.registry.TaskSpec`: the model zoo,
the widths, the default learning rates, the artifacts directory and the
symmetry augmentation.  The entry points keep their historical
signatures, so the ``train_*.py`` scripts and the recorded recipes call
exactly what they always called.
"""

from __future__ import annotations

import pandas as pd

from core.train import (  # noqa: F401 -- the harness, re-exported
    batches, decode, device_of, evaluate, logits_of, seed_everything,
    sweep_compute, train_epoch)
from core import registry
from sudoku import data as datasets
from sudoku import models as zoo
from sudoku.config import ARTIFACTS, DEFAULT_LR, LR_GRID, WIDTHS, Budget

#: This task, as the registry sees it.
TASK = registry.TaskSpec(
    build=zoo.build, widths=WIDTHS, default_lr=DEFAULT_LR,
    artifacts=ARTIFACTS, augment=datasets.augment)


def checkpoint_path(name: str, budget: Budget, seed: int):
    """ Where a run's artifacts live. """
    return registry.checkpoint_path(TASK, name, budget, seed)


def train_model(name: str, budget: Budget, seed: int, train_split,
                valid_split, lr: float = None, device=None,
                resume: bool = True, log=print):
    """
    Train one model with one seed, or load it back when it is already
    cached; see :func:`core.registry.train_model`.

    Parameters:
        name : ``"goi"``, ``"rrn"`` or ``"trm"``.
        budget : The budget giving data size, epochs, depth and batch size.
        seed : The seed, fixed before the model is built.
        train_split : The training puzzles, already subsampled.
        valid_split : The split evaluated after every epoch.
        lr : The learning rate, from :data:`sudoku.config.DEFAULT_LR` by
             default.
        resume : Whether to load a cached checkpoint if one exists.
        log : Where to print progress.
    """
    return registry.train_model(
        TASK, name, budget, seed, train_split, valid_split,
        lr=lr, device=device, resume=resume, log=log)


def lr_search(name: str, budget: Budget, train_split, valid_split,
              grid=LR_GRID, device=None, log=print) -> pd.DataFrame:
    """
    A small learning-rate grid on the validation split, cached to disk;
    see :func:`core.registry.lr_search`.
    """
    return registry.lr_search(
        TASK, name, budget, train_split, valid_split,
        grid=grid, device=device, log=log)
