# -*- coding: utf-8 -*-

"""
The study protocol on top of the :mod:`core.train` harness: checkpointing,
the registry-bound training entry point and the learning-rate grid.

Every run is cached under :data:`sudoku.config.ARTIFACTS` as a checkpoint
holding the weights, the per-epoch history and the metadata of the run, so
re-running a training script re-loads rather than re-trains.  The
harness itself -- deep supervision, evaluation, batching -- is generic and
lives in :mod:`core.train`; it is re-exported here for convenience.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch

from core.train import (  # noqa: F401 -- the harness, re-exported
    batches, decode, device_of, evaluate, logits_of, seed_everything,
    sweep_compute, train_epoch)
from sudoku import data as datasets
from sudoku import models as zoo
from sudoku.config import ARTIFACTS, DEFAULT_LR, LR_GRID, WIDTHS, Budget


def checkpoint_path(name: str, budget: Budget, seed: int):
    """ Where a run's artifacts live. """
    return ARTIFACTS / f"{budget.name}-{name}-seed{seed}.pt"


def train_model(name: str, budget: Budget, seed: int, train_split,
                valid_split, lr: float = None, device=None,
                resume: bool = True, log=print):
    """
    Train one model with one seed, or load it back when it is already
    cached.

    Parameters:
        name : ``"goi"``, ``"rrn"`` or ``"trm"``.
        budget : The budget giving data size, epochs, depth and batch size.
        seed : The seed, fixed before the model is built so that the
               initialisation is reproducible too.
        train_split : The training puzzles, already subsampled.
        valid_split : The split evaluated after every epoch.
        lr : The learning rate, from :data:`DEFAULT_LR` by default.
        resume : Whether to load a cached checkpoint if one exists.
        log : Where to print progress.
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    lr = DEFAULT_LR[name] if lr is None else lr
    path = checkpoint_path(name, budget, seed)

    seed_everything(seed)
    model = zoo.build(name, budget).to(device)
    if resume and path.exists():
        stored = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(stored["state_dict"])
        return model, pd.DataFrame(stored["history"]), stored["meta"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    augment = datasets.augment if budget.augment else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, start = [], time.perf_counter()
    for epoch in range(budget.epochs):
        tick = time.perf_counter()
        stats = train_epoch(model, train_split, budget, optimizer, rng,
                            device, augment=augment)
        scores = evaluate(model, valid_split)
        history.append({
            "epoch": epoch + 1, "seconds": time.perf_counter() - tick,
            **stats, **scores})
        log(f"  {name} seed {seed} epoch {epoch + 1:2d}/{budget.epochs}  "
            f"loss {stats['loss']:.4f}  cell {scores['cell']:.4f}  "
            f"board {scores['board']:.4f}  ({history[-1]['seconds']:.0f}s)")
    meta = {
        "model": name, "seed": seed, "lr": lr, "budget": asdict(budget),
        "widths": WIDTHS[name].asdict(),
        "parameters": zoo.count_parameters(model),
        "wires": model.n_wires, "boxes": len(model.grid.boxes),
        "seconds": time.perf_counter() - start,
        "seconds_per_epoch": float(np.mean([h["seconds"] for h in history])),
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2 ** 20
            if device.type == "cuda" else float("nan")),
        "opt_steps_per_epoch": history[-1]["opt_steps"],
        "checkpoints_per_epoch": history[-1]["checkpoints"]}
    torch.save({"state_dict": model.state_dict(), "history": history,
                "meta": meta}, path)
    return model, pd.DataFrame(history), meta


def lr_search(name: str, budget: Budget, train_split, valid_split,
              grid=LR_GRID, device=None, log=print) -> pd.DataFrame:
    """
    A small learning-rate grid on the validation split, cached to disk.

    One seed, a reduced number of epochs and a reduced training subsample:
    enough to rule out a badly scaled learning rate without spending the
    experiment budget on tuning.
    """
    path = ARTIFACTS / f"{budget.name}-lr-{name}.json"
    if path.exists():
        return pd.DataFrame(json.loads(path.read_text()))
    # the epochs go in the name: a cached search run must never be reused
    # after `lr_epochs` changes, or a 1-epoch result would be read back as
    # if it had been trained for the new number of epochs.
    small = replace(budget, epochs=budget.lr_epochs,
                    name=f"{budget.name}-lrsearch{budget.lr_epochs}e-{name}")
    subsample = train_split.subsample(budget.lr_n_train)
    rows = []
    for lr in grid:
        _, history, _ = train_model(
            name, replace(small, name=f"{small.name}-{lr:g}"), 0, subsample,
            valid_split, lr=lr, device=device, log=lambda *a: None)
        rows.append({"model": name, "lr": lr,
                     "cell": history["cell"].iloc[-1],
                     "board": history["board"].iloc[-1],
                     "loss": history["loss"].iloc[-1]})
        log(f"  {name} lr {lr:g}: cell {rows[-1]['cell']:.4f}  "
            f"board {rows[-1]['board']:.4f}")
    path.write_text(json.dumps(rows, indent=2))
    return pd.DataFrame(rows)
