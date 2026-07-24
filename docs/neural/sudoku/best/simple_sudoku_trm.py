# -*- coding: utf-8 -*-

"""
Train the best TRM configuration found by the optuna search.

    python simple_sudoku_trm.py

The configuration is trial 5 of the ``c-trm-v2`` study: the best run of the
search, 0.9933 validation board-solve rate on the full Palm et al. (2018)
benchmark (best at epoch 12/15), and cheaper per epoch than the round-1
winner (``n_sup=6`` instead of 8). Validation and the saved checkpoint use
an exponential moving average of the weights, the recipe detail this trial
selected. Hyperparameters are copied verbatim from the trial's record;
initialisation and data order are drawn fresh on every run, so board rates
vary a little from run to run. Training prints the hyperparameters, then
loss and validation accuracy after every epoch, keeps the weights of the
best epoch, and finishes with the one and only test-set evaluation -- the
search itself never touched the test split.

The model is built at the search widths, deliberately smaller than
``sudoku.config.WIDTHS["trm"]``: this *is* the model that scored
0.9933, not a scaled-down proxy of it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# so the script imports ``sudoku`` and ``core`` regardless of the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sudoku import data as datasets
from sudoku import models as zoo
from core import recipes
from sudoku.config import ARTIFACTS, Widths
from sudoku.train import evaluate

torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+/H100

#: The winning trial, verbatim (study "c-trm-v2", trial 5, valid 0.9933).
HYPERPARAMETERS = {
    "n": 8, "T": 4, "n_sup": 6,
    "lr": 1.514758938606213e-3,
    "weight_decay": 2.8161126429784275e-4,
    "warmup_frac": 0.0587365178457409,
    "ema_decay": 0.9941523779564319,
    "epochs": 15, "batch_size": 256}

#: The widths the search ran at -- the exact model that scored 0.9933.
WIDTHS = Widths(dim=24, state_dim=64, hidden=128, y_dim=32)

#: Where the best-epoch weights (the averaged ones) are saved.
CHECKPOINT = ARTIFACTS / "simple-sudoku-trm.pt"


def main(epochs: int = None, n_train: int = 180_000,
         n_valid: int = 18_000, n_test: int = 18_000) -> int:
    hp = HYPERPARAMETERS
    epochs = hp["epochs"] if epochs is None else epochs
    print("hyperparameters:")
    for key, value in hp.items():
        print(f"  {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = datasets.load()
    train_clues, train_targets = recipes.to_device(
        splits["train"].subsample(n_train), device)
    valid_split = splits["valid"].subsample(n_valid)

    model = zoo.TRMSolver(WIDTHS, rounds=hp["n"], cycles=hp["T"],
                          n_sup=hp["n_sup"]).to(device)
    print(f"parameters: {zoo.count_parameters(model)}")
    if device.type == "cuda":
        model.compile_cells(mode="reduce-overhead")

    per_epoch = len(train_clues) // hp["batch_size"]
    total = per_epoch * epochs * hp["n_sup"]
    optimizer = recipes.adamw(model, hp["lr"], hp["weight_decay"])
    scheduler = recipes.cosine_schedule(
        optimizer, int(hp["warmup_frac"] * total), total)
    ema = recipes.EMA(model, hp["ema_decay"])
    batch_stream = recipes.stream(
        train_clues, train_targets, hp["batch_size"],
        np.random.default_rng())

    best = -1.0  # below any board rate, so epoch 1 always saves a checkpoint
    for epoch in range(1, epochs + 1):
        tick = time.perf_counter()
        loss = recipes.train_chunk(model, batch_stream, optimizer, scheduler,
                                   per_epoch, ema=ema)
        with ema.averaged(model):
            scores = evaluate(model, valid_split)
            if scores["board"] > best:
                best = scores["board"]
                torch.save({
                    "state_dict": model.state_dict(), "hyperparameters": hp,
                    "widths": WIDTHS.asdict(), "epoch": epoch,
                    "valid_board": best}, CHECKPOINT)
        print(f"epoch {epoch:2d}/{epochs}  loss {loss:.4f}"
              f"  cell {scores['cell']:.4f}  board {scores['board']:.4f}"
              f"  ({time.perf_counter() - tick:.0f}s)")

    # the single test evaluation, on the best epoch's weights.
    model.load_state_dict(torch.load(
        CHECKPOINT, map_location=device, weights_only=False)["state_dict"])
    scores = evaluate(model, splits["test"].subsample(n_test))
    print(f"\nbest valid board {best:.4f}  (checkpoint {CHECKPOINT.name})")
    print(f"test  cell {scores['cell']:.4f}  board {scores['board']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
