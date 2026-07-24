# -*- coding: utf-8 -*-

"""
Train the best TRM configuration found on the sudoku-extreme benchmark.

    python extreme_sudoku_trm.py

The configuration is trial 5 of the ``trm-extreme-3x`` study: the best run
of the search on ``sudoku_extreme_special_large`` (1,001,000 training
examples from 1,000 base puzzles), valid board 0.4632 at its trained depth
and 0.4801 when evaluated at 32 supervision steps -- test-time compute is
part of the recipe, so the periodic evaluation scores 8/16/32 steps and
the final protocol picks the best depth before the one test evaluation.
Hyperparameters are copied verbatim from the trial's record;
initialisation and data order are drawn fresh on every run.

Training is measured in iterations (batches of 512), not epochs: 6,000
iterations is about three epochs of the training set. Every 200 iterations
the model is scored on 2,000 held-out puzzles as a single GPU batch, and
the best weights so far are checkpointed immediately -- a crash never
loses more than one check interval.

The model is ~1.0M parameters (three times the widths the Palm-benchmark
recipes use): the extreme puzzles need the capacity, see
``docs/optuna/optuna_trm_extreme.py`` for the search this comes from.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# so the script imports ``sudoku`` and ``core`` regardless of the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sudoku import models as zoo
from core import recipes
from sudoku import sudoku_extreme
from sudoku.config import ARTIFACTS, Widths
from sudoku.train import evaluate

torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+/H100

#: The winning trial, verbatim ("trm-extreme-3x", trial 5, valid 0.4632).
HYPERPARAMETERS = {
    "n": 10, "T": 4, "n_sup": 12,
    "lr": 8.97846346433278e-4,
    "weight_decay": 6.271288420139338e-5,
    "warmup_frac": 0.060740911088609954,
    "iterations": 6000, "eval_every": 200, "batch_size": 512}

#: The widths the search ran at -- the exact model that scored 0.4632.
WIDTHS = Widths(dim=72, state_dim=192, hidden=384, y_dim=96)

#: Supervision steps swept at every periodic evaluation; the checkpoint
#: criterion is the best of the three, matching the search protocol.
EVAL_COMPUTE = (8, 16, 32)

#: Depths swept on the full validation split at the end, to pick the
#: inference depth for the single test evaluation.
FINAL_COMPUTE = (8, 16, 32, 64, 128)

#: Where the best weights are saved, updated at every improved check.
CHECKPOINT = ARTIFACTS / "extreme-sudoku-trm.pt"


def main(iterations: int = None, n_train: int = 1_001_000,
         n_valid: int = 2000, n_valid_full: int = None,
         n_test: int = None, compile: bool = True) -> int:
    hp = HYPERPARAMETERS
    iterations = hp["iterations"] if iterations is None else iterations
    print("hyperparameters:")
    for key, value in hp.items():
        print(f"  {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = sudoku_extreme.load("special_large")
    train_clues, train_targets = recipes.to_device(
        splits["train"].subsample(n_train), device)
    valid_small = splits["valid"].subsample(n_valid)

    model = zoo.TRMSolver(WIDTHS, rounds=hp["n"], cycles=hp["T"],
                          n_sup=hp["n_sup"]).to(device)
    print(f"parameters: {zoo.count_parameters(model)}")
    if compile and device.type == "cuda":
        model.compile_cells(mode="reduce-overhead")

    total = iterations * hp["n_sup"]
    optimizer = recipes.adamw(model, hp["lr"], hp["weight_decay"])
    scheduler = recipes.cosine_schedule(
        optimizer, int(hp["warmup_frac"] * total), total)
    batch_stream = recipes.stream(
        train_clues, train_targets, hp["batch_size"],
        np.random.default_rng())

    best = -1.0  # below any board rate, so the first check always saves
    for check in range(1, iterations // hp["eval_every"] + 1):
        tick = time.perf_counter()
        loss = recipes.train_chunk(model, batch_stream, optimizer, scheduler,
                                   hp["eval_every"])
        scores = {c: evaluate(model, valid_small, compute=c,
                              batch_size=max(n_valid, 1))
                  for c in EVAL_COMPUTE}
        top = max(EVAL_COMPUTE, key=lambda c: scores[c]["board"])
        if scores[top]["board"] > best:
            best = scores[top]["board"]
            torch.save({
                "state_dict": model.state_dict(), "hyperparameters": hp,
                "widths": WIDTHS.asdict(), "valid_board": best,
                "iteration": check * hp["eval_every"], "compute": top,
            }, CHECKPOINT)
        print(f"iteration {check * hp['eval_every']:5d}/{iterations}"
              f"  loss {loss:.4f}  cell {scores[top]['cell']:.4f}  board "
              + "/".join(f"{scores[c]['board']:.4f}" for c in EVAL_COMPUTE)
              + f" @n_sup {'/'.join(map(str, EVAL_COMPUTE))}"
              f"  ({time.perf_counter() - tick:.0f}s)", flush=True)

    # pick the inference depth on the full validation split, then run the
    # one and only test evaluation at that depth.
    model.load_state_dict(torch.load(
        CHECKPOINT, map_location=device, weights_only=False)["state_dict"])
    valid_full = splits["valid"] if n_valid_full is None \
        else splits["valid"].subsample(n_valid_full)
    boards = {}
    for compute in FINAL_COMPUTE:
        s = evaluate(model, valid_full, compute=compute, batch_size=2000)
        boards[compute] = s["board"]
        print(f"valid-full @n_sup {compute:3d}: cell {s['cell']:.4f} "
              f"board {s['board']:.4f}", flush=True)
    depth = max(boards, key=boards.get)
    test_split = splits["test"] if n_test is None \
        else splits["test"].subsample(n_test)
    s = evaluate(model, test_split, compute=depth, batch_size=2000)
    print(f"\nbest valid board {best:.4f} (checkpoint {CHECKPOINT.name})")
    print(f"test @n_sup {depth}: cell {s['cell']:.4f} board {s['board']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
