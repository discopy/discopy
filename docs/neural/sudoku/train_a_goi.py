# -*- coding: utf-8 -*-

"""
Model A -- the geometry-of-interaction factor graph.  Full training run.

    python train_a_goi.py --seed 0

What a model is here -- a closed combinatorial map, its boxes, its wires and
its rounds -- is explained once for all three models in the README next to
this file.  Model A's particular map is bipartite, 108 boxes and 405 wires:

* 81 **cell boxes**, one per sudoku cell, all sharing one ``GoICell`` module,
  with 3 unit-message ports plus a state loop and a clue loop;
* 27 **unit boxes**, one per row, column and 3x3 block, all sharing one
  ``FactorBox`` module -- a permutation-equivariant Deep-Sets relation over
  the nine members of that unit.

A cell is wired to exactly the three units it belongs to: a *constraint* is
a single hyperedge box over nine variables, rather than the 36 pairwise
wires a clique needs to say the same thing.  That is the whole architectural
bet, and it is why this map has 405 wires where model B's has 972.  The
price is distance: a belief travels cell -> unit -> cell, so one
cell-to-cell hop costs two rounds here and one round in model B -- the
winning depth of 64 rounds is 32 hops.  Clues are re-injected on the clue
loop every round, a shared linear head reads each cell's state, and training
supervises every round inside one backward graph: one optimizer step per
batch, gradients through all 64 rounds.

BEST RESULT REACHED BY THIS CONFIGURATION.  Trained on 50,000 puzzles of the
Palm et al. (2018) benchmark for 8 epochs at 64 rounds, lr 1e-3, batch 128,
Adam, grad-norm clip 1.0.  Held-out test split, 18,000 puzzles, mean over
seeds 0 and 1:

    cell accuracy    0.9842
    boards solved    0.8872          (seed 0: 0.9094, seed 1: 0.8649)

Run at more test-time rounds than it was trained with, the same weights
reach **0.9182 boards at 144 rounds** -- the map keeps refining past its
trained depth.  This configuration was found by a search over learning rate
and depth; the same architecture at 3e-3 and 20 rounds -- what a one-epoch
learning-rate proxy had selected -- reaches only 0.4107 boards and
*degrades* past 32 test rounds.  The encoding was never the bottleneck; the
step size was.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# so the script imports ``sudoku`` and ``core`` regardless of the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sudoku import data as datasets
from sudoku.config import FULL, QUICK
from sudoku.train import evaluate, sweep_compute, train_model

#: The configuration this file exists to record: the best found for model A.
BUDGET = replace(FULL, name="best", rounds=64)

#: Test-time rounds swept after training with the same weights.
SWEEP = (64, 96, 144)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="run the few-minute miniature budget instead")
    arguments = parser.parse_args(argv)
    budget = replace(QUICK, name="quick-best") if arguments.quick else BUDGET

    splits = datasets.load()
    print(f"model A - GoI factor graph, seed {arguments.seed}, "
          f"budget {budget.name}")
    model, _, meta = train_model(
        "goi", budget, arguments.seed,
        splits["train"].subsample(budget.n_train),
        splits["valid"].subsample(budget.n_valid))
    print(f"  {meta['parameters']:,} parameters, {meta['wires']} wires, "
          f"{meta['boxes']} boxes, checkpoint under artifacts/")

    test_split = splits["test"].subsample(budget.n_test)
    test = evaluate(model, test_split)
    print(f"\ntest ({len(test_split):,} puzzles): "
          f"cell {test['cell']:.4f}  boards {test['board']:.4f}")
    for _, row in sweep_compute(model, test_split, SWEEP).iterrows():
        print(f"  at {int(row['compute']):4d} test rounds: "
              f"boards {row['board']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
