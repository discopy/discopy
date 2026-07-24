# -*- coding: utf-8 -*-

"""
Model B -- the recurrent relational network of Palm et al. (2018).  Full run.

    python train_b_rrn.py --seed 0

What a model is here -- a closed combinatorial map, its boxes, its wires and
its rounds -- is explained once for all three models in the README next to
this file.  Relative to model A exactly two things change, both in the
wiring rather than the formalism:

* **The involution.**  There are no unit boxes: all 81 boxes are cells
  sharing one ``RRNCell``, each wired directly to its 20 peers -- the other
  cells of its row, column and block.  A constraint is a clique of pairwise
  wires rather than one hyperedge box: 972 wires against model A's 405, and
  that is the dominant cost of this model.
* **The width on a wire.**  A peer wire carries a full hidden state
  (``Dim(96)``), not a small message (``Dim(24)``): the message a cell
  receives from a peer literally is that peer's ``h``.

One round therefore *is* one cell-to-cell hop, where model A needs two --
but a round moves about six times the data.  The cell computes
``f([h_own, h_peer])`` per peer and sums the results, then updates with an
``LSTMCell`` reading ``[pooled, clue]``; the state loop carries ``[h, c]``
concatenated.  Supervision is identical to model A's: a loss on every round,
averaged inside one backward graph, one optimizer step per batch.

BEST RESULT REACHED BY THIS CONFIGURATION.  Trained on 50,000 puzzles of the
Palm et al. (2018) benchmark for 8 epochs at 20 rounds, lr 1e-3, batch 128,
Adam, grad-norm clip 1.0.  Held-out test split, 18,000 puzzles, mean over
seeds 0 and 1:

    cell accuracy    0.9456
    boards solved    0.7201          (seed 0: 0.7510, seed 1: 0.6892)

Run deeper at test time the same weights reach **0.8293 boards at 288
rounds**, but cell accuracy saturates at 0.9537 by 48 rounds: this map
converges to a fixed point, and the remaining errors are properties of that
fixed point rather than of not having iterated enough.  Unlike models A and
C, this configuration is the one the one-epoch learning-rate proxy had
already selected -- searching did not improve it.

NOTE ON COMPARISON.  These numbers are below models A and C in this folder,
and the honest framing matters: this is a faithful re-implementation at a
**reduced budget** (50k of 180k puzzles, 8 epochs, 20 rounds).  Palm et al.
report 96.6% with 32 steps over the full training set for far longer.  The
gap should be read as a budget difference first.
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

#: The configuration this file exists to record: the best found for model B.
BUDGET = replace(FULL, name="best")

#: Test-time rounds swept after training with the same weights.
SWEEP = (20, 48, 288)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="run the few-minute miniature budget instead")
    arguments = parser.parse_args(argv)
    budget = replace(QUICK, name="quick-best") if arguments.quick else BUDGET

    splits = datasets.load()
    print(f"model B - RRN peer clique, seed {arguments.seed}, "
          f"budget {budget.name}")
    model, _, meta = train_model(
        "rrn", budget, arguments.seed,
        splits["train"].subsample(budget.n_train),
        splits["valid"].subsample(budget.n_valid))
    print(f"  {meta['parameters']:,} parameters, {meta['wires']} wires, "
          f"{meta['boxes']} boxes, checkpoint under artifacts/")

    test_split = splits["test"].subsample(budget.n_test)
    test = evaluate(model, test_split)
    print(f"\ntest ({len(test_split):,} puzzles): "
          f"cell {test['cell']:.4f}  boards {test['board']:.4f}")
    # this map converges: more rounds help the board rate but not the cells
    for _, row in sweep_compute(model, test_split, SWEEP).iterrows():
        print(f"  at {int(row['compute']):4d} test rounds: "
              f"cells {row['cell']:.4f}  boards {row['board']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
