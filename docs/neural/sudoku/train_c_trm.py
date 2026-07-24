# -*- coding: utf-8 -*-

"""
Model C -- TRM-inspired recursion on model A's map.  Full training run.

    python train_c_trm.py --seed 0

What a model is here -- a closed combinatorial map, its boxes, its wires and
its rounds -- is explained once for all three models in the README next to
this file.  Model C is the interesting case, because **the map barely
changes at all**: its wiring is model A's -- the same 81 shared cell boxes,
the same 27 shared Deep-Sets unit boxes, the same bipartite involution --
plus one extra trace per cell, an *answer loop* of width 48 carrying an
embedding ``y`` of the current guess, which the cell reads but never writes.
486 wires against A's 405, the same 108 boxes.

What changes is the **evaluation strategy**, licensed by a property of the
formalism itself: running the map is composable, ``F^(a+b) = F^b . F^a``, so
a long run can be cut into segments and resumed, provided the state at the
cut is complete.  Two details make it complete here: the cell re-emits its
clue instead of zeros (``resumable=True``), so a run carries its own clues
and message passing runs with ``inject=False``; and reading the flat
messages back (``return_flat=True``) captures the whole state of the run.
On top of that sits the recursion of Jolicoeur-Martineau (2025):

    a cycle  = ``n`` rounds of message passing, then one refresh of the
               answer ``y`` from the latent state by a ``GRUCell``;
    a step   = ``T`` cycles, the first ``T - 1`` under ``no_grad``;
    an epoch = per batch, ``N_sup`` steps, each with its own loss, backward
               pass and optimizer step, detaching the state in between.

Where models A and B take one optimizer step per batch with gradients
through all their rounds, model C takes ``N_sup = 8`` steps per batch, each
differentiating only the final ``n = 6``-round cycle.  Many more forward
rounds per example, far shallower gradients, and activation memory set by
the segment length rather than the depth run.  The readout decodes ``y``
only, never the latent state.

BEST RESULT REACHED BY THIS CONFIGURATION.  Trained on 50,000 puzzles of the
Palm et al. (2018) benchmark for 8 epochs with recursion shape (n=6, T=3,
N_sup=8), lr 1e-3, batch 128, Adam, grad-norm clip 1.0.  Held-out test
split, 18,000 puzzles, mean over seeds 0 and 1:

    cell accuracy    0.9750
    boards solved    0.8737          (seed 0: 0.8463, seed 1: 0.9011)

The declared protocol had picked 3e-4 for this model, which reaches 0.8354
-- tuning bought about four points here, against forty-six for model A.
The headline property is not accuracy but **memory**: 6 x 3 x 8 = 144
effective rounds per example in about 1.2 GiB of activations, against
13.7 GiB for model B at comparable depth, because only the last cycle of
each supervision step is differentiated.
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

#: The configuration this file exists to record: the best found for model C.
BUDGET = replace(FULL, name="best")

#: Test-time supervision steps swept after training with the same weights.
SWEEP = (8, 12, 16)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="run the few-minute miniature budget instead")
    arguments = parser.parse_args(argv)
    budget = replace(QUICK, name="quick-best") if arguments.quick else BUDGET

    splits = datasets.load()
    effective = budget.trm_n * budget.trm_T * budget.trm_n_sup
    print(f"model C - TRM recursion on model A's map, seed {arguments.seed}, "
          f"budget {budget.name}")
    print(f"  n={budget.trm_n} x T={budget.trm_T} x N_sup={budget.trm_n_sup}"
          f" = {effective} effective rounds, gradients {budget.trm_n} deep")
    model, _, meta = train_model(
        "trm", budget, arguments.seed,
        splits["train"].subsample(budget.n_train),
        splits["valid"].subsample(budget.n_valid))
    print(f"  {meta['parameters']:,} parameters, {meta['wires']} wires, "
          f"{meta['boxes']} boxes, checkpoint under artifacts/")

    test_split = splits["test"].subsample(budget.n_test)
    test = evaluate(model, test_split)
    print(f"\ntest ({len(test_split):,} puzzles): "
          f"cell {test['cell']:.4f}  boards {test['board']:.4f}")
    # test-time compute is measured in supervision steps for this model
    for _, row in sweep_compute(model, test_split, SWEEP).iterrows():
        n_sup = int(row["compute"])
        print(f"  at {n_sup:2d} supervision steps "
              f"({n_sup * budget.trm_n * budget.trm_T:4d} rounds): "
              f"boards {row['board']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
