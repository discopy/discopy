# -*- coding: utf-8 -*-

"""
Model A -- the geometry-of-interaction factor graph.  Full training run.

    python train_a_goi.py --seed 0

================================================================================
BEST RESULT REACHED BY THIS CONFIGURATION
================================================================================

Trained on 50,000 puzzles of the Palm et al. (2018) benchmark for 8 epochs at
64 message-passing rounds, learning rate 1e-3, batch size 128, Adam with
gradient-norm clipping at 1.0.  Held-out test split, 18,000 puzzles, mean over
seeds 0 and 1:

    cell accuracy    0.9842
    boards solved    0.8872          (seed 0: 0.9094, seed 1: 0.8649)

Run at more test-time rounds than it was trained with, the same weights reach
**0.9182 boards at 144 rounds** -- the map keeps refining past its trained
depth.  Validation board-solve rate was 0.8906 at the final epoch and 0.9113
at its best epoch; the checkpoint saved here is the final one, so 0.8872 is a
lower bound on what this run touched.

This configuration was found by a search over learning rate and depth; the
same architecture at 3e-3 and 20 rounds -- what a one-epoch learning-rate
proxy had selected -- reaches only 0.4107 boards and *degrades* past 32 test
rounds.  The encoding was never the bottleneck; the step size was.

================================================================================
WHAT THIS MODEL IS, IN THE LANGUAGE OF `discopy.neural`
================================================================================

All three models in this folder are the same kind of object: a **closed
combinatorial map**.  That means two things and nothing else.

1.  A finite family of *boxes*.  A box is a `Network(name, dom, cod, module)`:
    a PyTorch module together with a list of typed ports.  A port carries a
    `Dim(w)` -- a wire of width `w`.  Boxes are **shared**: the same module
    instance appears at many sites of the map, which is what makes the model
    size independent of the grid size.

2.  A *fixpoint-free involution* on the set of all ports -- an edge relation
    pairing each port with exactly one other, never itself.  `CMap.from_wiring`
    takes the boxes and that pairing, given as `(box_index, port_position)`
    endpoints.

Running the map for one **round** is then completely determined:

        every box reads its in-ports, its module runs, it writes its out-ports,
        and the involution sigma permutes those emissions into the next round's
        inputs.

which is exactly the execution formula of the geometry of interaction,
`m |-> sigma (+) f_i (m)`.  Repeated rounds compose, `F^(a+b) = F^b . F^a`,
which is the law model C exploits to run its loop in resumable segments.

A **trace** is a port wired to another port of the same box: private memory
that survives a round.  Every cell here has two, a *state loop* and a *clue
loop*.

--------------------------------------------------------------------------------
Model A's particular map
--------------------------------------------------------------------------------

Bipartite, 108 boxes and 405 wires:

  * 81 **cell boxes**, one per sudoku cell, all sharing one `GoICell` module.
    Ports: 3 unit messages + a state loop (2 ports) + a clue loop (2 ports).
  * 27 **unit boxes**, one per row, column and 3x3 block, all sharing one
    `FactorBox` module -- a permutation-equivariant Deep-Sets relation over
    the nine members of that unit.

A cell is wired to exactly the three units it belongs to.  A *constraint* is
therefore a single hyperedge box over nine variables, rather than the 36
pairwise wires a clique would need to say the same thing.  That is the whole
architectural bet, and it is why this map has 405 wires where model B's has
972.

The price is distance: a belief travels cell -> unit -> cell, so **one
cell-to-cell hop costs two rounds** here and one round in model B.  Any
comparison of "rounds" between the two is off by that factor of two, which is
why the winning depth here (64) is not extravagant -- it is 32 hops.

Clues enter as the initial message on every clue loop and are re-injected each
round (`inject=True`).  A shared linear head reads each cell's state into nine
digit logits.  Training supervises **every round**: one forward pass returns
the logits of all 64 rounds, the cross-entropies are averaged, and that single
loss is backpropagated through the whole unrolled map -- one optimizer step
per batch, gradients flowing through all 64 rounds.

================================================================================
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from experiments import data as datasets
from experiments import models as zoo
from experiments.train import decode, evaluate, seed_everything

#: The configuration this file exists to record: the best found for model A.
CONFIG = dict(rounds=64, learning_rate=1e-3, epochs=8, batch_size=128,
              n_train=50_000, n_valid=6_000, n_test=18_000, grad_clip=1.0)

CROSS_ENTROPY = torch.nn.CrossEntropyLoss()


def build(rounds: int = None) -> zoo.GoISolver:
    """ Model A at the widths that match all three models to ~205k weights. """
    return zoo.GoISolver(rounds=rounds or CONFIG["rounds"])


def batches(split, batch_size: int, rng, device):
    """ One shuffled epoch of ``(clues, target)``; targets are 0-indexed. """
    order = rng.permutation(len(split))
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        yield (torch.as_tensor(split.puzzles[index], dtype=torch.long,
                               device=device),
               torch.as_tensor(split.solutions[index], dtype=torch.long,
                               device=device) - 1)


def train_one_epoch(model, split, optimizer, rng, device) -> float:
    """
    One epoch of deep supervision over the unrolled map.

    The loss is the mean cross-entropy over *every* round, inside a single
    backward graph: one optimizer step per batch, with gradients reaching back
    through all `rounds` rounds.  This is the expensive-gradient scheme, and
    it is what model C changes.
    """
    model.train()
    total, checkpoints = 0.0, 0
    for clues, target in batches(split, CONFIG["batch_size"], rng, device):
        every_round = model(clues, deep=True)
        losses = [CROSS_ENTROPY(logits.reshape(-1, model.n), target.reshape(-1))
                  for logits in every_round]
        loss = sum(losses) / len(losses)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()

        total += sum(item.item() for item in losses)
        checkpoints += len(losses)
    return total / max(checkpoints, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--rounds", type=int, default=CONFIG["rounds"])
    parser.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--out", default="model_a_goi.pt")
    arguments = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = datasets.load()
    train_split = splits["train"].subsample(CONFIG["n_train"])
    valid_split = splits["valid"].subsample(CONFIG["n_valid"])
    test_split = splits["test"].subsample(CONFIG["n_test"])

    # the seed is fixed *before* the model is built, so the initialisation is
    # reproducible as well as the batching
    seed_everything(arguments.seed)
    model = build(arguments.rounds).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.lr)
    rng = np.random.default_rng(arguments.seed)

    print(f"model A - GoI factor graph, seed {arguments.seed}")
    print(f"  {zoo.count_parameters(model):,} parameters, {model.n_wires} "
          f"wires, {len(model.grid.boxes)} boxes")
    print(f"  {arguments.rounds} rounds, lr {arguments.lr:g}, "
          f"{len(train_split):,} puzzles, {arguments.epochs} epochs\n")

    for epoch in range(1, arguments.epochs + 1):
        tick = time.perf_counter()
        loss = train_one_epoch(model, train_split, optimizer, rng, device)
        scores = evaluate(model, valid_split)
        print(f"  epoch {epoch:2d}/{arguments.epochs}  loss {loss:.4f}  "
              f"valid cell {scores['cell']:.4f}  board {scores['board']:.4f}  "
              f"({time.perf_counter() - tick:.0f}s)")

    test = evaluate(model, test_split)
    print(f"\ntest ({len(test_split):,} puzzles): "
          f"cell {test['cell']:.4f}  boards {test['board']:.4f}")

    # the same weights, given more rounds than they were trained with
    for rounds in (arguments.rounds, 96, 144):
        extra = evaluate(model, test_split, compute=rounds)
        print(f"  at {rounds:4d} test rounds: boards {extra['board']:.4f}")

    torch.save({"state_dict": model.state_dict(), "config": CONFIG,
                "seed": arguments.seed, "test": test}, arguments.out)
    print(f"\nsaved {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
