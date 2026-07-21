# -*- coding: utf-8 -*-

"""
Model B -- the recurrent relational network of Palm et al. (2018).  Full run.

    python train_b_rrn.py --seed 0

================================================================================
BEST RESULT REACHED BY THIS CONFIGURATION
================================================================================

Trained on 50,000 puzzles of the Palm et al. (2018) benchmark for 8 epochs at
20 message-passing rounds, learning rate 1e-3, batch size 128, Adam with
gradient-norm clipping at 1.0.  Held-out test split, 18,000 puzzles, mean over
seeds 0 and 1:

    cell accuracy    0.9456
    boards solved    0.7201          (seed 0: 0.7510, seed 1: 0.6892)

Run deeper at test time the same weights reach **0.8293 boards at 288 rounds**,
but cell accuracy saturates at 0.9537 by 48 rounds and never improves after:
this map converges to a fixed point, and the remaining errors are properties
of that fixed point rather than of not having iterated enough.

Unlike models A and C, this configuration is the one a one-epoch learning-rate
proxy had already selected -- searching did not improve it.  Its two deeper
variants were worse: 3e-3 at 20 rounds gives 0.637 validation boards, and
3e-3 at 32 rounds collapses to 0.000 by the final epoch after peaking at
0.307.  A depth probe at 1e-3/32 rounds was started and cut short, so whether
this baseline gains from deeper *training* is genuinely untested here.

NOTE ON COMPARISON.  These numbers are below models A and C in this folder,
and the honest framing matters: this is a faithful re-implementation at a
**reduced budget** (50k of 180k puzzles, 8 epochs, 20 rounds).  Palm et al.
report 96.6% with 32 steps over the full training set for far longer.  Nothing
here is comparable to that, and the gap should be read as a budget difference
first.

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
`m |-> sigma (+) f_i (m)`.  A **trace** is a port wired to another port of the
same box: private memory that survives a round.

--------------------------------------------------------------------------------
Model B's particular map: what changes relative to model A
--------------------------------------------------------------------------------

Exactly two things change, and they are both in the *wiring*, not in the
formalism:

  * **The involution.**  There are no unit boxes.  All 81 boxes are cells
    sharing one `RRNCell`, and each is wired directly to its 20 peers -- the
    other cells of its row, column and block.  A constraint is a clique of
    pairwise wires rather than one hyperedge box.  That is 972 wires against
    model A's 405, and it is the dominant cost of this model.

  * **The width on a wire.**  A peer wire carries a *full hidden state*
    (`Dim(96)`), not a small message (`Dim(24)`).  The message a cell receives
    from a peer literally is that peer's `h`.

The consequences follow from those two facts alone.  One round *is* one
cell-to-cell hop here, where model A needs two -- so B's 20 rounds are 20
hops against A's 64 rounds being 32.  And a round costs about six times what
A's does, because 972 wide wires move far more than 405 narrow ones.

The cell computes `f([h_own, h_peer])` for each peer and **sums** the results,
then updates with an `LSTMCell` reading `[pooled, clue_embedding]`.  Computing
`f` at the receiver rather than at the sender is mathematically the same edge
function -- the receiver holds both endpoint states -- and it lets one shared
box play both roles.  The state loop carries `[h, c]` concatenated, since an
LSTM keeps two states; only `h` goes out on the peer wires.

Supervision is identical to model A's: a loss on every round, averaged inside
one backward graph, one optimizer step per batch.

================================================================================
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from experiments import data as datasets
from experiments import models as zoo
from experiments.train import evaluate, seed_everything

#: The configuration this file exists to record: the best found for model B.
CONFIG = dict(rounds=20, learning_rate=1e-3, epochs=8, batch_size=128,
              n_train=50_000, n_valid=6_000, n_test=18_000, grad_clip=1.0)

CROSS_ENTROPY = torch.nn.CrossEntropyLoss()


def build(rounds: int = None) -> zoo.RRNSolver:
    """ Model B at the widths that match all three models to ~205k weights. """
    return zoo.RRNSolver(rounds=rounds or CONFIG["rounds"])


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

    Identical in form to model A's -- the mean cross-entropy over every round
    inside a single backward graph -- so that any difference between the two
    models is a difference of wiring and not of training scheme.
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
    parser.add_argument("--out", default="model_b_rrn.pt")
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

    print(f"model B - RRN peer clique, seed {arguments.seed}")
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

    # this map converges: more rounds help the board rate but not the cells
    for rounds in (arguments.rounds, 48, 288):
        extra = evaluate(model, test_split, compute=rounds)
        print(f"  at {rounds:4d} test rounds: cells {extra['cell']:.4f}  "
              f"boards {extra['board']:.4f}")

    torch.save({"state_dict": model.state_dict(), "config": CONFIG,
                "seed": arguments.seed, "test": test}, arguments.out)
    print(f"\nsaved {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
