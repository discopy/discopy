# -*- coding: utf-8 -*-

"""
Model C -- TRM-inspired recursion on model A's map.  Full training run.

    python train_c_trm.py --seed 0

================================================================================
BEST RESULT REACHED BY THIS CONFIGURATION
================================================================================

Trained on 50,000 puzzles of the Palm et al. (2018) benchmark for 8 epochs
with recursion shape (n=6 rounds per cycle, T=3 cycles per supervision step,
N_sup=8 supervision steps), learning rate 1e-3, batch size 128, Adam with
gradient-norm clipping at 1.0.  Held-out test split, 18,000 puzzles, mean over
seeds 0 and 1:

    cell accuracy    0.9750
    boards solved    0.8737          (seed 0: 0.8463, seed 1: 0.9011)

Validation board-solve rate was 0.8768.  The declared protocol had picked
3e-4 for this model, which reaches 0.8354 -- so tuning bought about four
points here, against forty-six for model A.

The headline property is not accuracy but **memory**.  This model runs
6 x 3 x 8 = 144 effective rounds per example, yet only the last cycle of each
supervision step is differentiated, so activation memory is set by the
*segment length* rather than by the depth actually run: 144 rounds in about
1.2 GiB, against 13.7 GiB for model B at comparable depth.  On the benchmark's
hardest slice -- the 17-given puzzles -- that buys a large margin over B.

A deeper variant (n=8) was being trained when this study was stopped; its
first seed peaked at 0.911 validation boards before drifting to 0.885, so
there is probably a little more here.  It is not included because only one of
its two seeds finished.

================================================================================
WHAT THIS MODEL IS, IN THE LANGUAGE OF `discopy.neural`
================================================================================

All three models in this folder are the same kind of object: a **closed
combinatorial map**.  That means two things and nothing else.

1.  A finite family of *boxes*.  A box is a `Network(name, dom, cod, module)`:
    a PyTorch module together with a list of typed ports.  A port carries a
    `Dim(w)` -- a wire of width `w`.  Boxes are **shared**: the same module
    instance appears at many sites of the map.

2.  A *fixpoint-free involution* on the set of all ports -- an edge relation
    pairing each port with exactly one other, never itself.

Running the map for one **round** is completely determined:

        every box reads its in-ports, its module runs, it writes its out-ports,
        and the involution sigma permutes those emissions into the next round's
        inputs.

which is the execution formula of the geometry of interaction,
`m |-> sigma (+) f_i (m)`.  A **trace** is a port wired to another port of the
same box: private memory that survives a round.

--------------------------------------------------------------------------------
Model C's particular map: what changes relative to models A and B
--------------------------------------------------------------------------------

This is the interesting case, because **the map barely changes at all**.

Model C's wiring is *model A's* -- the same 81 shared cell boxes, the same 27
shared Deep-Sets unit boxes, the same bipartite involution -- plus one extra
trace per cell: an **answer loop** of width 48 carrying an embedding `y` of
the current guess.  The cell reads `y` but never writes it; the outer loop
does.  486 wires against A's 405, and the same 108 boxes.

What changes is not the morphism but the **evaluation strategy**.  And the
thing that licenses a different strategy is a property of the formalism
itself: running the map is *composable*,

        F^(a + b) = F^b . F^a

so a long run can be cut into segments and resumed, provided the state at the
cut is complete.  Two details make it complete here:

  * the cell **re-emits its clue** instead of zeros (`resumable=True`), so a
    run carries its own clues and does not need them re-injected -- message
    passing therefore runs with `inject=False`;
  * `experiments.maps.route` turns one round's emissions back into the next
    round's inputs, which is just sigma applied explicitly.

On top of that the training loop is the TRM recursion:

    a **cycle**  = `n` rounds of resumable message passing, then one refresh
                   of the answer `y` from the latent state `z` by a GRUCell;
    a **step**   = `T` cycles, the first `T - 1` under `torch.no_grad()`;
    an **epoch** = for each batch, `N_sup` steps, each with its own loss,
                   backward pass and optimizer step, detaching the state
                   between steps.

So where models A and B take **one** optimizer step per batch with gradients
through all their rounds, model C takes `N_sup` = 8 steps per batch, each
differentiating only the final cycle -- `n` = 6 rounds deep.  That is the
whole trade: many more forward rounds per example, far shallower gradients,
and activation memory that does not grow with the depth run.

The readout decodes `y` only, never the latent state.

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

#: The configuration this file exists to record: the best found for model C.
CONFIG = dict(rounds_per_cycle=6, cycles_per_step=3, supervision_steps=8,
              learning_rate=1e-3, epochs=8, batch_size=128,
              n_train=50_000, n_valid=6_000, n_test=18_000, grad_clip=1.0)

CROSS_ENTROPY = torch.nn.CrossEntropyLoss()


def build(n: int = None, cycles: int = None, n_sup: int = None):
    """ Model C at the widths that match all three models to ~205k weights. """
    return zoo.TRMSolver(
        rounds=n or CONFIG["rounds_per_cycle"],
        cycles=cycles or CONFIG["cycles_per_step"],
        n_sup=n_sup or CONFIG["supervision_steps"])


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
    One epoch of the segmented outer loop -- the one place model C differs.

    Each batch runs `N_sup` supervision steps.  A step is `T` cycles of `n`
    rounds; `model.step` runs the first `T - 1` under `no_grad` and
    differentiates only the last, so each backward pass is `n` rounds deep
    however many rounds have actually been run.  The state is **detached**
    between steps, which is what keeps activation memory flat.
    """
    model.train()
    total, checkpoints = 0.0, 0
    for clues, target in batches(split, CONFIG["batch_size"], rng, device):
        flat = target.reshape(-1)
        state = model.initial(clues)

        for _ in range(model.n_sup):
            state, logits = model.step(state)
            loss = CROSS_ENTROPY(logits.reshape(-1, model.n), flat)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           CONFIG["grad_clip"])
            optimizer.step()

            state = state.detach()          # the segment boundary
            total, checkpoints = total + loss.item(), checkpoints + 1
    return total / max(checkpoints, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--n", type=int, default=CONFIG["rounds_per_cycle"],
                        help="rounds per cycle")
    parser.add_argument("--cycles", type=int,
                        default=CONFIG["cycles_per_step"])
    parser.add_argument("--n-sup", type=int,
                        default=CONFIG["supervision_steps"])
    parser.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--out", default="model_c_trm.pt")
    arguments = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = datasets.load()
    train_split = splits["train"].subsample(CONFIG["n_train"])
    valid_split = splits["valid"].subsample(CONFIG["n_valid"])
    test_split = splits["test"].subsample(CONFIG["n_test"])

    # the seed is fixed *before* the model is built, so the initialisation is
    # reproducible as well as the batching
    seed_everything(arguments.seed)
    model = build(arguments.n, arguments.cycles, arguments.n_sup).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.lr)
    rng = np.random.default_rng(arguments.seed)

    effective = arguments.n * arguments.cycles * arguments.n_sup
    print(f"model C - TRM recursion on model A's map, seed {arguments.seed}")
    print(f"  {zoo.count_parameters(model):,} parameters, {model.n_wires} "
          f"wires, {len(model.grid.boxes)} boxes")
    print(f"  n={arguments.n} x T={arguments.cycles} x "
          f"N_sup={arguments.n_sup} = {effective} effective rounds, "
          f"gradients {arguments.n} rounds deep")
    print(f"  lr {arguments.lr:g}, {len(train_split):,} puzzles, "
          f"{arguments.epochs} epochs\n")

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

    # test-time compute is measured in supervision steps for this model
    for n_sup in (arguments.n_sup, 12, 16):
        extra = evaluate(model, test_split, compute=n_sup)
        print(f"  at {n_sup:2d} supervision steps "
              f"({n_sup * arguments.n * arguments.cycles:4d} rounds): "
              f"boards {extra['board']:.4f}")

    torch.save({"state_dict": model.state_dict(), "config": CONFIG,
                "seed": arguments.seed, "test": test}, arguments.out)
    print(f"\nsaved {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
