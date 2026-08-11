# -*- coding: utf-8 -*-

"""
Freeze the behaviour of the solver family, byte for byte.

This script is the *oracle* of the refactor of ``discopy.neural`` into a
category-generic engine: it is run once against the pre-refactor code, its
output is committed, and :mod:`test.neural.test_equivalence` then asserts
that the refactored code reproduces it exactly.  Nothing here may depend on
the refactor -- it only calls the public interface of the models.

Five fingerprints per model, all deterministic:

* **structure** -- the wiring, as exact integers: the ``edges``
  involution, the port widths, the routing permutation ``src``, the
  box-order layout, inverse and permutation of
  :attr:`CMap._fused_routing`, the group metas and the port families a
  solver reads.  Two levels are recorded, because they answer two
  different questions.  ``total``, ``src``, ``perm`` and ``metas`` are
  what the arithmetic is a function of: together they fix which module
  reads which bytes in which order.  ``edges``, ``port_widths``,
  ``layout`` and ``inverse`` are the finer bookkeeping of how those bytes
  are *named*, which refining one role into two relabels without moving a
  single number.
* **parameters** -- the ordered ``(name, shape)`` list of
  ``named_parameters()``, the ``state_dict`` keys, and a SHA-256 of every
  weight concatenated in that order.
* **forward** -- the logits at every supervised checkpoint on a fixed
  batch of 16 puzzles.
* **backward** -- the gradient of a fixed cross-entropy loss with respect
  to every parameter.
* **trajectory** -- the loss after each of 20 optimizer steps.

Run as::

    python docs/neural/golden/record.py [model ...]

writing ``<model>.json`` for what is small enough to read and
``<model>.npz`` for the permutations and the tensors.  A model already on
disk is skipped, so a long run resumes.  Both a float32 and a float64 pass
are recorded; the float64 model is the float32 model cast up, so the two
share an initialisation and differ only in the precision of the
arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "examples" / "sudoku"))

import dataset as datasets                                       # noqa: E402
import model as zoo                                              # noqa: E402
from config import GRAD_CLIP, WIDTHS                             # noqa: E402

CE = torch.nn.functional.cross_entropy

#: The batch every fingerprint is taken on: the first 16 test puzzles.
BATCH = 16

#: The number of CPU threads the fingerprints are taken with.  This is not
#: a performance knob: a multi-threaded reduction splits its sum across
#: threads and adds the partial sums back in a different order, so the
#: thread count is part of what "bitwise" means on the CPU.  It also
#: happens to be the fast setting -- at these widths, handing a cell to a
#: thread pool costs far more than its arithmetic.
THREADS = 1

#: The optimizer steps of the recorded trajectory.
STEPS = 20

#: The models, at deliberately small depths: the fingerprints have to
#: separate implementations, not to score well, and every code path of the
#: full budgets is already exercised at these depths.
MODELS = {
    "goi": dict(kind="goi", widths=WIDTHS["goi"], rounds=4),
    "rrn": dict(kind="rrn", widths=WIDTHS["rrn"], rounds=4),
    "trm": dict(kind="trm", widths=WIDTHS["trm"], rounds=2, cycles=2,
                steps=3),
    "act": dict(kind="act", widths=WIDTHS["trm"], rounds=2, cycles=2,
                steps=3, halt_head="softmin"),
}


def build(spec: dict):
    """ One model of :data:`MODELS`, freshly seeded. """
    torch.manual_seed(0)
    np.random.seed(0)
    arguments = {key: value for key, value in spec.items() if key != "kind"}
    return zoo.BUILDERS[spec["kind"]](**arguments)


# --- the fingerprints ------------------------------------------------------

def structure(model, arrays: dict) -> dict:
    """
    The wiring of a model, as plain integers.

    The four long permutations go into ``arrays`` -- they compress by an
    order of magnitude and are unreadable anyway -- and the small, telling
    numbers stay in the JSON.
    """
    interaction = model.interaction
    grid = interaction.cmap
    routing, fused = grid._routing, grid._fused_routing
    metas = [[grid.boxes[indices[0]].name, list(indices), n_boxes, width]
             for _, indices, n_boxes, width in fused["metas"]]
    arrays.update(
        src=routing["src"].numpy(), perm=fused["perm"].numpy(),
        layout=fused["layout"].numpy(), inverse=fused["inverse"].numpy(),
        edges=np.array(list(grid.edges)),
        port_widths=np.array(list(grid.port_widths)))
    return {
        "computation": {
            "total": routing["total"],
            "metas": metas,
        },
        "counts": {
            "n_boxes": len(grid.boxes),
            "n_ports": grid.n_ports,
            "n_wires": interaction.n_wires,
            "n_cells": model.n_cells,
            "router_total": interaction.total,
            "parameters": zoo.count_parameters(model),
        },
        "families": {
            # the key names are frozen with the goldens; the families are
            # now addressed by (generator name, role).
            "clue_ports": list(interaction.ports["cell", zoo.CLUE]),
            "state_ports": list(interaction.heads.get(
                ("cell", zoo.STATE),
                interaction.heads.get(("cell", zoo.HIDDEN), ()))),
            "answer_ports": list(interaction.ports.get(
                ("cell", zoo.ANSWER), ())),
        },
    }


def parameters(model) -> dict:
    """ The ordered parameter names, shapes and a SHA of their values. """
    named = list(model.named_parameters())
    digest = hashlib.sha256()
    for _, value in named:
        digest.update(value.detach().to(torch.float64).numpy().tobytes())
    return {
        "named": [[name, list(value.shape)] for name, value in named],
        "state_dict_keys": list(model.state_dict().keys()),
        "sha256": digest.hexdigest(),
    }


def forward(model, clues) -> list:
    """ The logits at every supervised checkpoint, without gradients. """
    model.eval()
    with torch.no_grad():
        return list(model(clues, deep=True))


def loss_of(model, clues, target):
    """
    The loss a training step of this model differentiates.

    The single-run solvers average the cross-entropy over every round of
    one backward graph; the recursion solvers are supervised once per
    detached segment, so the recorded gradient is the one of their first
    supervision step -- the same call ``train.train_epoch`` makes.
    """
    flat = target.reshape(-1)
    if model.segmented:
        state = model.initial(clues)
        if hasattr(model.map.solver, "halt"):
            _, logits, halt = model.act_step(state)
            return CE(logits.reshape(-1, model.n), flat) \
                + model.halt_loss(halt, logits, target)
        _, logits = model.step(state)
        return CE(logits.reshape(-1, model.n), flat)
    losses = [CE(logits.reshape(-1, model.n), flat)
              for logits in model(clues, deep=True)]
    return sum(losses) / len(losses)


def backward(model, clues, target) -> dict:
    """ The gradient of :func:`loss_of` with respect to every parameter. """
    model.train()
    model.zero_grad(set_to_none=True)
    loss_of(model, clues, target).backward()
    grads = {name: (torch.zeros_like(value) if value.grad is None
                    else value.grad.detach().clone())
             for name, value in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return grads


def trajectory(model, clues, target) -> list:
    """
    The loss after each of :data:`STEPS` optimizer steps on one batch.

    One step means one optimizer step, so the recursion solvers walk
    through their supervision steps one at a time, carrying and detaching
    their state exactly as ``train.train_epoch`` does.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    flat = target.reshape(-1)
    losses, state, remaining = [], None, 0
    for _ in range(STEPS):
        if model.segmented:
            if remaining == 0:
                state, remaining = model.initial(clues), model.n_sup
            state, logits = model.step(state)
            loss = CE(logits.reshape(-1, model.n), flat)
        else:
            every = model(clues, deep=True)
            per_round = [CE(logits.reshape(-1, model.n), flat)
                         for logits in every]
            loss = sum(per_round) / len(per_round)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if model.segmented:
            state, remaining = state.detach(), remaining - 1
        losses.append(loss.item())
    return losses


# --- the run ---------------------------------------------------------------

def record(name: str, spec: dict, clues, target) -> None:
    """ Write every fingerprint of one model under ``golden/``. """
    fingerprint = {"spec": {key: (value.asdict()
                                  if hasattr(value, "asdict") else value)
                            for key, value in spec.items()}}
    arrays: dict = {"clues": clues.numpy(), "target": target.numpy()}

    model = build(spec)
    fingerprint["structure"] = structure(model, arrays)
    fingerprint["parameters"] = parameters(model)

    for dtype, tag in ((torch.float32, "f32"), (torch.float64, "f64")):
        logits = forward(build(spec).to(dtype), clues)
        arrays[f"logits_{tag}"] = torch.stack(logits).numpy()
        for key, value in backward(
                build(spec).to(dtype), clues, target).items():
            arrays[f"grad_{tag}/{key}"] = value.numpy()
        fingerprint[f"trajectory_{tag}"] = trajectory(
            build(spec).to(dtype), clues, target)

    (HERE / f"{name}.json").write_text(json.dumps(fingerprint, indent=1))
    np.savez_compressed(HERE / f"{name}.npz", **arrays)
    print(f"  {name}: {len(fingerprint['parameters']['named'])} tensors, "
          f"sha {fingerprint['parameters']['sha256'][:12]}, "
          f"loss {fingerprint['trajectory_f32'][0]:.6f} -> "
          f"{fingerprint['trajectory_f32'][-1]:.6f}")


def main(argv=None) -> int:
    """
    Record every model of :data:`MODELS`, skipping the ones already on
    disk so that a long run can be resumed a model at a time.
    """
    names = list(argv or MODELS)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    split = datasets.load()["test"]
    clues = torch.as_tensor(split.puzzles[:BATCH], dtype=torch.long)
    target = torch.as_tensor(split.solutions[:BATCH], dtype=torch.long) - 1
    print(f"golden batch: {BATCH} test puzzles, "
          f"{int((clues > 0).sum(1).float().mean())} givens on average",
          flush=True)
    for name in names:
        if all((HERE / f"{name}.{suffix}").exists()
               for suffix in ("json", "npz")):
            print(f"  {name}: already recorded", flush=True)
            continue
        record(name, MODELS[name], clues, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
