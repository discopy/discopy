# -*- coding: utf-8 -*-

"""
Training the factor-graph SAT model -- an ordinary PyTorch training loop.

    python train.py --quick               # a few-minute miniature
    python train.py --seed 0              # the recorded baseline
    python train.py --stateful            # the NeuroSAT-faithful clause

Nothing here is in :mod:`discopy.neural`, and that is the point: a
:class:`~model.Model` is a :class:`torch.nn.Module`, so it trains with an
ordinary optimizer and an ordinary loop.  What the file contains is the
*protocol* of the study, in three layers:

* :func:`train_epoch` -- the supervision scheme.  The loss of every round
  of one differentiated run, averaged, one optimizer step per batch: the
  ``Iterate(deep=True)`` recipe of the sudoku study's model A, with
  ``inject=False`` because there is nothing to re-inject.
* :class:`Pool` -- what is different here.  In ``examples/sudoku`` one
  diagram served every puzzle and was compiled once; here **every instance
  is its own diagram**, so a batch is a fresh compilation, and a
  compilation is Python while a step is a fixed number of kernel launches.
  A pool is therefore a set of batches compiled together and reused for
  several epochs before being thrown away for fresh instances -- the knob
  that trades instance diversity against wall clock.  ``NOTES.md`` measures
  both sides of it.
* :func:`train_model` -- the registry: every run is cached under
  ``artifacts/`` as weights plus history plus metadata, so re-running a
  script re-loads instead of re-training.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, replace

import numpy as np
import torch

import dataset
import evaluate as evaluations
import model as zoo
from config import ARTIFACTS, FULL, GRAD_CLIP, QUICK, WIDTHS, Budget


def seed_everything(seed: int) -> None:
    """ Fix every source of randomness we use. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def default_device() -> torch.device:
    """ The GPU if there is one. """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- the pool of compiled batches ------------------------------------------

class Pool:
    """
    A set of batches whose diagrams are compiled once and reused.

    Building a batch is drawing a diagram (:func:`~model.factor_graph`) and
    interpreting it (:meth:`~discopy.neural.MapNN.compile`); running one is
    a fixed number of kernel launches per round, near enough independent of
    how many instances it holds.  So the arithmetic of the study is: pay a
    compilation once, amortise it over ``epochs`` passes, then replace the
    pool to see new formulas.

    The compiled interactions live in the model's own cache, keyed by the
    identity of the diagram, so a pool must not be larger than that cache
    or every epoch recompiles; :func:`~model.build` takes the size.

    Parameters:
        model : The model whose interpretation compiles the diagrams.
        instances : The formulas of this pool.
        batch_size : The instances per batch.
        device : Where the index tensors live.
        log : Where to print progress.
    """
    def __init__(self, model, instances, batch_size: int, device,
                 log=print):
        tick = time.perf_counter()
        chunks = [tuple(instances[start:start + batch_size])
                  for start in range(0, len(instances), batch_size)]
        self.batches = [zoo.Batch.of(chunk).to(device) for chunk in chunks]
        self.built = time.perf_counter() - tick
        tick = time.perf_counter()
        for batch in self.batches:
            model.map.compile(batch.diagram)
        self.compiled = time.perf_counter() - tick
        self.boxes = sum(len(batch.diagram.boxes) for batch in self.batches)
        log(f"  pool: {len(self.batches)} batches, {len(instances)} "
            f"instances, {self.boxes} boxes -- drawn in {self.built:.1f}s, "
            f"compiled in {self.compiled:.1f}s")

    def __len__(self) -> int:
        return len(self.batches)

    def shuffled(self, rng: random.Random) -> list:
        """ The batches in a fresh order. """
        order = list(self.batches)
        rng.shuffle(order)
        return order


def pool_slice(instances, index: int, size: int) -> tuple:
    """
    The ``index``-th slice of ``size`` instances, wrapping around when the
    split is smaller than the run asks for.

    Parameters:
        instances : The training instances.
        index : Which pool this is.
        size : The instances per pool.
    """
    start = (index * size) % max(len(instances), 1)
    taken = list(instances[start:start + size])
    while len(taken) < size:
        taken += list(instances[:size - len(taken)])
    return tuple(taken)


# --- one epoch -------------------------------------------------------------

def train_epoch(model, pool: Pool, optimizer, rng: random.Random,
                supervised: int = None) -> dict:
    """
    One pass over the pool, one optimizer step per batch.

    The loss is the mean over the supervised rounds of one differentiated
    run, so the number reported is per checkpoint and comparable across
    depths.  The decode rate is free: rounding the last round's assignment
    and counting unsatisfied clauses is exact arithmetic on the clauses
    themselves.

    Parameters:
        model : The model to train.
        pool : The compiled batches.
        optimizer : The optimizer.
        rng : The generator behind the batch order.
        supervised : The last rounds a loss is put on, all of them by
                     default.

    Returns:
        The mean loss per checkpoint, the training decode rate, the mean
        clause-satisfaction rate and the seconds spent.
    """
    model.train()
    tick = time.perf_counter()
    total, solved, satisfied, instances, steps = 0.0, 0, 0.0, 0, 0
    for batch in pool.shuffled(rng):
        rounds = model(batch, deep=True)
        kept = rounds if supervised is None else rounds[-supervised:]
        losses = [zoo.smooth_sat_loss(logits, batch) for logits in kept]
        loss = sum(losses) / len(losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        _, unsat, done = zoo.decode(rounds[-1].detach(), batch)
        total += float(loss.detach())
        solved += int(done.sum())
        satisfied += float(
            (1 - unsat / batch.clause_count).sum())
        instances += len(batch)
        steps += 1
    return {"loss": total / max(steps, 1),
            "decode": solved / max(instances, 1),
            "clause": satisfied / max(instances, 1),
            "steps": steps, "seconds": time.perf_counter() - tick}


# --- the registry ----------------------------------------------------------

def checkpoint_path(name: str, budget: Budget, seed: int):
    """ Where a run's artifacts live. """
    return ARTIFACTS / f"{budget.name}-{name}-seed{seed}.pt"


def train_model(budget: Budget, seed: int, splits: dict,
                name: str = "factor", stateful: bool = False,
                widths=None, device=None, resume: bool = True, log=print):
    """
    Train the model with one seed, or load it back when it is already
    cached.

    Parameters:
        budget : The budget giving data size, pool shape, depth and epochs.
        seed : The seed, fixed *before* the model is built so that the
               initialisation is reproducible too.
        splits : The output of :func:`dataset.load`.
        name : The name of the run, used for the artifact filename.
        stateful : Whether the clause box carries a recurrent state.
        widths : The widths, ``WIDTHS["factor"]`` by default.
        device : Where to train.
        resume : Whether to load a cached checkpoint if one exists.
        log : Where to print progress.

    Returns:
        The model, its per-pool history and its metadata.
    """
    device = device or default_device()
    path = checkpoint_path(name, budget, seed)
    valid_batches = -(-budget.n_valid // budget.batch_size)

    seed_everything(seed)
    model = zoo.build(
        widths or WIDTHS["factor"], rounds=budget.rounds, stateful=stateful,
        cache=budget.pool_batches + valid_batches + 4).to(device)
    if resume and path.exists():
        stored = zoo.load_checkpoint(model, path)
        log(f"  loaded {path.name}")
        return model, stored["history"], stored["meta"]

    train = splits["train"].subsample(budget.n_train - budget.n_valid)
    valid = evaluations.prepare(
        dataset.Split("valid", splits["train"].instances[-budget.n_valid:]),
        budget.batch_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=budget.lr)
    rng = random.Random(seed)
    order = list(train.instances)
    rng.shuffle(order)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history, start = [], time.perf_counter()
    drawn, compiled, stepped = 0.0, 0.0, 0.0
    for index in range(budget.pools):
        pool = Pool(model, pool_slice(order, index, budget.instances_per_pool),
                    budget.batch_size, device, log=log)
        drawn, compiled = drawn + pool.built, compiled + pool.compiled
        for epoch in range(budget.epochs):
            stats = train_epoch(model, pool, optimizer, rng,
                                budget.supervised)
            stepped += stats["seconds"]
            log(f"  pool {index + 1}/{budget.pools} epoch "
                f"{epoch + 1:2d}/{budget.epochs}  loss {stats['loss']:.4f}  "
                f"train decode {stats['decode']:.3f}  clause "
                f"{stats['clause']:.4f}  ({stats['seconds']:.0f}s)")
        scores = evaluations.evaluate(model, valid)
        history.append({"pool": index + 1, **stats, **{
            f"valid_{key}": value for key, value in scores.items()}})
        log(f"  pool {index + 1}/{budget.pools} valid decode "
            f"{scores['decode']:.4f}  clause {scores['clause']:.4f}")
        del pool

    meta = {
        "name": name, "seed": seed, "stateful": stateful,
        "budget": asdict(budget), "widths": (widths or WIDTHS["factor"]),
        "parameters": zoo.count_parameters(model),
        "instances_seen": budget.pools * budget.instances_per_pool,
        "seconds": time.perf_counter() - start,
        "seconds_drawing": drawn, "seconds_compiling": compiled,
        "seconds_stepping": stepped,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 2 ** 20
            if device.type == "cuda" else float("nan"))}
    meta["widths"] = meta["widths"].asdict()
    torch.save({"state_dict": model.state_dict(), "history": history,
                "meta": meta}, path)
    log(f"  wrote {path.name}: {meta['seconds']:.0f}s total "
        f"({drawn:.0f}s drawing, {compiled:.0f}s compiling, "
        f"{stepped:.0f}s stepping)")
    return model, history, meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="the few-minute miniature budget")
    parser.add_argument("--stateful", action="store_true",
                        help="a recurrent clause instead of a Deep-Sets one")
    parser.add_argument("--pools", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                        help="processes the satisfiability filter uses")
    arguments = parser.parse_args(argv)

    budget = QUICK if arguments.quick else FULL
    overrides = {key: value for key, value in (
        ("pools", arguments.pools), ("epochs", arguments.epochs),
        ("rounds", arguments.rounds)) if value is not None}
    budget = replace(budget, **overrides) if overrides else budget
    name = "stateful" if arguments.stateful else "factor"

    splits = dataset.load(budget, workers=arguments.workers)
    print(f"{name}, seed {arguments.seed}, budget {budget.name}: "
          f"{budget.pools} pools of {budget.instances_per_pool} instances, "
          f"{budget.rounds} rounds")
    model, _, meta = train_model(
        budget, arguments.seed, splits, name=name,
        stateful=arguments.stateful, resume=not arguments.no_resume)
    print(f"  {meta['parameters']:,} parameters, "
          f"{meta['instances_seen']:,} instances seen")
    print(evaluations.report(model, evaluations.prepare_grid(
        splits["grid"], budget.batch_size, default_device())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
