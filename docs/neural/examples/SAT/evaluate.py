# -*- coding: utf-8 -*-

"""
Scoring a trained SAT model.

    python evaluate.py artifacts/full-factor-seed0.pt
    python evaluate.py artifacts/quick-factor-seed0.pt --quick --baselines
    python evaluate.py artifacts/full-factor-seed0.pt --sweep 32 64 128 256

Four protocols, and they answer different questions:

* :func:`evaluate` -- fixed compute.  The model runs exactly the depth it
  is asked for and is scored on the assignment it ends with.  Two numbers
  are reported and neither replaces the other: the **decode rate**, the
  fraction of instances whose rounded assignment satisfies *every* clause,
  and the **clause rate**, the mean fraction of clauses satisfied.  A model
  can push the second to 0.99 while the first stays at zero, which is
  exactly the failure a per-clause metric hides.
* :func:`sweep_compute` -- the same at several depths, which is where the
  "keeps refining past its trained depth" claim is made or refuted.
* :func:`report` -- the ``(n, alpha)`` grid, i.e. how both rates degrade
  with size and with proximity to the satisfiability threshold.
* :func:`baseline_report` -- the honest comparison.  Random, greedy and
  WalkSAT on the *same* test sets, the local searches given the wall-clock
  the model spent.  No claim is made against CDCL and none should be.

And one diagnostic that is not a score: :func:`equivariance` measures how
far the two learned cells are from the permutation equivariance their
signatures declare, which should be rounding error and nothing more.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import baselines
import dataset
import model as zoo
from config import ARTIFACTS, FULL, QUICK, SWEEP, Widths
from discopy.neural import check_equivariant


# --- fixed compute ---------------------------------------------------------

def prepare(split, batch_size: int = 32, device=None) -> list:
    """
    The batches of a split, drawn and left on a device.

    Built once and reused: a batch owns a diagram, a diagram is compiled by
    identity, and rebuilding it per protocol would recompile it per
    protocol.  Sweeping four depths over a prepared split therefore costs
    one compilation, not four.

    Parameters:
        split : The split to cut up.
        batch_size : The instances per batch.
        device : Where the index tensors live.
    """
    return [zoo.Batch.of(chunk).to(device)
            for chunk in split.batches(batch_size)]


@torch.no_grad()
def evaluate(model, batches, rounds: int = None, seed: int = 0) -> dict:
    """
    The decode rate and the clause rate of a prepared split.

    The initial state is a learned vector plus noise, so a run is
    stochastic; the noise is drawn from a generator seeded per batch, which
    makes every number here reproducible and makes a sweep over depths a
    comparison of depths rather than of noise draws.

    Two wall clocks are reported and they are not the same number.
    ``seconds`` is the message passing alone, which is what a local search
    should be matched against; ``seconds_total`` also counts compiling the
    diagrams, which is a cost of this implementation rather than of the
    method and is measured in ``NOTES.md``.

    Parameters:
        model : The trained model.
        batches : The output of :func:`prepare`.
        rounds : The test-time compute, the trained depth by default.
        seed : The seed of the initial noise.

    Returns:
        ``decode`` (fraction of instances solved), ``clause`` (mean
        fraction of clauses satisfied), the instances seen and the two
        wall clocks.
    """
    model.eval()
    device = next(model.parameters()).device
    overrides = {} if rounds is None else {"rounds": rounds}
    solved, satisfied, seen, running = 0, 0.0, 0, 0.0
    start = time.perf_counter()
    for index, batch in enumerate(batches):
        model.map.compile(batch.diagram)
        generator = torch.Generator(device=device)
        generator.manual_seed(1_000_003 * seed + index)
        tick = time.perf_counter()
        logits = model(batch, generator=generator, **overrides)
        _, unsat, done = zoo.decode(logits, batch)
        running += time.perf_counter() - tick
        solved += int(done.sum())
        satisfied += float((1 - unsat / batch.clause_count).sum())
        seen += len(batch)
    return {"decode": solved / max(seen, 1),
            "clause": satisfied / max(seen, 1), "n": seen,
            "seconds": running,
            "seconds_total": time.perf_counter() - start}


def sweep_compute(model, batches, values=SWEEP, seed: int = 0) -> list:
    """
    Accuracy as a function of test-time compute, on one prepared split.

    Parameters:
        model : The trained model.
        batches : The output of :func:`prepare`.
        values : The message-passing rounds to run.
        seed : The seed of the initial noise, shared by every depth.
    """
    return [dict(rounds=value, **evaluate(model, batches, value, seed))
            for value in values]


def prepare_grid(grid: dict, batch_size: int = 32, device=None) -> dict:
    """ :func:`prepare` on every test set of a grid. """
    return {key: prepare(split, batch_size, device)
            for key, split in grid.items()}


def report(model, prepared: dict, rounds: int = None, sweep=None,
           seed: int = 0) -> str:
    """
    The ``(n, alpha)`` table every curve of ``README.md`` is read off, as
    aligned text.

    Parameters:
        model : The trained model.
        prepared : The output of :func:`prepare_grid`.
        rounds : The test-time compute of the table, trained depth by
                 default.
        sweep : The depths to append a per-depth decode rate for, or
                ``None``.
        seed : The seed of the initial noise.
    """
    sizes = sorted({n for n, _ in prepared})
    ratios = sorted({alpha for _, alpha in prepared})
    lines = ["decode rate (clause rate) at "
             f"{rounds or model.rounds} rounds",
             "     n" + "".join(f"{alpha:>16g}" for alpha in ratios)]
    for n in sizes:
        cells = []
        for alpha in ratios:
            batches = prepared.get((n, alpha))
            if batches is None:
                cells.append(f"{'--':>16}")
                continue
            scores = evaluate(model, batches, rounds, seed)
            cells.append(
                f"{scores['decode']:>9.3f} ({scores['clause']:.3f})")
        lines.append(f"{n:>6d}" + "".join(cells))
    if sweep:
        lines.append("")
        lines.append("decode rate vs test-time rounds")
        lines.append("     n  alpha" + "".join(
            f"{value:>8d}" for value in sweep))
        for (n, alpha), batches in sorted(prepared.items()):
            row = sweep_compute(model, batches, sweep, seed)
            lines.append(f"{n:>6d} {alpha:>6g}" + "".join(
                f"{value['decode']:>8.3f}" for value in row))
    return "\n".join(lines)


# --- the classical baselines ----------------------------------------------

def baseline_report(grid: dict, seconds: float, seed: int = 0,
                    log=print) -> dict:
    """
    Random, greedy and WalkSAT on the same test sets, at a matched
    wall-clock budget per instance.

    ``seconds`` is meant to be the per-instance wall clock the model spent,
    which :func:`evaluate` reports; a local search given more time than the
    model is a different experiment and should be labelled as one.

    Parameters:
        grid : The held-out test sets, keyed by ``(n, alpha)``.
        seconds : The per-instance budget of the local searches.
        seed : The seed of their random generator.
        log : Where to print progress.
    """
    rng = random.Random(seed)
    results: dict = {}
    for key, split in sorted(grid.items()):
        row = {"random": 0.0, "greedy": 0.0, "walksat": 0.0}
        for instance in split:
            assignment = baselines.random_assignment(instance.n, rng)
            row["random"] += float(instance.satisfied_by(assignment))
            for name in ("greedy", "walksat"):
                row[name] += float(baselines.solve_within(
                    name, instance.n, instance.clauses, seconds, rng))
        results[key] = {name: value / max(len(split), 1)
                        for name, value in row.items()}
        log(f"  n={key[0]:3d} alpha={key[1]:<5g} "
            + "  ".join(f"{name} {value:.3f}"
                        for name, value in results[key].items()))
    return results


def comparison(model_rows: dict, baseline_rows: dict) -> str:
    """
    The model and the baselines side by side, as aligned text.

    Parameters:
        model_rows : The model's decode rate per ``(n, alpha)``.
        baseline_rows : The output of :func:`baseline_report`.
    """
    lines = ["     n  alpha     model    random    greedy   walksat"]
    for key in sorted(baseline_rows):
        row = baseline_rows[key]
        lines.append(
            f"{key[0]:>6d} {key[1]:>6g}"
            f"{model_rows.get(key, float('nan')):>10.3f}"
            f"{row['random']:>10.3f}{row['greedy']:>10.3f}"
            f"{row['walksat']:>10.3f}")
    return "\n".join(lines)


# --- the diagnostic --------------------------------------------------------

def equivariance(model, degree: int = 5, atol: float = 1e-5) -> dict:
    """
    The equivariance residual of each learned cell, measured in double
    precision on a copy so that nothing about the model is disturbed.

    A :class:`~discopy.neural.Site` and a
    :class:`~discopy.neural.Relation` are permutation-equivariant because
    they pool symmetrically, and the law is **lax**: it holds up to the
    reordering of a floating-point reduction.  So this returns the residual
    rather than a boolean, and
    :func:`~discopy.neural.check_equivariant` raises if it is not rounding
    error.  The copy is moved to the cpu in double precision, which is
    where the check builds its inputs and what makes "rounding error" a
    number worth reporting.

    Parameters:
        model : The trained model.
        degree : The arity the cells are tested at; both serve any degree,
                 so the number is a choice and is reported with the result.
        atol : The residual above which a cell is rejected.
    """
    widths = zoo.widths_of(model.map.ob)
    cells = {"lit": (zoo.literal(degree), widths),
             "flip": (zoo.members(2), {zoo.MSG: widths[zoo.MSG]})}
    cells["clause"] = (zoo.clause(degree), widths) \
        if widths[zoo.CSTATE] else (zoo.members(degree),
                                    {zoo.MSG: widths[zoo.MSG]})
    result = {}
    for name, (signature, sizes) in cells.items():
        residuals = check_equivariant(
            copy.deepcopy(model.map.ar[name]).double().cpu(), signature,
            sizes, atol=atol)
        result[name] = {str(role): value for role, value in residuals.items()}
    return result


# --- loading a trained model ----------------------------------------------

def load(path, device=None):
    """
    Rebuild a model from a checkpoint, taking its shape from the record.

    Parameters:
        path : The checkpoint written by :func:`~train.train_model`.
        device : Where to put it.
    """
    stored = torch.load(Path(path), map_location="cpu", weights_only=False)
    meta = stored["meta"]
    model = zoo.build(
        Widths(**meta["widths"]), rounds=meta["budget"]["rounds"],
        stateful=meta["stateful"],
        cache=meta["budget"]["pool_batches"] + 8)
    model.load_state_dict(stored["state_dict"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-eval", type=int, default=None,
                        help="instances per test set")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="instances per compiled batch")
    parser.add_argument("--sweep", type=int, nargs="*", default=None,
                        help="test-time rounds to sweep")
    parser.add_argument("--baselines", action="store_true",
                        help="also run random / greedy / WalkSAT")
    parser.add_argument("--seconds", type=float, default=None,
                        help="per-instance budget of the local searches, "
                             "the model's own by default")
    parser.add_argument("--stem", default=None)
    arguments = parser.parse_args(argv)

    budget = QUICK if arguments.quick else FULL
    if arguments.batch_size:
        budget = replace(budget, batch_size=arguments.batch_size)
    model, meta = load(arguments.checkpoint)
    grid = dataset.load(budget)["grid"]
    if arguments.n_eval:
        grid = {key: split.subsample(arguments.n_eval)
                for key, split in grid.items()}
    print(f"{meta['name']} seed {meta['seed']}, "
          f"{meta['parameters']:,} parameters, trained at "
          f"{meta['budget']['rounds']} rounds on "
          f"{meta['instances_seen']:,} instances")

    sweep = arguments.sweep if arguments.sweep is not None else list(SWEEP)
    prepared = prepare_grid(grid, budget.batch_size,
                            next(model.parameters()).device)
    print()
    print(report(model, prepared, sweep=sweep))

    print("\nequivariance residuals (lax: rounding error is the answer)")
    for name, residuals in equivariance(model).items():
        print(f"  {name}: " + ", ".join(
            f"{role} {value:.2e}" for role, value in residuals.items()))

    rows = {key: evaluate(model, batches)
            for key, batches in prepared.items()}
    payload = {"model": {f"{n}-{alpha:g}": value
                         for (n, alpha), value in rows.items()},
               "sweep": {f"{n}-{alpha:g}": sweep_compute(
                   model, batches, sweep)
                   for (n, alpha), batches in prepared.items()},
               "meta": {key: value for key, value in meta.items()
                        if key != "budget"}}
    if arguments.baselines:
        seconds = arguments.seconds or float(np.mean(
            [row["seconds"] / max(row["n"], 1) for row in rows.values()]))
        print(f"\nbaselines at {seconds * 1000:.1f} ms per instance, "
              "the model's own")
        classical = baseline_report(grid, seconds)
        print()
        print(comparison({key: row["decode"] for key, row in rows.items()},
                         classical))
        payload["baselines"] = {f"{n}-{alpha:g}": value
                                for (n, alpha), value in classical.items()}
        payload["baseline_seconds"] = seconds

    stem = arguments.stem or f"scores-{arguments.checkpoint.stem}"
    (ARTIFACTS / f"{stem}.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {ARTIFACTS / f'{stem}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
