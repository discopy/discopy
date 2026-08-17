# -*- coding: utf-8 -*-

"""
The benchmark: uniform random 3-SAT, filtered to the satisfiable instances.

    python dataset.py                  # build and verify the default splits
    python dataset.py --budget quick   # the miniature

A sample is a formula and nothing else.  There is no input to encode: the
formula *is* the diagram, so a :class:`Split` carries no ``x`` and no ``y``,
only :class:`Instance` objects, and the target of the unsupervised loss is
computed from the clauses themselves.  That is the first structural
difference from ``examples/sudoku``, where a sample was a pair of arrays
read by one fixed diagram.

Three families of split, and the third is deliberately untouched here:

* the **training** distribution -- ``n`` uniform in
  :data:`~config.N_RANGE`, ``alpha`` uniform over :data:`~config.ALPHAS`;
* the **grid** -- one held-out test set per ``(n, alpha)`` in
  :data:`~config.GRID`, which is what every curve in ``README.md`` is
  reported on;
* the **large** instances, ``n`` in :data:`~config.BIG_N`, generated and
  cached but read by nothing in Part 1: they are Part 2's
  size-generalization study, and keeping them out of reach here is what
  makes that study honest.

Only satisfiable instances are kept, and satisfiability is *decided*, never
guessed: see :func:`~baselines.satisfiable`.  Near the threshold this throws
away roughly half of what is generated, which is the point -- an unsat
instance has no assignment to find and would only add noise to a loss that
asks for one.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from functools import cached_property
from multiprocessing import Pool

import numpy as np

import baselines
from config import ALPHAS, BIG_N, DATA_DIR, FULL, K, N_RANGE, QUICK


@dataclass(frozen=True)
class Instance:
    """
    One satisfiable formula: the number of variables and the clauses, as
    tuples of DIMACS literals.

    Frozen and hashable, so it can key the ``lru_cache`` that interns its
    diagram; ``alpha`` is carried along because a curve is reported per
    ``(n, alpha)`` and recomputing it from ``m / n`` would round.

    Parameters:
        n : The number of variables.
        clauses : The clauses, each a tuple of ``k`` DIMACS literals.
        alpha : The clause-to-variable ratio the instance was drawn at.

    Example
    -------
    >>> instance = Instance(3, ((1, -2, 3), (-1, 2, 3)), 0.67)
    >>> instance.m, instance.n_literals
    (2, 6)
    >>> instance.literals.tolist()
    [[0, 3, 4], [1, 2, 4]]
    """

    n: int
    clauses: tuple
    alpha: float

    @property
    def m(self) -> int:
        """ The number of clauses. """
        return len(self.clauses)

    @property
    def n_literals(self) -> int:
        """ The number of literal nodes, i.e. twice the variables. """
        return 2 * self.n

    @cached_property
    def literals(self) -> np.ndarray:
        """
        The clauses as literal-node indices, of shape ``(m, k)``: ``2 * v``
        for the positive literal of variable ``v``, ``2 * v + 1`` for its
        negation, which is the order :func:`~model.factor_graph` wires them
        in.
        """
        signed = np.asarray(self.clauses, dtype=np.int64)
        return 2 * (np.abs(signed) - 1) + (signed < 0)

    def satisfied_by(self, assignment) -> bool:
        """ Whether an assignment satisfies every clause. """
        return baselines.satisfies(self.n, self.clauses, assignment)


@dataclass(frozen=True)
class Split:
    """
    A named stack of instances.

    Parameters:
        name : What the split is, e.g. ``"train"`` or ``"n50-a4.26"``.
        instances : The formulas.

    Example
    -------
    >>> split = Split("toy", (Instance(3, ((1, 2, 3), ), 0.33), ))
    >>> len(split), split.sizes
    (1, (3,))
    """

    name: str
    instances: tuple

    def __len__(self) -> int:
        return len(self.instances)

    def __iter__(self):
        return iter(self.instances)

    @property
    def sizes(self) -> tuple[int, ...]:
        """ The number of variables of each instance. """
        return tuple(instance.n for instance in self.instances)

    def subsample(self, count: int) -> Split:
        """
        The first ``count`` instances, or all of them when there are fewer.

        The splits are generated in random order, so a prefix keeps the
        distribution.
        """
        if count >= len(self):
            return self
        return Split(self.name, self.instances[:count])

    def batches(self, size: int, rng: random.Random = None) -> list:
        """
        The instances cut into batches of ``size``, shuffled when a
        generator is given.

        A batch is run as **one diagram** -- the disjoint union of its
        members' factor graphs -- so nothing is padded and instances of
        different sizes cost exactly what they are.  There is therefore no
        reason to bucket by size, unlike a model that pads a dense tensor.

        Parameters:
            size : The instances per batch.
            rng : The generator behind the shuffle, ``None`` to keep order.
        """
        order = list(self.instances)
        if rng is not None:
            rng.shuffle(order)
        return [tuple(order[start:start + size])
                for start in range(0, len(order), size)]


# --- generation ------------------------------------------------------------

def random_formula(n: int, alpha: float, rng: random.Random, k: int = K):
    """
    A uniform random k-SAT formula: ``round(alpha * n)`` clauses, each over
    ``k`` distinct variables with independent uniform signs.

    Parameters:
        n : The number of variables.
        alpha : The clause-to-variable ratio.
        rng : The random generator.
        k : The clause width.

    Example
    -------
    >>> formula = random_formula(10, 4.0, random.Random(0))
    >>> len(formula), len(formula[0])
    (40, 3)
    """
    return tuple(
        tuple(variable if rng.random() < 0.5 else -variable
              for variable in rng.sample(range(1, n + 1), k))
        for _ in range(round(alpha * n)))


def sample(spec: tuple) -> Instance:
    """
    One satisfiable instance, rejecting unsatisfiable draws.

    Takes a single tuple so that it can be handed to a process pool, which
    is where the satisfiability filter spends its time.

    Parameters:
        spec : ``(sizes, alphas, seed, k)`` with ``sizes`` and ``alphas``
               the values to draw ``n`` and ``alpha`` uniformly from.
    """
    sizes, alphas, seed, k = spec
    rng = random.Random(seed)
    while True:
        n, alpha = rng.choice(sizes), rng.choice(alphas)
        clauses = random_formula(n, alpha, rng, k)
        if baselines.satisfiable(n, clauses, rng):
            return Instance(n, clauses, alpha)


def generate(count: int, sizes, alphas, seed: int, name: str,
             workers: int = None, k: int = K, log=print) -> Split:
    """
    ``count`` satisfiable instances drawn from ``sizes`` and ``alphas``.

    Each instance gets its own seed, so the split is reproducible however
    many worker processes build it and whatever order they finish in.

    Parameters:
        count : The number of instances.
        sizes : The values of ``n`` to draw from.
        alphas : The values of ``alpha`` to draw from.
        seed : The base seed.
        name : The name of the resulting split.
        workers : The processes to spread the filter over, ``None`` for the
                  cpu count.
        k : The clause width.
        log : Where to print progress.
    """
    sizes, alphas = tuple(sizes), tuple(alphas)
    specs = [(sizes, alphas, seed * 1_000_003 + index, k)
             for index in range(count)]
    tick = time.perf_counter()
    if workers == 1:
        instances = [sample(spec) for spec in specs]
    else:
        with Pool(workers) as pool:
            instances = pool.map(sample, specs, chunksize=16)
    log(f"  {name}: {count} instances in "
        f"{time.perf_counter() - tick:.1f}s")
    return Split(name, tuple(instances))


# --- caching ---------------------------------------------------------------

def path_of(name: str):
    """ Where a split is cached. """
    return DATA_DIR / f"{name}.npz"


def save(split: Split) -> None:
    """
    Cache a split as one ``npz``: the sizes, the ratios, the clause offsets
    and every clause of every instance stacked into one ``(total, k)``
    array.
    """
    np.savez_compressed(
        path_of(split.name),
        n=np.array(split.sizes, dtype=np.int32),
        alpha=np.array([instance.alpha for instance in split.instances],
                       dtype=np.float64),
        offsets=np.cumsum(
            [0] + [instance.m for instance in split.instances],
            dtype=np.int64),
        clauses=np.concatenate(
            [np.asarray(instance.clauses, dtype=np.int32)
             for instance in split.instances]))


def read(name: str) -> Split:
    """ A cached split, or ``None`` when it has not been built. """
    if not path_of(name).exists():
        return None
    stored = np.load(path_of(name))
    offsets, clauses = stored["offsets"], stored["clauses"]
    return Split(name, tuple(
        Instance(int(size), tuple(map(tuple, clauses[start:stop].tolist())),
                 float(alpha))
        for size, alpha, start, stop in zip(
            stored["n"], stored["alpha"], offsets[:-1], offsets[1:])))


def cached(name: str, count: int, sizes, alphas, seed: int,
           workers: int = None, log=print) -> Split:
    """
    A split, read from the cache or generated and cached.

    Parameters:
        name : The name of the split, which is also its filename.
        count, sizes, alphas, seed, workers, log : See :func:`generate`.
    """
    stored = read(name)
    if stored is not None and len(stored) >= count:
        return stored.subsample(count)
    split = generate(count, sizes, alphas, seed, name, workers, log=log)
    save(split)
    return split


def cell_seed(n: int, alpha: float, offset: int = 1) -> int:
    """
    The seed of one ``(n, alpha)`` test set, derived from the cell itself so
    that a set is the same however many budgets ask for it.

    Example
    -------
    >>> cell_seed(50, 4.26), cell_seed(50, 4.0)
    (50427, 50401)
    """
    return 1000 * n + round(100 * alpha) + offset


def load(budget=FULL, workers: int = None, log=print) -> dict:
    """
    The training split and the held-out grid, built on first use.

    Parameters:
        budget : The :class:`~config.Budget` giving the sizes.
        workers : The processes the filter is spread over.
        log : Where to print progress.

    Returns:
        ``{"train": Split, "grid": {(n, alpha): Split}}``.
    """
    train = cached(
        f"{budget.name}-train", budget.n_train,
        range(N_RANGE[0], N_RANGE[1] + 1), budget.alphas, 0, workers, log)
    grid = {
        (n, alpha): cached(
            f"n{n}-a{alpha:g}", budget.n_test, (n, ), (alpha, ),
            cell_seed(n, alpha), workers, log)
        for n, alpha in budget.grid}
    return {"train": train, "grid": grid}


def load_large(budget=FULL, workers: int = None, log=print) -> dict:
    """
    The larger instances kept for Part 2's size-generalization study.

    Nothing in Part 1 calls this: it is here so that the data exists and is
    reproducible, and so that the split a later part evaluates on was drawn
    before any of it was tuned.
    """
    return {
        (n, alpha): cached(
            f"large-n{n}-a{alpha:g}", max(budget.n_test // 4, 1),
            (n, ), (alpha, ), cell_seed(n, alpha, offset=2), workers, log)
        for n in BIG_N for alpha in ALPHAS}


def check(split: Split, log=print) -> None:
    """
    Verify a split: every instance is satisfiable, every clause has ``k``
    distinct variables, and the ratio is the one asked for.

    Satisfiability is re-decided rather than trusted, which is cheap enough
    to be worth doing once on the data every number in ``README.md`` rests
    on.
    """
    rng = random.Random(0)
    for instance in split:
        assert instance.m == round(instance.alpha * instance.n), \
            f"{split.name}: {instance.m} clauses at alpha {instance.alpha}"
        for clause in instance.clauses:
            variables = [abs(literal) for literal in clause]
            assert len(set(variables)) == len(clause) == K, \
                f"{split.name}: {clause} is not a {K}-clause"
            assert min(variables) >= 1 and max(variables) <= instance.n, \
                f"{split.name}: {clause} is out of range"
        assert baselines.satisfiable(instance.n, instance.clauses, rng), \
            f"{split.name}: an unsatisfiable instance survived the filter"
    log(f"  {split.name}: {len(split)} instances, "
        f"n in {min(split.sizes)}-{max(split.sizes)}, verified")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", default="full",
                        choices=("full", "quick"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--large", action="store_true",
                        help="also build Part 2's large-n splits")
    parser.add_argument("--check", action="store_true",
                        help="re-decide satisfiability of every instance")
    arguments = parser.parse_args(argv)
    budget = QUICK if arguments.budget == "quick" else FULL

    splits = load(budget, workers=arguments.workers)
    if arguments.large:
        splits["grid"].update(load_large(budget, workers=arguments.workers))
    if arguments.check:
        check(splits["train"])
        for split in splits["grid"].values():
            check(split)
    print(f"train: {len(splits['train'])} instances; "
          f"grid: {len(splits['grid'])} test sets of "
          f"{budget.n_test} instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
