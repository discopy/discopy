# -*- coding: utf-8 -*-

"""
Benchmark for diagram composition, reproducing the experiments of
arXiv:2105.09257 for both :class:`discopy.symmetric.Diagram` and
:class:`discopy.symmetric.Hypergraph`.

This module is deliberately kept outside of ``test/*/*.py`` so that it is
not collected by the default ``pytest`` run (see ``testpaths`` in
``pyproject.toml``). It is meant to be run explicitly, e.g.

    uv run pytest benchmark/ -v --log-cli-level=INFO --log-cli-format="%(message)s"

Timings are emitted via :mod:`logging` at ``INFO`` level so they appear
inline in the pytest output without needing ``-s``.

Timings use ``time.process_time()`` (CPU time, immune to scheduling noise)
with the garbage collector disabled during the timed region, and report
the median of a few repetitions for stability.
"""

import gc
import logging
import time
from statistics import median

from discopy.symmetric import Ty, Box, Id, Diagram

REPEATS = 3
TIME_BUDGET = 300  # 5 minutes, in seconds.

K_NOT = 200      # Width of the NOT-gate tensors/series.
N_ADDER = 20     # Size of the ripple-carry adder.
N_SPIRAL = 16    # Number of nested cups/caps in the spiral.

_results = []


def timeit(label, fn, repeats=REPEATS):
    """ Median process-time of calling ``fn()``, with the GC disabled. """
    times = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.process_time()
            result = fn()
            times.append(time.process_time() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    duration = median(times)
    _results.append((label, duration))
    logging.info("%-55s  %.4f s", label, duration)
    return result, duration


def repeated(op, box, k):
    """ Combine ``k`` copies of ``box`` with ``op``, by repeated doubling. """
    if k == 1:
        return box
    half = repeated(op, box, k // 2)
    result = op(half, half)
    if k % 2:
        result = op(result, box)
    return result


def test_not_gate_diagram():
    bit = Ty('bit')
    not_gate = Box('NOT', bit, bit)

    tensor_k, _ = timeit(
        "build k-fold tensor (Diagram)",
        lambda: repeated(lambda a, b: a.tensor(b), not_gate, K_NOT))
    series_k, _ = timeit(
        "build k-fold series (Diagram)",
        lambda: repeated(lambda a, b: a.then(b), not_gate, K_NOT))

    _, _ = timeit(
        "Benchmark #3, large-boundary composition (Diagram)",
        lambda: tensor_k.then(tensor_k))
    _, _ = timeit(
        "tensor of two k-fold series (Diagram)",
        lambda: series_k.tensor(series_k))


def test_not_gate_hypergraph():
    bit = Ty('bit')
    not_gate = Box('NOT', bit, bit).to_hypergraph()

    tensor_k, _ = timeit(
        "build k-fold tensor (Hypergraph)",
        lambda: repeated(lambda a, b: a.tensor(b), not_gate, K_NOT))
    series_k, _ = timeit(
        "build k-fold series (Hypergraph)",
        lambda: repeated(lambda a, b: a.then(b), not_gate, K_NOT))

    _, _ = timeit(
        "Benchmark #3, large-boundary composition (Hypergraph)",
        lambda: tensor_k.then(tensor_k))
    _, _ = timeit(
        "tensor of two k-fold series (Hypergraph)",
        lambda: series_k.tensor(series_k))


def make_adder(n, full_adder):
    """ Ripple-carry adder with ``n`` full-adder cells, built incrementally.

    Wire convention: dom = carry_in @ (a_i @ b_i for i < n),
    cod = carry_out @ (sum_i for i < n).
    """
    bit = full_adder.dom[:1]
    adder = full_adder
    for k in range(1, n):
        reorder1 = list(range(1, k + 1)) + [0, k + 1, k + 2]
        reorder2 = [k] + list(range(k)) + [k + 1]
        adder = (adder @ Id(bit @ bit)).permute(*reorder1)\
            .then(Id(bit ** k) @ full_adder).permute(*reorder2)
    return adder


def adder_step(adder, k, full_adder, bit):
    """ One incremental ripple-carry step: adder(k) -> adder(k + 1). """
    reorder1 = list(range(1, k + 1)) + [0, k + 1, k + 2]
    reorder2 = [k] + list(range(k)) + [k + 1]
    return (adder @ Id(bit @ bit)).permute(*reorder1)\
        .then(Id(bit ** k) @ full_adder).permute(*reorder2)


def test_adder_diagram():
    bit = Ty('bit')
    full_adder = Box('FA', bit @ bit @ bit, bit @ bit)
    adder_n = make_adder(N_ADDER, full_adder)

    adder_next, _ = timeit(
        "adder(n) -> adder(n+1) (Diagram)",
        lambda: adder_step(adder_n, N_ADDER, full_adder, bit))

    assert adder_next.dom == bit ** (2 * N_ADDER + 3)
    assert adder_next.cod == bit ** (N_ADDER + 2)


def adder_step_hypergraph(adder_hg, k, full_adder_hg, bit):
    """ Hypergraph counterpart of :func:`adder_step`. """
    reorder1 = list(range(1, k + 1)) + [0, k + 1, k + 2]
    reorder2 = [k] + list(range(k)) + [k + 1]
    perm1 = Diagram.permutation(reorder1, bit ** (k + 3)).to_hypergraph()
    perm2 = Diagram.permutation(reorder2, bit ** (k + 2)).to_hypergraph()
    id_bb = Id(bit @ bit).to_hypergraph()
    id_k = Id(bit ** k).to_hypergraph()
    return adder_hg.tensor(id_bb).then(perm1)\
        .then(id_k.tensor(full_adder_hg)).then(perm2)


def test_adder_hypergraph():
    bit = Ty('bit')
    full_adder = Box('FA', bit @ bit @ bit, bit @ bit)
    adder_n = make_adder(N_ADDER, full_adder).to_hypergraph()
    full_adder_hg = full_adder.to_hypergraph()

    adder_next, _ = timeit(
        "adder(n) -> adder(n+1) (Hypergraph)",
        lambda: adder_step_hypergraph(adder_n, N_ADDER, full_adder_hg, bit))

    assert adder_next.dom == bit ** (2 * N_ADDER + 3)
    assert adder_next.cod == bit ** (N_ADDER + 2)


def make_spiral(n_cups):
    """ The diagram of arXiv:1804.07832, built with symmetric boxes. """
    x = Ty('x')
    unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
    cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
    result = unit
    for i in range(n_cups):
        result = result >> Id(x ** i) @ cap @ Id(x ** (i + 1))
    result = result >> Id(x ** n_cups) @ counit @ Id(x ** n_cups)
    for i in range(n_cups):
        result = result >>\
            Id(x ** (n_cups - i - 1)) @ cup @ Id(x ** (n_cups - i - 1))
    return result, unit, counit


def test_spiral_diagram():
    spiral, _ = timeit(
        "build spiral (Diagram)", lambda: make_spiral(N_SPIRAL)[0])
    unit, counit = spiral.boxes[0], spiral.boxes[N_SPIRAL + 1]

    spiral_nf, _ = timeit(
        "spiral normal_form (Diagram)", lambda: spiral.normal_form())

    assert spiral_nf.boxes[N_SPIRAL] == unit
    assert spiral_nf.boxes[-1] == counit


def test_spiral_hypergraph():
    spiral, _, _ = make_spiral(N_SPIRAL)

    spiral_hg, _ = timeit(
        "build spiral (Hypergraph)", lambda: spiral.to_hypergraph())
    unrolled_hg = spiral.normal_form().to_hypergraph()

    _, _ = timeit(
        "spiral isomorphism check (Hypergraph)",
        lambda: spiral_hg == unrolled_hg)


def test_zz_total_time_budget():
    """ Sanity check that the whole benchmark fits in the time budget.

    Named ``test_zz_*`` so it runs last, after every other test has
    appended its timing to ``_results``. Run with ``-s`` to see the table.
    """
    width = max(len(label) for label, _ in _results)
    lines = [f"{'label':<{width}}  seconds"]
    lines += [f"{label:<{width}}  {duration:.4f}" for label, duration in _results]
    total = sum(duration for _, duration in _results)
    lines.append(f"{'total':<{width}}  {total:.4f}")
    logging.info("\n%s", "\n".join(lines))
    assert total < TIME_BUDGET, \
        f"Benchmark took {total:.1f}s, over the {TIME_BUDGET}s budget."
