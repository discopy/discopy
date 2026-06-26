# -*- coding: utf-8 -*-

"""
Benchmark for diagram composition, reproducing the experiments of
arXiv:2105.09257 for both :class:`discopy.symmetric.Diagram` and
:class:`discopy.symmetric.Hypergraph`.

This module is deliberately kept outside of ``test/*/*.py`` so that it is
not collected by the default ``pytest`` run (see ``testpaths`` in
``pyproject.toml``). It is meant to be run explicitly, e.g.

    uv run pytest benchmark/ -v --log-cli-level=INFO --log-cli-format="%(message)s"

Each case is swept over a range of sizes (:data:`SIZES`) to expose its
scaling trend; a case stops scaling once a single measurement, or its
untimed setup, exceeds :data:`MAX_SECONDS`, leaving the larger cells blank.
The final ``test_zz_report`` renders every measurement as a plain markdown
table, with one row per case and one column per size. Timings are emitted
via :mod:`logging` at ``INFO`` level so they appear inline in the pytest
output without needing ``-s``.

Timings use ``time.process_time()`` (CPU time, immune to scheduling noise)
with the garbage collector disabled during the timed region, and report
the median of a few repetitions for the cheap cases.
"""

import gc
import logging
import time
from statistics import median

from tabulate import tabulate

from discopy.symmetric import Ty, Box, Id, Diagram

# Sizes swept for every case. A case stops once a measurement or its setup
# crosses MAX_SECONDS, so slow cases simply leave the larger columns blank.
SIZES = [10, 20, 50, 100, 200, 500, 1000]
REPEATS = 3
MAX_SECONDS = 2.0          # per-cell cap: stop scaling a case past this
TIME_BUDGET = 300          # total wall budget for the whole sweep, in seconds

# case name -> {size: seconds}, in first-seen (i.e. report row) order.
_results: "dict[str, dict[int, float]]" = {}
_start_time = time.process_time()


def measure(fn):
    """ Median process-time of calling ``fn()``, with the GC disabled.

    Cheap calls (under a second) are repeated for stability; expensive ones
    are timed a single time so the sweep stays within its budget.
    """
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.process_time()
        fn()
        first = time.process_time() - start
        times = [first]
        if first <= 1.0:
            for _ in range(REPEATS - 1):
                start = time.process_time()
                fn()
                times.append(time.process_time() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    return median(times)


def run_scaling(name, prepare, sizes=SIZES):
    """ Sweep ``name`` over ``sizes``, recording one row of the report.

    ``prepare(n)`` performs any *untimed* setup for size ``n`` and returns a
    zero-argument callable that is then timed. Scaling stops for this case
    once either the setup or the measured time exceeds :data:`MAX_SECONDS`,
    or the global :data:`TIME_BUDGET` is exhausted.
    """
    row = _results.setdefault(name, {})
    for n in sizes:
        if time.process_time() - _start_time > TIME_BUDGET:
            break
        setup_start = time.process_time()
        thunk = prepare(n)
        setup = time.process_time() - setup_start
        duration = measure(thunk)
        row[n] = duration
        logging.info("%-32s n=%-5d %9.4f s", name, n, duration)
        if duration > MAX_SECONDS or setup > MAX_SECONDS:
            break


def repeated(op, box, k):
    """ Combine ``k`` copies of ``box`` with ``op``, by repeated doubling. """
    if k == 1:
        return box
    half = repeated(op, box, k // 2)
    result = op(half, half)
    if k % 2:
        result = op(result, box)
    return result


def test_not_gate_tensor():
    bit = Ty('bit')
    box = Box('NOT', bit, bit)
    run_scaling(
        "k-fold tensor (Diagram)",
        lambda n: lambda: repeated(lambda a, b: a.tensor(b), box, n))
    hbox = box.to_hypergraph()
    run_scaling(
        "k-fold tensor (Hypergraph)",
        lambda n: lambda: repeated(lambda a, b: a.tensor(b), hbox, n))


def test_not_gate_series():
    bit = Ty('bit')
    box = Box('NOT', bit, bit)
    run_scaling(
        "k-fold series (Diagram)",
        lambda n: lambda: repeated(lambda a, b: a.then(b), box, n))
    hbox = box.to_hypergraph()
    run_scaling(
        "k-fold series (Hypergraph)",
        lambda n: lambda: repeated(lambda a, b: a.then(b), hbox, n))


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


def test_adder():
    bit = Ty('bit')
    full_adder = Box('FA', bit @ bit @ bit, bit @ bit)

    def prepare_diagram(n):
        adder_n = make_adder(n, full_adder)
        return lambda: adder_step(adder_n, n, full_adder, bit)
    run_scaling("adder step (Diagram)", prepare_diagram)

    full_adder_hg = full_adder.to_hypergraph()

    def prepare_hypergraph(n):
        adder_n = make_adder(n, full_adder).to_hypergraph()
        return lambda: adder_step_hypergraph(adder_n, n, full_adder_hg, bit)
    run_scaling("adder step (Hypergraph)", prepare_hypergraph)


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


def test_spiral():
    run_scaling(
        "spiral build (Diagram)", lambda n: lambda: make_spiral(n)[0])

    def prepare_build_hg(n):
        spiral = make_spiral(n)[0]
        return lambda: spiral.to_hypergraph()
    run_scaling("spiral build (Hypergraph)", prepare_build_hg)

    def prepare_normal_form(n):
        spiral = make_spiral(n)[0]
        return lambda: spiral.normal_form()
    run_scaling("spiral normal_form (Diagram)", prepare_normal_form)

    def prepare_equality(n):
        # Two independent builds of the same closed spiral: the equality
        # check must decide they are isomorphic (the spiral is closed, so
        # this exercises the graph-isomorphism fallback).
        left = make_spiral(n)[0].to_hypergraph()
        right = make_spiral(n)[0].to_hypergraph()
        assert left == right
        return lambda: left == right
    run_scaling("spiral equality (Hypergraph)", prepare_equality)


def test_zz_report():
    """ Render the scaling sweep as a plain markdown table. Runs last.

    Named ``test_zz_*`` so it runs after every other test has populated
    ``_results``. Run with ``--log-cli-level=INFO`` to see the table.
    """
    sizes = sorted({n for row in _results.values() for n in row})
    table = [
        [name] + [f"{row[n]:.4f}" if n in row else "" for n in sizes]
        for name, row in _results.items()]
    logging.info("\n%s", tabulate(
        table, headers=["case", *sizes], tablefmt="github",
        colalign=["left", *["right"] * len(sizes)], disable_numparse=True))

    total = time.process_time() - _start_time
    assert total < TIME_BUDGET, \
        f"Benchmark took {total:.1f}s, over the {TIME_BUDGET}s budget."
