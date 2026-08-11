# -*- coding: utf-8 -*-

"""
Composition benchmark, reproducing the experiments of `arXiv:2105.09257
<https://arxiv.org/pdf/2105.09257>`_ for
:class:`discopy.symmetric.Diagram`, :class:`discopy.symmetric.Hypergraph`
and :class:`discopy.symmetric.CMap`.

Each test is decorated with a suite, family, case and list of sizes. The
small/medium sizes always run; ``BENCH_FLAGS=bench:full`` adds the heavy tail
on ``main`` and manual dispatch. The ``benchmark`` fixture owns CPU-time,
GC-disabled measurement and ``benchmark.pedantic`` takes the median of a few
rounds. We only supply the workload.

This module lives outside ``test/*/*.py`` so the default ``pytest`` run never
collects it. Run it explicitly and render the table/plot from the JSON it
emits:

    uv run pytest benchmark/ -v --benchmark-json=benchmark-results/bench.json
    uv run python benchmark/report.py benchmark-results/bench.json
"""

from functools import partial
import random

from discopy import rigid, symmetric
from discopy.python import Function
from benchmark import generators as generator
from benchmark.config import ROUNDS, WARMUP, case, sizes


case = partial(case, suite="composition")


def full_adder_function(a, b, carry_in):
    """ A full adder as Python bit ops: ``(sum, carry_out)``. """
    return a ^ b ^ carry_in, (a & b) | (carry_in & (a ^ b))


def carry_save_value(outputs):
    """ Decode a carry-save accumulator: sum bit + weight-2 carries. """
    return outputs[0] + 2 * sum(outputs[1:])


def _adder_functor(full_adder):
    return symmetric.Functor(
        ob_map={full_adder.dom[:1]: int},
        ar_map={full_adder: full_adder_function}, cod=Function)


# --- k-fold tensor ---------------------------------------------------------

@case("Diagram", "k-fold tensor",
      sizes(10, 20, 50, full=(100, 200, 500)))
def test_tensor_diagram(benchmark, n):
    box = generator.not_box(symmetric.Box)
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.tensor(b), box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram", "k-fold tensor, 1 layer",
      sizes(10, 20, 50, 100, full=(200, 500, 1000)))
def test_single_layer_tensor_diagram(benchmark, n):
    box = generator.not_box(symmetric.Box)
    benchmark.pedantic(
        lambda: generator.single_layer_tensor(box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph", "k-fold tensor", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_hypergraph(benchmark, n):
    hbox = generator.not_box(symmetric.Box).to_hypergraph()
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.tensor(b), hbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap", "k-fold tensor", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_cmap(benchmark, n):
    mbox = generator.not_box(symmetric.Box).to_map()
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.tensor(b), mbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


# --- foliation -------------------------------------------------------------

@case("Diagram", "foliation",
      sizes(10, 20, full=(50,)))  # ~O(n^3): 50 ~ 3.7s
def test_foliation_diagram(benchmark, n):
    st = generator.staircase(generator.not_box(symmetric.Box), n)
    benchmark.pedantic(
        lambda: st.foliation(), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- k-fold series ---------------------------------------------------------

@case("Diagram", "k-fold series",
      sizes(10, 20, 50, 100, full=(200, 500, 1000)))
def test_series_diagram(benchmark, n):
    box = generator.not_box(symmetric.Box)
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.then(b), box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph", "k-fold series", sizes(10, 20, 50, full=(100, 200)))
def test_series_hypergraph(benchmark, n):
    hbox = generator.not_box(symmetric.Box).to_hypergraph()
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.then(b), hbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap", "k-fold series", sizes(10, 20, 50, full=(100, 200)))
def test_series_cmap(benchmark, n):
    mbox = generator.not_box(symmetric.Box).to_map()
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.then(b), mbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


# --- ripple-carry adder ----------------------------------------------------

@case("Diagram", "adder step", sizes(2, 5, 10, 20, full=(50, 100)))
def test_adder_step_diagram(benchmark, n):
    full_adder = generator.full_adder_box(symmetric.Box)
    adder = generator.build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: generator.adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph", "adder step",
      sizes(2, 5, 10, 20, full=(50,)))  # ~O(n^2)
def test_adder_step_hypergraph(benchmark, n):
    full_adder = generator.full_adder_box(symmetric.Box).to_hypergraph()
    adder = generator.build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: generator.adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap", "adder step", sizes(2, 5, 10, 20, full=(50,)))
def test_adder_step_cmap(benchmark, n):
    full_adder = generator.full_adder_box(symmetric.Box).to_map()
    adder = generator.build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: generator.adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram", "adder functor", sizes(2, 5, 10, 20, full=(50, 100)))
def test_adder_functor_diagram(benchmark, n):
    full_adder = generator.full_adder_box(symmetric.Box)
    functor = _adder_functor(full_adder)
    adder = generator.build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: functor(adder), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- spiral (arXiv:1804.07832) ---------------------------------------------

@case("Diagram", "spiral build", sizes(10, 20, 50, full=(100, 200)))
def test_spiral_build_diagram(benchmark, n):
    benchmark.pedantic(
        lambda: generator.spiral(symmetric.Box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram", "spiral normal form", sizes(5, 10))
def test_spiral_normal_form_diagram(benchmark, n):
    spiral = generator.spiral(symmetric.Box, n)
    benchmark.pedantic(
        lambda: spiral.normal_form(), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- transpose normal form --------------------------------------------------

@case("Diagram", "transpose normal form",
      sizes(5, 10, 20, full=(50,)))
def test_transpose_normal_form_diagram(benchmark, n):
    # rigid.normal_form genuinely yanks the snakes back to f (super-linear).
    x = rigid.Ty('x')
    g = generator.transpose_snakes(rigid.Box('f', x, x), n)
    benchmark.pedantic(
        lambda: g.normal_form(), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- correctness (run once, not benchmarks) --------------------------------

def test_adder_functor_correct():
    """ The functor compiles adder(k) to a carry-save accumulator: encoding
    ``2k + 1`` random bits and decoding the outputs with weights
    ``[1, 2, 2, ...]`` recovers their popcount. """
    rng = random.Random(0)
    full_adder = generator.full_adder_box(symmetric.Box)
    functor = _adder_functor(full_adder)
    adder = full_adder
    for k in range(1, 5):
        compiled = functor(adder)
        for _ in range(20):
            bits = [rng.randrange(2) for _ in range(2 * k + 1)]
            assert carry_save_value(compiled(*bits)) == sum(bits)
        adder = generator.adder_step(full_adder, adder, k)


def test_transpose_snakes_is_identity():
    """ The rigid snake-wrapped box normalises back to the bare box. """
    x = rigid.Ty('x')
    f = rigid.Box('f', x, x)
    assert generator.transpose_snakes(f, 1).normal_form() == f
