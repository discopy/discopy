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

from discopy import compact, rigid
from discopy.symmetric import Functor
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
    return Functor(
        ob_map={full_adder.dom[:1]: int},
        ar_map={full_adder: full_adder_function}, cod=Function)


# --- k-fold tensor ---------------------------------------------------------

@case("Diagram", "k-fold tensor",
      sizes(10, 20, 50, full=(100, 200, 500)))
def test_tensor_diagram(benchmark, n):
    box = generator.not_box()
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.tensor(b), box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram", "k-fold tensor, 1 layer",
      sizes(10, 20, 50, 100, full=(200, 500, 1000)))
def test_tensor_single_layer_diagram(benchmark, n):
    box = generator.not_box()
    benchmark.pedantic(
        lambda: generator.single_layer_tensor(box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph", "k-fold tensor", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_hypergraph(benchmark, n):
    hbox = generator.not_box().to_hypergraph()
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.tensor(b), hbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap", "k-fold tensor", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_cmap(benchmark, n):
    mbox = generator.not_box().to_map()
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.tensor(b), mbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


# --- staircase / foliation -------------------------------------------------

@case("Diagram", "staircase",
      sizes(10, 20, full=(50,)))  # ~O(n^3): 50 ~ 3.7s
def test_foliation_diagram(benchmark, n):
    st = generator.staircase(generator.not_box(), n)
    benchmark.pedantic(
        lambda: st.foliation(), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- k-fold series ---------------------------------------------------------

@case("Diagram", "k-fold series",
      sizes(10, 20, 50, 100, full=(200, 500, 1000)))
def test_series_diagram(benchmark, n):
    box = generator.not_box()
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.then(b), box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph", "k-fold series", sizes(10, 20, 50, full=(100, 200)))
def test_series_hypergraph(benchmark, n):
    hbox = generator.not_box().to_hypergraph()
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.then(b), hbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap", "k-fold series", sizes(10, 20, 50, full=(100, 200)))
def test_series_cmap(benchmark, n):
    mbox = generator.not_box().to_map()
    benchmark.pedantic(
        lambda: generator.repeated(lambda a, b: a.then(b), mbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


# --- ripple-carry adder ----------------------------------------------------

@case("Diagram", "adder step", sizes(2, 5, 10, 20, full=(50, 100)))
def test_adder_step_diagram(benchmark, n):
    full_adder = generator.full_adder_box()
    adder = generator.build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: generator.adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph", "adder step",
      sizes(2, 5, 10, 20, full=(50,)))  # ~O(n^2)
def test_adder_step_hypergraph(benchmark, n):
    full_adder = generator.full_adder_box().to_hypergraph()
    adder = generator.build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: generator.adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap", "adder step", sizes(2, 5, 10, 20, full=(50,)))
def test_adder_step_cmap(benchmark, n):
    full_adder = generator.full_adder_box().to_map()
    adder = generator.build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: generator.adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram", "adder functor", sizes(2, 5, 10, 20, full=(50, 100)))
def test_adder_functor_diagram(benchmark, n):
    full_adder = generator.full_adder_box()
    functor = _adder_functor(full_adder)
    adder = generator.build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: functor(adder), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- spiral (arXiv:1804.07832) ---------------------------------------------

@case("Diagram", "spiral build", sizes(10, 20, 50, full=(100, 200)))
def test_spiral_build_diagram(benchmark, n):
    benchmark.pedantic(
        lambda: generator.make_spiral(n)[0],
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph", "spiral build",
      sizes(10, 20, full=(50,)))  # ~O(n^3): 50 ~ 10s
def test_spiral_build_hypergraph(benchmark, n):
    spiral = generator.make_spiral(n)[0]
    benchmark.pedantic(
        lambda: spiral.to_hypergraph(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap", "spiral build", sizes(10, 20, full=(50,)))
def test_spiral_build_cmap(benchmark, n):
    spiral = generator.make_spiral(n)[0]
    benchmark.pedantic(
        lambda: spiral.to_map(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram", "spiral equality",
      sizes(5, 10, full=(20,)))  # ~O(n^3): 20 ~ 8.4s
def test_spiral_normal_form_diagram(benchmark, n):
    spiral = generator.make_spiral(n)[0]
    benchmark.pedantic(
        lambda: spiral.normal_form(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph", "spiral equality",
      sizes(5, 10, 20, full=(50,)))  # VF2: 100 risky
def test_spiral_equality_hypergraph(benchmark, n):
    # Two independent builds of the same closed spiral: equality must decide
    # they are isomorphic. The spiral is closed (empty boundary), hence not
    # monogamous: exercises the networkx VF2 fallback, not the fast path.
    left = generator.make_spiral(n)[0].to_hypergraph()
    right = generator.make_spiral(n)[0].to_hypergraph()
    benchmark.pedantic(
        lambda: left == right, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap", "spiral equality", sizes(5, 10, 20, full=(50,)))
def test_spiral_equality_cmap(benchmark, n):
    left = generator.make_spiral(n)[0].to_map()
    right = generator.make_spiral(n)[0].to_map()
    benchmark.pedantic(
        lambda: left == right, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- transpose snakes ------------------------------------------------------

@case("Diagram", "transpose equality",
      sizes(5, 10, 20, full=(50, 100)))
def test_transpose_snake_removal_diagram(benchmark, n):
    # rigid.normal_form genuinely yanks the snakes back to f (super-linear).
    x = rigid.Ty('x')
    g = generator.alternating_transpositions(rigid.Box('f', x, x), n)
    benchmark.pedantic(
        lambda: g.normal_form(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph", "transpose equality",
      sizes(10, 20, 50, full=(100, 200)))
def test_transpose_equality_hypergraph(benchmark, n):
    # Timed call includes to_hypergraph (snake-absorbing construction) plus
    # equality; the snaked diagram is monogamous, so the linear fast path.
    x = compact.Ty('x')
    f = compact.Box('f', x, x)
    bare = f.to_hypergraph()
    g = generator.alternating_transpositions(f, n)
    benchmark.pedantic(
        lambda: g.to_hypergraph() == bare, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap", "transpose equality", sizes(10, 20, 50, full=(100, 200)))
def test_transpose_equality_cmap(benchmark, n):
    x = compact.Ty('x')
    f = compact.Box('f', x, x)
    bare = f.to_map()
    g = generator.alternating_transpositions(f, n)
    benchmark.pedantic(
        lambda: g.to_map() == bare, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- correctness (run once, not benchmarks) --------------------------------

def test_adder_functor_correct():
    """ The functor compiles adder(k) to a carry-save accumulator: encoding
    ``2k + 1`` random bits and decoding the outputs with weights
    ``[1, 2, 2, ...]`` recovers their popcount. """
    rng = random.Random(0)
    full_adder = generator.full_adder_box()
    functor = _adder_functor(full_adder)
    adder = full_adder
    for k in range(1, 5):
        compiled = functor(adder)
        for _ in range(20):
            bits = [rng.randrange(2) for _ in range(2 * k + 1)]
            assert carry_save_value(compiled(*bits)) == sum(bits)
        adder = generator.adder_step(full_adder, adder, k)


def test_alternating_transposition_is_identity():
    """ The rigid snake-wrapped box normalises back to the bare box. """
    x = rigid.Ty('x')
    f = rigid.Box('f', x, x)
    assert generator.alternating_transpositions(f, 1).normal_form() == f


def test_transpose_equality_holds():
    """ Compact graph encodings remove snakes from the wrapped box. """
    x = compact.Ty('x')
    f = compact.Box('f', x, x)
    assert generator.alternating_transpositions(
        f, 3).to_hypergraph() == f.to_hypergraph()
    assert generator.alternating_transpositions(f, 3).to_map() == f.to_map()
