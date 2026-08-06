# -*- coding: utf-8 -*-

"""
Composition benchmark, reproducing the experiments of `arXiv:2105.09257
<https://arxiv.org/pdf/2105.09257>`_ for
:class:`discopy.symmetric.Diagram`, :class:`discopy.symmetric.Hypergraph`
and :class:`discopy.symmetric.CMap`.

Each case is a declarative `pytest-benchmark` test: one ``(case, n)`` per data
point, swept by ``@pytest.mark.parametrize`` over a size list that
:data:`BENCH_FLAGS` gates -- the small/medium sizes always run, the heavy tail
only under ``BENCH_FLAGS=bench:full`` (set on ``main`` / manual dispatch). The
``benchmark`` fixture owns timing: ``@pytest.mark.benchmark(timer=process_time,
disable_gc=True)`` gives CPU-time, GC-disabled measurement and
``benchmark.pedantic`` the median of a few rounds. We only supply the workload.

This module lives outside ``test/*/*.py`` so the default ``pytest`` run never
collects it. Run it explicitly and render the table/plot from the JSON it
emits:

    uv run pytest benchmark/ -v --benchmark-json=benchmark-results/bench.json
    uv run python benchmark/report.py benchmark-results/bench.json
"""

import random

import pytest

from discopy import compact, rigid
from discopy.symmetric import Functor
from discopy.python import Function
from benchmark.config import ROUNDS, WARMUP, case, sizes
from benchmark.generators import (
    adder_step, build_adder, full_adder_box, make_spiral, not_box, repeated,
    single_layer_tensor, staircase, with_snakes,
)


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

@case("k-fold tensor (Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200, 500)))
def test_tensor_diagram(benchmark, n):
    box = not_box()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.tensor(b), box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("k-fold tensor, 1 layer (Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, 100, full=(200, 500, 1000)))
def test_tensor_single_layer_diagram(benchmark, n):
    box = not_box()
    benchmark.pedantic(
        lambda: single_layer_tensor(box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("k-fold tensor (Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_hypergraph(benchmark, n):
    hbox = not_box().to_hypergraph()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.tensor(b), hbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("k-fold tensor (CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_cmap(benchmark, n):
    mbox = not_box().to_map()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.tensor(b), mbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


# --- staircase / foliation -------------------------------------------------

@case("staircase foliation (Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, full=(50,)))  # ~O(n^3): 50 ~ 3.7s
def test_foliation_diagram(benchmark, n):
    st = staircase(not_box(), n)
    benchmark.pedantic(
        lambda: st.foliation(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("staircase to hypergraph (Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, full=(50,)))  # ~O(n^3): 50 ~ 4.2s
def test_staircase_to_hypergraph(benchmark, n):
    st = staircase(not_box(), n)
    benchmark.pedantic(
        lambda: st.to_hypergraph(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("staircase to cmap (CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, full=(50,)))
def test_staircase_to_cmap(benchmark, n):
    st = staircase(not_box(), n)
    benchmark.pedantic(
        lambda: st.to_map(), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- k-fold series ---------------------------------------------------------

@case("k-fold series (Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, 100, full=(200, 500, 1000)))
def test_series_diagram(benchmark, n):
    box = not_box()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.then(b), box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("k-fold series (Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_series_hypergraph(benchmark, n):
    hbox = not_box().to_hypergraph()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.then(b), hbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("k-fold series (CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_series_cmap(benchmark, n):
    mbox = not_box().to_map()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.then(b), mbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


# --- ripple-carry adder ----------------------------------------------------

@case("adder step (Diagram)")
@pytest.mark.parametrize("n", sizes(2, 5, 10, 20, full=(50, 100)))
def test_adder_step_diagram(benchmark, n):
    full_adder = full_adder_box()
    adder = build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("adder step (Hypergraph)")
@pytest.mark.parametrize("n", sizes(2, 5, 10, 20, full=(50,)))  # ~O(n^2)
def test_adder_step_hypergraph(benchmark, n):
    full_adder = full_adder_box().to_hypergraph()
    adder = build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("adder step (CMap)")
@pytest.mark.parametrize("n", sizes(2, 5, 10, 20, full=(50,)))
def test_adder_step_cmap(benchmark, n):
    full_adder = full_adder_box().to_map()
    adder = build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("adder functor (Diagram)")
@pytest.mark.parametrize("n", sizes(2, 5, 10, 20, full=(50, 100)))
def test_adder_functor_diagram(benchmark, n):
    full_adder = full_adder_box()
    functor = _adder_functor(full_adder)
    adder = build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: functor(adder), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- spiral (arXiv:1804.07832) ---------------------------------------------

@case("spiral build (Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_spiral_build_diagram(benchmark, n):
    benchmark.pedantic(
        lambda: make_spiral(n)[0], rounds=ROUNDS, warmup_rounds=WARMUP)


@case("spiral build (Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, full=(50,)))  # ~O(n^3): 50 ~ 10s
def test_spiral_build_hypergraph(benchmark, n):
    spiral = make_spiral(n)[0]
    benchmark.pedantic(
        lambda: spiral.to_hypergraph(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("spiral build (CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, full=(50,)))
def test_spiral_build_cmap(benchmark, n):
    spiral = make_spiral(n)[0]
    benchmark.pedantic(
        lambda: spiral.to_map(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("spiral normal_form (Diagram)")
@pytest.mark.parametrize("n", sizes(5, 10, full=(20,)))  # ~O(n^3): 20 ~ 8.4s
def test_spiral_normal_form_diagram(benchmark, n):
    spiral = make_spiral(n)[0]
    benchmark.pedantic(
        lambda: spiral.normal_form(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("spiral equality (Hypergraph)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))  # VF2: 100 risky
def test_spiral_equality_hypergraph(benchmark, n):
    # Two independent builds of the same closed spiral: equality must decide
    # they are isomorphic. The spiral is closed (empty boundary), hence not
    # monogamous: exercises the networkx VF2 fallback, not the fast path.
    left = make_spiral(n)[0].to_hypergraph()
    right = make_spiral(n)[0].to_hypergraph()
    benchmark.pedantic(
        lambda: left == right, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("spiral equality (CMap)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_spiral_equality_cmap(benchmark, n):
    left = make_spiral(n)[0].to_map()
    right = make_spiral(n)[0].to_map()
    benchmark.pedantic(
        lambda: left == right, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- transpose snakes ------------------------------------------------------

@case("transpose snake removal (Diagram)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50, 100)))
def test_transpose_snake_removal_diagram(benchmark, n):
    # rigid.normal_form genuinely yanks the snakes back to f (super-linear).
    x = rigid.Ty('x')
    g = with_snakes(rigid.Box('f', x, x), n)
    benchmark.pedantic(
        lambda: g.normal_form(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("transpose equality (Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_transpose_equality_hypergraph(benchmark, n):
    # Timed call includes to_hypergraph (snake-absorbing construction) plus
    # equality; the snaked diagram is monogamous, so the linear fast path.
    x = compact.Ty('x')
    f = compact.Box('f', x, x)
    bare = f.to_hypergraph()
    g = with_snakes(f, n)
    benchmark.pedantic(
        lambda: g.to_hypergraph() == bare, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("transpose equality (CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_transpose_equality_cmap(benchmark, n):
    x = compact.Ty('x')
    f = compact.Box('f', x, x)
    bare = f.to_map()
    g = with_snakes(f, n)
    benchmark.pedantic(
        lambda: g.to_map() == bare, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- correctness (run once, not benchmarks) --------------------------------

def test_adder_functor_correct():
    """ The functor compiles adder(k) to a carry-save accumulator: encoding
    ``2k + 1`` random bits and decoding the outputs with weights
    ``[1, 2, 2, ...]`` recovers their popcount. """
    rng = random.Random(0)
    full_adder = full_adder_box()
    functor = _adder_functor(full_adder)
    adder = full_adder
    for k in range(1, 5):
        compiled = functor(adder)
        for _ in range(20):
            bits = [rng.randrange(2) for _ in range(2 * k + 1)]
            assert carry_save_value(compiled(*bits)) == sum(bits)
        adder = adder_step(full_adder, adder, k)


def test_transpose_snake_is_identity():
    """ The rigid snake-wrapped box normalises back to the bare box. """
    x = rigid.Ty('x')
    f = rigid.Box('f', x, x)
    assert with_snakes(f, 1).normal_form() == f


def test_transpose_equality_holds():
    """ Compact graph encodings remove snakes from the wrapped box. """
    x = compact.Ty('x')
    f = compact.Box('f', x, x)
    assert with_snakes(f, 3).to_hypergraph() == f.to_hypergraph()
    assert with_snakes(f, 3).to_map() == f.to_map()
