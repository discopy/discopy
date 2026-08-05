# -*- coding: utf-8 -*-

"""
The Diagram and Hypergraph cases reproduce the composition experiments of
`arXiv:2105.09257 <https://arxiv.org/pdf/2105.09257>`_. CMap and tensor
contraction cases extend the original benchmark.

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

import os
import random
import time

import numpy as np
import pytest

from discopy import compact, rigid, tensor
from discopy.symmetric import Ty, Box, Diagram, Functor
from discopy.monoidal import Layer
from discopy.python import Function

# --- size gating -----------------------------------------------------------
# A 1-2-5-ish series so points are evenly spaced on the log axis the scaling
# plot uses. Quadratic cases get shorter lists than the linear ones.
_FULL = "bench:full" in os.environ.get("BENCH_FLAGS", "").lower()


def sizes(*base, full=()):
    """Sizes for a case: always ``base``, plus the heavy ``full`` tail under
    ``BENCH_FLAGS=bench:full``."""
    return list(base) + (list(full) if _FULL else [])


def case(group):
    """Shared benchmark marker: CPU-time clock (immune to scheduling noise),
    garbage collector disabled in the timed region, grouped by case name."""
    return pytest.mark.benchmark(
        group=group, timer=time.process_time, disable_gc=True)


# Median of ROUNDS timed calls after WARMUP untimed ones. Inputs are built
# once *outside* the timed thunk (discopy values are immutable), so only the
# operation under test is timed -- no fresh build per round.
ROUNDS, WARMUP = 3, 1


# --- workload builders -----------------------------------------------------

def repeated(op, box, k):
    """ Combine ``k`` copies of ``box`` with ``op``, by repeated doubling. """
    if k == 1:
        return box
    half = repeated(op, box, k // 2)
    result = op(half, half)
    if k % 2:
        result = op(result, box)
    return result


def single_layer_tensor(box, k):
    """ The ``k``-fold tensor of ``box`` as a *single* :class:`Layer`.

    Tensoring two diagrams pads every layer of each with the other's wires,
    so a ``k``-fold tensor by repeated ``@`` rebuilds the layer list over and
    over. The same morphism is one layer with the ``k`` boxes side by side,
    sidestepping that overhead -- the residual cost is only the type
    concatenation in ``Layer``.
    """
    empty = box.dom[:0]
    layer = Layer(empty, box, empty, *([box, empty] * (k - 1)))
    return Diagram((layer,), layer.dom, layer.cod)


def staircase(box, k):
    """ The ``k``-fold tensor of ``box`` as a staircase of ``k`` layers -- the
    same morphism as :func:`single_layer_tensor`, spread over ``k`` layers. """
    g = box
    for _ in range(k - 1):
        g = g @ box
    return g


def permutation(factory, xs, dom):
    """ A permutation arrow built from swaps, generic over the category.

    Mirrors :meth:`symmetric.Diagram.permutation` using only ``id``, ``swap``,
    ``tensor`` and ``then``, so the same code builds a Diagram (``factory`` a
    ``symmetric.Box``/``Diagram``), a Hypergraph or a CMap directly.
    """
    if len(dom) <= 1:
        return factory.id(dom)
    i = xs[0]
    head = factory.swap(dom[:i], dom[i:i + 1]).tensor(factory.id(dom[i + 1:]))
    tail = factory.id(dom[i:i + 1]).tensor(permutation(
        factory, [x - 1 if x > i else x for x in xs[1:]],
        dom[:i] + dom[i + 1:]))
    return head.then(tail)


def adder_step(full_adder, adder, k):
    """ One incremental ripple-carry step: adder(k) -> adder(k + 1).

    Parameterised by the addition box ``full_adder``: a ``symmetric.Box``
    grows a Diagram-valued adder, a ``Hypergraph`` or ``CMap`` the equivalent
    graph-valued one, from one recipe -- everything else is taken from
    ``type(full_adder)``.
    """
    factory = type(full_adder)
    bit = full_adder.dom[:1]
    reorder1 = list(range(1, k + 1)) + [0, k + 1, k + 2]
    reorder2 = [k] + list(range(k)) + [k + 1]
    step = adder.tensor(factory.id(bit @ bit))
    step = step.then(permutation(factory, reorder1, step.cod))
    step = step.then(factory.id(bit ** k).tensor(full_adder))
    return step.then(permutation(factory, reorder2, step.cod))


def build_adder(full_adder, n):
    """ Build the ``n``-cell ripple-carry adder from scratch (untimed). """
    adder = full_adder
    for k in range(1, n):
        adder = adder_step(full_adder, adder, k)
    return adder


def make_spiral(n_cups, x=None, box=Box):
    """ The diagram of arXiv:1804.07832 in the category of ``box``. """
    x = x or Ty('x')
    unit_type = x[:0]
    unit = box('unit', unit_type, x)
    counit = box('counit', x, unit_type)
    cup = box('cup', x @ x, unit_type)
    cap = box('cap', unit_type, x @ x)
    identity = type(unit).id
    result = unit
    for i in range(n_cups):
        result = result >> identity(x ** i) @ cap @ identity(x ** (i + 1))
    result = result >>\
        identity(x ** n_cups) @ counit @ identity(x ** n_cups)
    for i in range(n_cups):
        result = result >>\
            identity(x ** (n_cups - i - 1)) @ cup\
            @ identity(x ** (n_cups - i - 1))
    return result, unit, counit


def with_snakes(f, n):
    """ Wrap ``f`` in ``n`` transpose round-trips; equals ``f`` cluttered.

    Transposing back and forth is a no-op on ``f``'s boundary type, so ``g``
    stays the same constant width at every step, only growing snake-shaped
    clutter that a normal form / hypergraph pass must yank back out. """
    g = f
    for _ in range(n):
        g = g.transpose(left=True).transpose(left=False)
    return g


def full_adder_function(a, b, carry_in):
    """ A full adder as Python bit ops: ``(sum, carry_out)``. """
    return a ^ b ^ carry_in, (a & b) | (carry_in & (a ^ b))


def carry_save_value(outputs):
    """ Decode a carry-save accumulator: sum bit + weight-2 carries. """
    return outputs[0] + 2 * sum(outputs[1:])


def _NOT():
    bit = Ty('bit')
    return Box('NOT', bit, bit)


def _full_adder():
    bit = Ty('bit')
    return Box('FA', bit @ bit @ bit, bit @ bit)


def _adder_functor(full_adder):
    return Functor(
        ob_map={full_adder.dom[:1]: int},
        ar_map={full_adder: full_adder_function}, cod=Function)


def dense_box(name, dom, cod):
    """ A tensor box filled with ones, with every wire of dimension two. """
    return tensor.Box(name, dom, cod, np.ones(2 ** len(dom @ cod)))


def close_boundaries(diagram):
    """ Compose atomic states and effects around an open tensor diagram. """
    unit = diagram.dom[:0]
    if diagram.dom:
        state = dense_box('state', unit, diagram.dom[:1])
        diagram = repeated(lambda a, b: a.tensor(b), state, len(diagram.dom))\
            >> diagram
    if diagram.cod:
        effect = dense_box('effect', diagram.cod[:1], unit)
        diagram >>= repeated(
            lambda a, b: a.tensor(b), effect, len(diagram.cod))
    return diagram


def _tensor_NOT():
    bit = tensor.Dim(2)
    return dense_box('NOT', bit, bit)


def _tensor_full_adder():
    bit = tensor.Dim(2)
    return dense_box('FA', bit ** 3, bit ** 2)


# --- k-fold tensor ---------------------------------------------------------

@case("k-fold tensor (Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200, 500)))
def test_tensor_diagram(benchmark, n):
    box = _NOT()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.tensor(b), box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("k-fold tensor, 1 layer (Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, 100, full=(200, 500, 1000)))
def test_tensor_single_layer_diagram(benchmark, n):
    box = _NOT()
    benchmark.pedantic(
        lambda: single_layer_tensor(box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("k-fold tensor (Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_hypergraph(benchmark, n):
    hbox = _NOT().to_hypergraph()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.tensor(b), hbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("k-fold tensor (CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_cmap(benchmark, n):
    mbox = _NOT().to_map()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.tensor(b), mbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


# tensor.Diagram.eval

@case("k-fold tensor (Tensor)")
@pytest.mark.parametrize("n", sizes(5, 10, 15, full=(20,)))
def test_tensor_evaluation(benchmark, n):
    diagram = close_boundaries(repeated(
        lambda a, b: a.tensor(b), _tensor_NOT(), n))
    benchmark.pedantic(
        lambda: diagram.eval(), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- staircase / foliation -------------------------------------------------

@case("staircase foliation (Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, full=(50,)))  # ~O(n^3): 50 ~ 3.7s
def test_foliation_diagram(benchmark, n):
    st = staircase(_NOT(), n)
    benchmark.pedantic(
        lambda: st.foliation(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("staircase to hypergraph (Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, full=(50,)))  # ~O(n^3): 50 ~ 4.2s
def test_staircase_to_hypergraph(benchmark, n):
    st = staircase(_NOT(), n)
    benchmark.pedantic(
        lambda: st.to_hypergraph(), rounds=ROUNDS, warmup_rounds=WARMUP)


@case("staircase to cmap (CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, full=(50,)))
def test_staircase_to_cmap(benchmark, n):
    st = staircase(_NOT(), n)
    benchmark.pedantic(
        lambda: st.to_map(), rounds=ROUNDS, warmup_rounds=WARMUP)


# tensor.Diagram.eval

@case("staircase evaluation (Tensor)")
@pytest.mark.parametrize("n", sizes(5, 10, 15, full=(20,)))
def test_staircase_evaluation(benchmark, n):
    diagram = close_boundaries(staircase(_tensor_NOT(), n))
    benchmark.pedantic(
        lambda: diagram.eval(), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- k-fold series ---------------------------------------------------------

@case("k-fold series (Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, 100, full=(200, 500, 1000)))
def test_series_diagram(benchmark, n):
    box = _NOT()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.then(b), box, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("k-fold series (Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_series_hypergraph(benchmark, n):
    hbox = _NOT().to_hypergraph()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.then(b), hbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("k-fold series (CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_series_cmap(benchmark, n):
    mbox = _NOT().to_map()
    benchmark.pedantic(
        lambda: repeated(lambda a, b: a.then(b), mbox, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


# tensor.Diagram.eval

@case("k-fold series (Tensor)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(30, 40)))
def test_series_evaluation(benchmark, n):
    diagram = repeated(lambda a, b: a.then(b), _tensor_NOT(), n)
    benchmark.pedantic(
        lambda: diagram.eval(), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- ripple-carry adder ----------------------------------------------------

@case("adder step (Diagram)")
@pytest.mark.parametrize("n", sizes(2, 5, 10, 20, full=(50, 100)))
def test_adder_step_diagram(benchmark, n):
    full_adder = _full_adder()
    adder = build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("adder step (Hypergraph)")
@pytest.mark.parametrize("n", sizes(2, 5, 10, 20, full=(50,)))  # ~O(n^2)
def test_adder_step_hypergraph(benchmark, n):
    full_adder = _full_adder().to_hypergraph()
    adder = build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("adder step (CMap)")
@pytest.mark.parametrize("n", sizes(2, 5, 10, 20, full=(50,)))
def test_adder_step_cmap(benchmark, n):
    full_adder = _full_adder().to_map()
    adder = build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: adder_step(full_adder, adder, n),
        rounds=ROUNDS, warmup_rounds=WARMUP)


@case("adder functor (Diagram)")
@pytest.mark.parametrize("n", sizes(2, 5, 10, 20, full=(50, 100)))
def test_adder_functor_diagram(benchmark, n):
    full_adder = _full_adder()
    functor = _adder_functor(full_adder)
    adder = build_adder(full_adder, n)
    benchmark.pedantic(
        lambda: functor(adder), rounds=ROUNDS, warmup_rounds=WARMUP)


# tensor.Diagram.eval

@case("adder evaluation (Tensor)")
@pytest.mark.parametrize("n", sizes(2, 5, 10))
def test_adder_evaluation(benchmark, n):
    diagram = close_boundaries(build_adder(_tensor_full_adder(), n))
    benchmark.pedantic(
        lambda: diagram.eval(), rounds=ROUNDS, warmup_rounds=WARMUP)


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


@case("spiral evaluation (Tensor)")
@pytest.mark.parametrize("n", sizes(2, 5, 10))
def test_spiral_evaluation(benchmark, n):
    diagram = make_spiral(n, tensor.Dim(2), dense_box)[0]
    benchmark.pedantic(
        lambda: diagram.eval(), rounds=ROUNDS, warmup_rounds=WARMUP)


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


@case("transpose evaluation (Tensor)")
@pytest.mark.parametrize("n", sizes(1, 3, 6))
def test_transpose_evaluation(benchmark, n):
    bit = tensor.Dim(2)
    diagram = with_snakes(dense_box('f', bit, bit), n)
    benchmark.pedantic(
        lambda: diagram.eval(), rounds=ROUNDS, warmup_rounds=WARMUP)


# --- correctness (run once, not benchmarks) --------------------------------

def test_adder_functor_correct():
    """ The functor compiles adder(k) to a carry-save accumulator: encoding
    ``2k + 1`` random bits and decoding the outputs with weights
    ``[1, 2, 2, ...]`` recovers their popcount. """
    rng = random.Random(0)
    full_adder = _full_adder()
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


def test_tensor_evaluation_holds():
    """ Three closed all-one nodes evaluate to ``4 ** 3``. """
    diagram = close_boundaries(staircase(_tensor_NOT(), 3))
    assert diagram.eval().array == 4 ** 3
