# -*- coding: utf-8 -*-

"""
Conversion benchmarks between :class:`discopy.symmetric.Diagram`,
:class:`discopy.symmetric.Hypergraph` and :class:`discopy.symmetric.CMap`.

The workloads exercise a deep series, a wide tensor product, a routing-heavy
reverse permutation and a snake of zipping cups and caps.  Every source
morphism is built in its native representation before the timer starts; the
measured operation is only the bound ``to_diagram``, ``to_hypergraph`` or
``to_map`` method.
"""

import os
import time

import pytest

from discopy import compact
from discopy.symmetric import Box, Ty


_FULL = "bench:full" in os.environ.get("BENCH_FLAGS", "").lower()


def sizes(*base, full=()):
    """Include the heavy tail only under ``BENCH_FLAGS=bench:full``."""
    return list(base) + (list(full) if _FULL else [])


def case(group):
    """Measure CPU time with garbage collection disabled."""
    return pytest.mark.benchmark(
        group=group, timer=time.process_time, disable_gc=True)


ROUNDS, WARMUP = 3, 1


def repeated(op, box, n):
    """Combine ``n`` copies of ``box`` with ``op`` by repeated doubling."""
    if n == 1:
        return box
    half = repeated(op, box, n // 2)
    result = op(half, half)
    return op(result, box) if n % 2 else result


def reverse_permutation(factory, dom):
    """Build the reversal of ``dom`` using native swaps and composition."""
    if len(dom) <= 1:
        return factory.id(dom)
    head = factory.swap(dom[:-1], dom[-1:])
    tail = factory.id(dom[-1:]).tensor(
        reverse_permutation(factory, dom[:-1]))
    return head.then(tail)


def source_box(representation):
    """An atomic endomorphism embedded in ``representation``."""
    box = Box("f", Ty("x"), Ty("x"))
    if representation == "Hypergraph":
        return box.to_hypergraph()
    if representation == "CMap":
        return box.to_map()
    return box


def series(representation, n):
    """A depth-``n`` source morphism."""
    box = source_box(representation)
    return repeated(lambda f, g: f.then(g), box, n)


def tensor(representation, n):
    """A width-``n`` source morphism."""
    box = source_box(representation)
    return repeated(lambda f, g: f.tensor(g), box, n)


def permutation(representation, n):
    """A routing-heavy source morphism on ``n`` wires."""
    box = source_box(representation)
    return reverse_permutation(type(box), box.dom ** n)


def snake(representation, n):
    """A snake made of ``n`` zipping cups and caps."""
    factory = {
        "Diagram": compact.Diagram,
        "Hypergraph": compact.Hypergraph,
        "CMap": compact.CMap,
    }[representation]
    x = compact.Ty("x")
    cups = repeated(lambda f, g: f.tensor(g), factory.cups(x, x.r), n)
    caps = repeated(lambda f, g: f.tensor(g), factory.caps(x.r, x), n)
    return factory.id(x).tensor(caps).then(cups.tensor(factory.id(x)))


# --- k-fold series ---------------------------------------------------------

@case("series (Diagram -> CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_series_diagram_to_map(benchmark, n):
    morphism = series("Diagram", n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("series (Diagram -> Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_series_diagram_to_hypergraph(benchmark, n):
    morphism = series("Diagram", n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("series (Hypergraph -> Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_series_hypergraph_to_diagram(benchmark, n):
    morphism = series("Hypergraph", n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("series (Hypergraph -> CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_series_hypergraph_to_map(benchmark, n):
    morphism = series("Hypergraph", n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("series (CMap -> Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_series_map_to_diagram(benchmark, n):
    morphism = series("CMap", n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("series (CMap -> Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_series_map_to_hypergraph(benchmark, n):
    morphism = series("CMap", n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- k-fold tensor ---------------------------------------------------------

@case("tensor (Diagram -> CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_diagram_to_map(benchmark, n):
    morphism = tensor("Diagram", n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("tensor (Diagram -> Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_diagram_to_hypergraph(benchmark, n):
    morphism = tensor("Diagram", n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("tensor (Hypergraph -> Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_hypergraph_to_diagram(benchmark, n):
    morphism = tensor("Hypergraph", n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("tensor (Hypergraph -> CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_hypergraph_to_map(benchmark, n):
    morphism = tensor("Hypergraph", n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("tensor (CMap -> Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(75,)))
def test_tensor_map_to_diagram(benchmark, n):
    morphism = tensor("CMap", n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("tensor (CMap -> Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_tensor_map_to_hypergraph(benchmark, n):
    morphism = tensor("CMap", n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- reverse permutation --------------------------------------------------

@case("permutation (Diagram -> CMap)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_diagram_to_map(benchmark, n):
    morphism = permutation("Diagram", n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("permutation (Diagram -> Hypergraph)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_diagram_to_hypergraph(benchmark, n):
    morphism = permutation("Diagram", n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("permutation (Hypergraph -> Diagram)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_hypergraph_to_diagram(benchmark, n):
    morphism = permutation("Hypergraph", n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("permutation (Hypergraph -> CMap)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_hypergraph_to_map(benchmark, n):
    morphism = permutation("Hypergraph", n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("permutation (CMap -> Diagram)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_map_to_diagram(benchmark, n):
    morphism = permutation("CMap", n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("permutation (CMap -> Hypergraph)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_map_to_hypergraph(benchmark, n):
    morphism = permutation("CMap", n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- zipping cups and caps -------------------------------------------------

@case("snake (Diagram -> CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_snake_diagram_to_map(benchmark, n):
    morphism = snake("Diagram", n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("snake (Diagram -> Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_snake_diagram_to_hypergraph(benchmark, n):
    morphism = snake("Diagram", n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("snake (Hypergraph -> Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_snake_hypergraph_to_diagram(benchmark, n):
    morphism = snake("Hypergraph", n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("snake (Hypergraph -> CMap)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_snake_hypergraph_to_map(benchmark, n):
    morphism = snake("Hypergraph", n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("snake (CMap -> Diagram)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_snake_map_to_diagram(benchmark, n):
    morphism = snake("CMap", n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("snake (CMap -> Hypergraph)")
@pytest.mark.parametrize("n", sizes(10, 20, 50, full=(100, 200)))
def test_snake_map_to_hypergraph(benchmark, n):
    morphism = snake("CMap", n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)
