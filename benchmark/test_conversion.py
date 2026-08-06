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

import pytest

from benchmark.config import ROUNDS, WARMUP, case, sizes
from benchmark.generators import reverse_permutation, series, snake, tensor


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
    morphism = reverse_permutation("Diagram", n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("permutation (Diagram -> Hypergraph)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_diagram_to_hypergraph(benchmark, n):
    morphism = reverse_permutation("Diagram", n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("permutation (Hypergraph -> Diagram)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_hypergraph_to_diagram(benchmark, n):
    morphism = reverse_permutation("Hypergraph", n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("permutation (Hypergraph -> CMap)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_hypergraph_to_map(benchmark, n):
    morphism = reverse_permutation("Hypergraph", n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("permutation (CMap -> Diagram)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_map_to_diagram(benchmark, n):
    morphism = reverse_permutation("CMap", n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("permutation (CMap -> Hypergraph)")
@pytest.mark.parametrize("n", sizes(5, 10, 20, full=(50,)))
def test_permutation_map_to_hypergraph(benchmark, n):
    morphism = reverse_permutation("CMap", n)
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
