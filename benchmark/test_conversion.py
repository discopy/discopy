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

from functools import partial

from discopy import compact, symmetric
from benchmark import generators as generator
from benchmark.config import ROUNDS, WARMUP, case, sizes


case = partial(case, suite="conversion")


# --- k-fold series ---------------------------------------------------------

@case("Diagram → CMap", "k-fold series",
      sizes(10, 20, 50, full=(100, 200)))
def test_series_diagram_to_cmap(benchmark, n):
    box = generator.not_box(symmetric.Box)
    morphism = generator.series(symmetric.Diagram, box, n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram → Hypergraph", "k-fold series",
      sizes(10, 20, 50, full=(100, 200)))
def test_series_diagram_to_hypergraph(benchmark, n):
    box = generator.not_box(symmetric.Box)
    morphism = generator.series(symmetric.Diagram, box, n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph → Diagram", "k-fold series",
      sizes(10, 20, 50, full=(100, 200)))
def test_series_hypergraph_to_diagram(benchmark, n):
    box = generator.not_box(symmetric.Box).to_hypergraph()
    morphism = generator.series(symmetric.Hypergraph, box, n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph → CMap", "k-fold series",
      sizes(10, 20, 50, full=(100, 200)))
def test_series_hypergraph_to_cmap(benchmark, n):
    box = generator.not_box(symmetric.Box).to_hypergraph()
    morphism = generator.series(symmetric.Hypergraph, box, n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap → Diagram", "k-fold series",
      sizes(10, 20, 50, full=(100, 200)))
def test_series_cmap_to_diagram(benchmark, n):
    box = generator.not_box(symmetric.Box).to_map()
    morphism = generator.series(symmetric.CMap, box, n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap → Hypergraph", "k-fold series",
      sizes(10, 20, 50, full=(100, 200)))
def test_series_cmap_to_hypergraph(benchmark, n):
    box = generator.not_box(symmetric.Box).to_map()
    morphism = generator.series(symmetric.CMap, box, n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- k-fold tensor ---------------------------------------------------------

@case("Diagram → CMap", "k-fold tensor",
      sizes(10, 20, 50, full=(100, 200)))
def test_tensor_diagram_to_cmap(benchmark, n):
    box = generator.not_box(symmetric.Box)
    morphism = generator.single_layer_tensor(box, n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram → Hypergraph", "k-fold tensor",
      sizes(10, 20, 50, full=(100, 200)))
def test_tensor_diagram_to_hypergraph(benchmark, n):
    box = generator.not_box(symmetric.Box)
    morphism = generator.single_layer_tensor(box, n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph → Diagram", "k-fold tensor",
      sizes(10, 20, 50, full=(100, 200)))
def test_tensor_hypergraph_to_diagram(benchmark, n):
    box = generator.not_box(symmetric.Box).to_hypergraph()
    morphism = generator.tensor(symmetric.Hypergraph, box, n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph → CMap", "k-fold tensor",
      sizes(10, 20, 50, full=(100, 200)))
def test_tensor_hypergraph_to_cmap(benchmark, n):
    box = generator.not_box(symmetric.Box).to_hypergraph()
    morphism = generator.tensor(symmetric.Hypergraph, box, n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap → Diagram", "k-fold tensor", sizes(10, 20, 50, full=(75,)))
def test_tensor_cmap_to_diagram(benchmark, n):
    box = generator.not_box(symmetric.Box).to_map()
    morphism = generator.tensor(symmetric.CMap, box, n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap → Hypergraph", "k-fold tensor",
      sizes(10, 20, 50, full=(100, 200)))
def test_tensor_cmap_to_hypergraph(benchmark, n):
    box = generator.not_box(symmetric.Box).to_map()
    morphism = generator.tensor(symmetric.CMap, box, n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- staircase -------------------------------------------------------------

@case("Diagram → Hypergraph", "staircase",
      sizes(10, 20, full=(50,)))  # ~O(n^3): 50 ~ 4.2s
def test_staircase_diagram_to_hypergraph(benchmark, n):
    box = generator.not_box(symmetric.Box)
    morphism = generator.staircase(box, n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram → CMap", "staircase", sizes(10, 20, full=(50,)))
def test_staircase_diagram_to_cmap(benchmark, n):
    box = generator.not_box(symmetric.Box)
    morphism = generator.staircase(box, n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- permutation ------------------------------------------------------------

@case("Diagram → CMap", "permutation", sizes(5, 10, 20, full=(50,)))
def test_permutation_diagram_to_cmap(benchmark, n):
    morphism = generator.reverse_permutation(symmetric.Diagram, n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram → Hypergraph", "permutation",
      sizes(5, 10, 20, full=(50,)))
def test_permutation_diagram_to_hypergraph(benchmark, n):
    morphism = generator.reverse_permutation(symmetric.Diagram, n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph → Diagram", "permutation",
      sizes(5, 10, 20, full=(50,)))
def test_permutation_hypergraph_to_diagram(benchmark, n):
    morphism = generator.reverse_permutation(symmetric.Hypergraph, n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Hypergraph → CMap", "permutation",
      sizes(5, 10, 20, full=(50,)))
def test_permutation_hypergraph_to_cmap(benchmark, n):
    morphism = generator.reverse_permutation(symmetric.Hypergraph, n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap → Diagram", "permutation", sizes(5, 10, 20, full=(50,)))
def test_permutation_cmap_to_diagram(benchmark, n):
    morphism = generator.reverse_permutation(symmetric.CMap, n)
    benchmark.pedantic(
        morphism.to_diagram, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("CMap → Hypergraph", "permutation",
      sizes(5, 10, 20, full=(50,)))
def test_permutation_cmap_to_hypergraph(benchmark, n):
    morphism = generator.reverse_permutation(symmetric.CMap, n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- snake ------------------------------------------------------------------

@case("Diagram → CMap", "snake", sizes(10, 20, 50, full=(100, 200)))
def test_snake_diagram_to_cmap(benchmark, n):
    morphism = generator.snake(compact.Diagram, n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram → Hypergraph", "snake",
      sizes(10, 20, 50, full=(100, 200)))
def test_snake_diagram_to_hypergraph(benchmark, n):
    morphism = generator.snake(compact.Diagram, n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- spiral (arXiv:1804.07832) ---------------------------------------------

@case("Diagram → Hypergraph", "spiral",
      sizes(10, 20, full=(50,)))  # ~O(n^3): 50 ~ 10s
def test_spiral_diagram_to_hypergraph(benchmark, n):
    morphism = generator.spiral(symmetric.Box, n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram → CMap", "spiral", sizes(10, 20, full=(50,)))
def test_spiral_diagram_to_cmap(benchmark, n):
    morphism = generator.spiral(symmetric.Box, n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- transpose snakes ------------------------------------------------------

@case("Diagram → Hypergraph", "transpose snakes",
      sizes(10, 20, 50, full=(100,)))
def test_transpose_diagram_to_hypergraph(benchmark, n):
    morphism = generator.transpose_snakes(
        compact.Box('f', compact.Ty('x'), compact.Ty('x')), n)
    benchmark.pedantic(
        morphism.to_hypergraph, rounds=ROUNDS, warmup_rounds=WARMUP)


@case("Diagram → CMap", "transpose snakes",
      sizes(10, 20, 50, full=(100,)))
def test_transpose_diagram_to_cmap(benchmark, n):
    morphism = generator.transpose_snakes(
        compact.Box('f', compact.Ty('x'), compact.Ty('x')), n)
    benchmark.pedantic(
        morphism.to_map, rounds=ROUNDS, warmup_rounds=WARMUP)


# --- correctness (run once, not benchmarks) --------------------------------

def test_transpose_snakes_are_absorbed():
    """ Compact graph encodings remove snakes from the wrapped box. """
    x = compact.Ty('x')
    f = compact.Box('f', x, x)
    assert generator.transpose_snakes(
        f, 3).to_hypergraph() == f.to_hypergraph()
    assert generator.transpose_snakes(f, 3).to_map() == f.to_map()
