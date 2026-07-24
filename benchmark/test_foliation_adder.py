# -*- coding: utf-8 -*-

"""
Benchmark for :meth:`discopy.symmetric.Diagram.to_layers`, the native
foliation, on the ripple-carry adder.

The adder is the natural stress test for permutation handling: chaining
``n`` full-adder boxes routes a carry across the whole register, so the
naive diagram is dominated by a *staircase* of ``Swap`` boxes -- about
``n ** 2`` of them. Foliating with :meth:`Diagram.to_layers` collapses each
routing permutation into a single :class:`Permutation` layer, turning the
``~ n ** 2`` naive layers into exactly ``2 * n`` (``n`` box-layers, one per
``FA``, interleaved with ``n`` permutation bands).

This module is deliberately kept outside of ``test/*/*.py`` so that it is
not collected by the default ``pytest`` run (see ``testpaths`` in
``pyproject.toml``). Run it explicitly, e.g.

    uv run pytest benchmark/test_foliation_adder.py \\
        --log-cli-level=INFO --log-cli-format="%(message)s"

The ``draw_comparison`` helper renders the naive diagram next to its
foliation (used to produce ``docs/_static/symmetric/adder_foliation.png``).
"""

import logging

from discopy.symmetric import Ty, Box, Swap, Diagram

bit = Ty('bit')
FA = Box('FA', bit @ bit @ bit, bit @ bit)

# Sweep used for the reported table; the (expensive, O(n ** 2)) round-trip
# and depth checks are only asserted up to ASSERT_MAX to keep the run quick.
SIZES = [2, 3, 4, 6, 8, 12, 16]
ASSERT_MAX = 8


def permutation(factory, xs, dom):
    """ A permutation arrow built from swaps, generic over the category.

    Mirrors :meth:`symmetric.Diagram.permutation` using only ``id``,
    ``swap``, ``tensor`` and ``then`` so the same recipe drives any monoidal
    category. (Same construction as the composition benchmark's adder.)
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
    """ One incremental ripple-carry step: ``adder(k) -> adder(k + 1)``. """
    factory = type(full_adder)
    b = full_adder.dom[:1]
    reorder1 = list(range(1, k + 1)) + [0, k + 1, k + 2]
    reorder2 = [k] + list(range(k)) + [k + 1]
    step = adder.tensor(factory.id(b @ b))
    step = step.then(permutation(factory, reorder1, step.cod))
    step = step.then(factory.id(b ** k).tensor(full_adder))
    return step.then(permutation(factory, reorder2, step.cod))


def adder(n, full_adder=FA):
    """ The ``n``-cell ripple-carry adder as a :class:`symmetric.Diagram`. """
    result = full_adder
    for k in range(1, n):
        result = adder_step(full_adder, result, k)
    return result


def count_swaps(diagram):
    """ Number of :class:`Swap` boxes in the naive diagram. """
    return sum(
        1 for layer in diagram.inside
        for box in layer.boxes if isinstance(box, Swap))


def foliation_sizes(diagram):
    """ ``(total, box_layers, perm_layers)`` for ``diagram.to_layers()``. """
    layers = diagram.to_layers()
    box_layers = sum(1 for layer in layers if layer.boxes)
    return len(layers), box_layers, len(layers) - box_layers


def test_adder_foliation():
    """ Foliation shrinks the swap-dominated adder to ``2 * n`` layers. """
    header = (f"{'n':>3} | {'naive':>6} {'swaps':>6} | "
              f"{'foliation':>9} {'box':>4} {'perm':>5} | {'shrink':>7}")
    logging.info(header)
    logging.info("-" * len(header))
    for n in SIZES:
        diagram = adder(n)
        naive = len(diagram.inside)
        swaps = count_swaps(diagram)
        total, box_layers, perm_layers = foliation_sizes(diagram)

        # The foliation is exactly 2 * n: one box-layer per FA (so box-layers
        # is the depth -- the schedule is depth-optimal / ASAP) and one
        # permutation band between them.
        assert box_layers == n
        assert total == 2 * n
        # ... and it is never longer than the naive layer list.
        assert total <= naive

        if n <= ASSERT_MAX:
            assert diagram.depth() == box_layers       # depth-optimal
            with Diagram.hypergraph_equality:           # round-trip
                assert Diagram.from_layers(diagram.to_layers()) == diagram

        logging.info(
            f"{n:>3} | {naive:>6} {swaps:>6} | "
            f"{total:>9} {box_layers:>4} {perm_layers:>5} | "
            f"{naive / total:>6.1f}x")


def draw_comparison(n=4, path=None):
    """ Render the naive adder next to its foliation. """
    from discopy.drawing import Equation
    diagram = adder(n)
    Equation(diagram, diagram.foliation_drawing(), symbol="~").draw(
        path=path, draw_type_labels=False, figsize=(12, 9))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    test_adder_foliation()
    draw_comparison(path="docs/_static/symmetric/adder_foliation.png")
