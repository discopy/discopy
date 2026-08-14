# -*- coding: utf-8 -*-

""" Morphism generators shared by the benchmark suites. """

from collections.abc import Callable

from discopy import monoidal
from discopy.abc import MonoidalCategory, RigidCategory, SymmetricCategory


def repeated[T](op: Callable[[T, T], T], box: T, k: int) -> T:
    """ Combine ``k`` copies of ``box`` with ``op``, by repeated doubling. """
    if k == 1:
        return box
    half = repeated(op, box, k // 2)
    result = op(half, half)
    return op(result, box) if k % 2 else result


def single_layer_tensor[D: monoidal.Diagram](box: D, k: int) -> D:
    """ The ``k``-fold tensor of ``box`` as a *single* :class:`Layer`.

    Tensoring two diagrams pads every layer of each with the other's wires,
    so a ``k``-fold tensor by repeated ``@`` rebuilds the layer list over and
    over. The same morphism is one layer with the ``k`` boxes side by side,
    sidestepping that overhead -- the residual cost is only the type
    concatenation in ``Layer``.
    """
    empty = box.dom[:0]
    layer = box.layer_factory(empty, box, empty, *([box, empty] * (k - 1)))
    return box.factory((layer,), layer.dom, layer.cod)


def staircase[C1: MonoidalCategory](box: C1, k: int) -> C1:
    """ The ``k``-fold tensor of ``box`` as a staircase of ``k`` layers -- the
    same morphism as :func:`single_layer_tensor`, spread over ``k`` layers. """
    result = box
    for _ in range(k - 1):
        result = result @ box
    return result


def adder_step[C1: SymmetricCategory](
        full_adder: C1, adder: C1, k: int) -> C1:
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
    return (adder @ factory.id(bit @ bit)
            >> factory.permutation(reorder1, bit ** (k + 3))
            >> factory.id(bit ** k) @ full_adder
            >> factory.permutation(reorder2, bit ** (k + 2)))


def build_adder[C1: SymmetricCategory](full_adder: C1, n: int) -> C1:
    """ Build the ``n``-cell ripple-carry adder from scratch. """
    adder = full_adder
    for k in range(1, n):
        adder = adder_step(full_adder, adder, k)
    return adder


def spiral[D: monoidal.Diagram](category: type[D], n_cups: int) -> D:
    """ The diagram of arXiv:1804.07832, built with ``category`` boxes. """
    x, empty = category.ob('x'), category.ob()
    wire = lambda k: category.id(x ** k)
    unit, counit = category('unit', empty, x), category('counit', x, empty)
    cup, cap = category('cup', x @ x, empty), category('cap', empty, x @ x)
    result = unit
    for i in range(n_cups):
        result >>= wire(i) @ cap @ wire(i + 1)
    result >>= wire(n_cups) @ counit @ wire(n_cups)
    for i in range(n_cups):
        result >>= wire(n_cups - i - 1) @ cup @ wire(n_cups - i - 1)
    return result


def transpose_snakes[C1: RigidCategory](morphism: C1, n: int) -> C1:
    """ Wrap ``morphism`` in ``n`` snakes, by alternating transposes.

    Transposing back and forth is a no-op on its boundary type, so the result
    stays the same constant width at every step, only growing snake-shaped
    clutter that a normal form or graph conversion must yank back out.
    """
    result = morphism
    for _ in range(n):
        result = result.transpose(left=True).transpose(left=False)
    return result


def not_box[B: monoidal.Box](factory: type[B]) -> B:
    """ The unary box used by tensor, staircase and series workloads. """
    bit = factory.ob('bit')
    return factory('NOT', bit, bit)


def full_adder_box[B: monoidal.Box](factory: type[B]) -> B:
    """ The ternary-to-binary box used by the adder workload. """
    bit = factory.ob('bit')
    return factory('FA', bit @ bit @ bit, bit @ bit)


def series[C0, C1](
        category: SymmetricCategory[C0, C1], box: C1, n: int) -> C1:
    """ A depth-``n`` source morphism. """
    return repeated(category.then, box, n)


def tensor[C0, C1](
        category: SymmetricCategory[C0, C1], box: C1, n: int) -> C1:
    """ A width-``n`` source morphism. """
    return repeated(category.tensor, box, n)


def reverse_permutation[C0, C1](
        category: SymmetricCategory[C0, C1], n: int) -> C1:
    """ A routing-heavy reversal on ``n`` wires. """
    x = category.ob("x")
    return category.permutation(list(reversed(range(n))), [x] * n)


def snake[C0, C1](category: RigidCategory[C0, C1], n: int) -> C1:
    """ A snake made of ``n`` zipping cups and caps. """
    x = category.ob("x")
    cups = repeated(category.tensor, category.cups(x, x.r), n)
    caps = repeated(category.tensor, category.caps(x.r, x), n)
    return category.id(x) @ caps >> cups @ category.id(x)
