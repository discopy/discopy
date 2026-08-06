# -*- coding: utf-8 -*-

""" Morphism generators shared by the benchmark suites. """

from discopy import compact
from discopy.monoidal import Layer
from discopy.symmetric import Ty, Box, Id, Diagram


def repeated(op, box, k):
    """ Combine ``k`` copies of ``box`` with ``op``, by repeated doubling. """
    if k == 1:
        return box
    half = repeated(op, box, k // 2)
    result = op(half, half)
    return op(result, box) if k % 2 else result


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
    result = box
    for _ in range(k - 1):
        result = result @ box
    return result


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
    """ Build the ``n``-cell ripple-carry adder from scratch. """
    adder = full_adder
    for k in range(1, n):
        adder = adder_step(full_adder, adder, k)
    return adder


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


def with_snakes(morphism, n):
    """ Wrap ``morphism`` in ``n`` transpose round-trips; equals it cluttered.

    Transposing back and forth is a no-op on its boundary type, so the result
    stays the same constant width at every step, only growing snake-shaped
    clutter that a normal form or graph conversion must yank back out.
    """
    result = morphism
    for _ in range(n):
        result = result.transpose(left=True).transpose(left=False)
    return result


def not_box():
    """ The unary box used by tensor, staircase and series workloads. """
    bit = Ty('bit')
    return Box('NOT', bit, bit)


def full_adder_box():
    """ The ternary-to-binary box used by the adder workload. """
    bit = Ty('bit')
    return Box('FA', bit @ bit @ bit, bit @ bit)


def source_box(representation):
    """ An atomic endomorphism embedded in ``representation``. """
    box = Box("f", Ty("x"), Ty("x"))
    if representation == "Hypergraph":
        return box.to_hypergraph()
    if representation == "CMap":
        return box.to_map()
    return box


def series(representation, n):
    """ A depth-``n`` source morphism. """
    box = source_box(representation)
    return repeated(lambda f, g: f.then(g), box, n)


def tensor(representation, n):
    """ A width-``n`` source morphism. """
    box = source_box(representation)
    return repeated(lambda f, g: f.tensor(g), box, n)


def reverse_permutation(representation, n):
    """ A routing-heavy reversal on ``n`` wires. """
    box = source_box(representation)
    return permutation(type(box), list(reversed(range(n))), box.dom ** n)


def snake(representation, n):
    """ A snake made of ``n`` zipping cups and caps. """
    factory = {
        "Diagram": compact.Diagram,
        "Hypergraph": compact.Hypergraph,
        "CMap": compact.CMap,
    }[representation]
    x = compact.Ty("x")
    cups = repeated(lambda f, g: f.tensor(g), factory.cups(x, x.r), n)
    caps = repeated(lambda f, g: f.tensor(g), factory.caps(x.r, x), n)
    return factory.id(x).tensor(caps).then(cups.tensor(factory.id(x)))
