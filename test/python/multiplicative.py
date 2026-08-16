# -*- coding: utf-8 -*-

import itertools
from typing import List
from pytest import raises

from discopy.biclosed import *
from discopy.hypergraph import Functor, Hypergraph
from discopy.python import *
from discopy.utils import AxiomError


def test_Function():
    x, y, z = (complex, ), (bool, ), (float, )
    f = Function(dom=y, cod=exp(z, x),
                 inside=lambda y: lambda x: abs(x) ** 2 if y else 0)
    g = Function(dom=x + y, cod=z, inside=lambda x, y: f(y)(x))

    assert f.uncurry().curry()(True)(1j) == f(True)(1j)
    assert f.uncurry(left=False).curry(left=False)(True)(1j) == f(True)(1j)
    assert g.curry().uncurry()(1j, True) == g(1j, True)
    assert g.curry(left=False).uncurry(left=False)(1j, True) == g(1j, True)


def test_fixed_point():
    from math import sqrt
    phi = Function(lambda x=1: 1 + 1 / x, dom=(float,), cod=(float,)).fix()
    assert phi() == (1 + sqrt(5)) / 2


def test_trace():
    with raises(NotImplementedError):
        Function.id(int).trace(left=True)


def test_list_generic_in_function():
    func = Function(sum, List[int], int)
    assert func([1, 2, 3]) == 6


def test_Hypergraph_call():
    add = Function(lambda x, y: x + y, (int, int), (int, ))

    # A wire copied into both addends and the other one forwarded: the value
    # on wire 0 is read twice (by the box and by the output) and wire 1 once.
    f = Hypergraph[Function](
        dom=(int, int), cod=(int, int),
        boxes=(add, ),
        wires=((0, 1), (((1, 0), (2, )), ), (2, 0)))
    assert f(2, 3) == (5, 2)

    # A discarded input (wire 1 is read by nothing) and a single output.
    g = Hypergraph[Function](
        dom=(int, int), cod=(int, ),
        boxes=(), wires=((0, 1), (), (0, )))
    assert g(2, 3) == 2

    with raises(ValueError, match="Expected 2 wires for"):
        g(2)

    # A box that returns the wrong number of values is caught by evaluation.
    wrong = Function(lambda x: (x, x, x), (int, ), (int, int))
    h = Hypergraph[Function](
        dom=(int, ), cod=(int, int), boxes=(wrong, ),
        wires=((0, ), (((0, ), (1, 2)), ), (1, 2)))
    with Function.no_type_checking:
        with raises(ValueError, match="Expected 2 wires for"):
            h(1)


def test_Hypergraph_axioms():
    swap = Function(lambda x, y: (y, x), (int, int), (int, int))
    add = Function(lambda x, y: x + y, (int, int), (int, ))

    # Not causal: the swap reads wires that only its own output produces.
    not_causal = Hypergraph[Function](
        (int, ), (int, ), (swap, ), ((0, ), (((0, 1), (1, 2)), ), (2, )))
    with raises(AxiomError, match="is not causal"):
        not_causal(5)

    # Not left-monogamous: wire 0 is produced by the input and the box.
    not_left_monogamous = Hypergraph[Function](
        (int, ), (int, ), (add, ), ((0, ), (((0, 0), (0, )), ), (0, )))
    with raises(AxiomError, match="is not left-monogamous"):
        not_left_monogamous(5)


def test_Hypergraph_representation():
    add = Function(lambda x, y: x + y, (int, int), (int, ))
    f, g = (Hypergraph[Function](
        dom=(int, int), cod=(int, ), boxes=(add, ),
        wires=((0, 1), (((0, 1), (2, )), ), (2, ))) for _ in range(2))
    assert f == g and hash(f) == hash(g)
    assert f != Hypergraph[Function](
        dom=(int, int), cod=(int, ), boxes=(), wires=((0, 1), (), (0, )))
    assert repr(f).startswith("hypergraph.Hypergraph[Function](dom=")


def adder_step(full_adder, adder, k):
    """ One incremental ripple-carry step: adder(k) -> adder(k + 1). """
    factory = type(full_adder)
    bit = full_adder.dom[:1]
    reorder1 = list(range(1, k + 1)) + [0, k + 1, k + 2]
    reorder2 = [k] + list(range(k)) + [k + 1]
    step = adder.tensor(factory.id(bit @ bit))
    step = step.then(factory.permutation(reorder1, step.cod))
    step = step.then(factory.id(bit ** k).tensor(full_adder))
    return step.then(factory.permutation(reorder2, step.cod))


def test_Hypergraph_adder():
    """ The carry-save adder of the #346 benchmark, evaluated directly as a
    hypergraph of Python functions rather than compiled to a diagram. """
    from discopy.symmetric import Ty, Box

    bit = Ty('bit')
    full_adder = Box('FA', bit @ bit @ bit, bit @ bit)

    def full_adder_function(a, b, carry_in):
        return a ^ b ^ carry_in, (a & b) | (carry_in & (a ^ b))

    fa = Function(full_adder_function, (int, int, int), (int, int))

    def carry_save_value(outputs):
        return outputs[0] + 2 * sum(outputs[1:])

    F = Functor({bit: int}, {full_adder: fa}, cod=Function)
    full_adder_hg = full_adder.to_hypergraph()
    adder = full_adder_hg
    for k in range(1, 5):
        compiled = F(adder)
        # The swaps of the reordering permutations are absorbed into the
        # wiring, so the only boxes left are the full adders themselves.
        assert all(box == fa for box in compiled.boxes)
        for bits in itertools.product((0, 1), repeat=len(compiled.dom)):
            assert carry_save_value(compiled(*bits)) == sum(bits)
        adder = adder_step(full_adder_hg, adder, k)
