# -*- coding: utf-8 -*-

from typing import List
from pytest import raises

from discopy.biclosed import *
from discopy.python import *


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


def test_tensor_is_simultaneous():
    """
    Tensoring ``n`` functions at once agrees with folding two at a time, and
    calls them in one Python frame rather than ``n`` nested ones.
    """
    from functools import reduce
    f = Function(lambda x: x + 1, (int, ), (int, ))
    g = Function(lambda x, y: (y, x), (int, int), (int, int))
    functions = [f, g, f, g]
    fold = reduce(lambda x, y: x.tensor(y), functions)
    nary = functions[0].tensor(*functions[1:])
    assert (nary.dom, nary.cod) == (fold.dom, fold.cod)
    assert nary(0, 1, 2, 3, 4, 5) == fold(0, 1, 2, 3, 4, 5)
    assert f.tensor() == f
    # The fold nests one closure per function, so calling it overflows the
    # stack where the n-ary tensor is flat, see discopy#489.
    identities = [Function(lambda x: x, (int, ), (int, ))] * 2000
    assert identities[0].tensor(*identities[1:])(*range(2000))\
        == tuple(range(2000))
