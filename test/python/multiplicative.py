# -*- coding: utf-8 -*-

from typing import List
from pytest import raises

from discopy.biclosed import *
from discopy.python import Function, exp


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


def test_Functor_into_Function_folds_with_matmul():
    from discopy import monoidal
    from discopy.monoidal import Ty, Box, Functor

    # python.Function.ob is the free monoid List[type], so a Functor folds the
    # image of a type with the monoid product @, not tuple + (issue #728).
    assert Function.ob is monoidal.List[type]
    x = Ty('x')
    g = Box('g', x @ x, x)
    F = Functor(ob_map={x: bool}, ar_map={g: lambda a, b: a and b}, cod=Function)
    assert F(x @ x) == F(x) @ F(x) == monoidal.List[type](bool, bool)
    assert F(Ty()) == monoidal.List[type]()
    assert F(g)(True, True) is True and F(g)(True, False) is False
