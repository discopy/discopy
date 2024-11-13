# -*- coding: utf-8 -*-

from pytest import raises

from discopy.closed import *
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


def test_FinSet():
    from discopy.markov import Ty, Diagram, Functor, Category
    from discopy.python.finset import Dict

    x = Ty('x')
    copy, discard, swap = Diagram.copy(x), Diagram.copy(x, 0), Diagram.swap(x, x)
    F = Functor({x: 1}, {}, cod=Category(int, Dict))

    assert F(copy >> discard @ x) == F(Diagram.id(x)) == F(copy >> x @ discard)
    assert F(copy >> copy @ x) == F(Diagram.copy(x, 3)) == F(copy >> x @ copy)
    assert F(copy >> swap) == F(copy)


def test_additive_Function():
    from discopy.python.additive import Ty as T, Function
    
    x = (int, )
    assert Function.swap(x, x).trace()(42) == 42  # Yanking equation

    from discopy.interaction import Ty, Diagram
    
    T, D = Ty[tuple], Diagram[Function]

    assert D.id(T(x, x)).transpose().inside(42, 0) == (42, 0)\
        == D.id(T(x, x)).transpose(left=True).inside(42, 0)
    assert D.id(T(x, x)).transpose().inside(42, 1) == (42, 1)\
        == D.id(T(x, x)).transpose(left=True).inside(42, 1)
