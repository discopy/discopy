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
    from discopy.interaction import Ty, Diagram
    from discopy.python.additive import Ty as T, Function, Id, Swap, Merge
    
    x = (int, )
    m, e = Function.merge(x, n=2), Function.merge(x, n=0)

    eq = lambda *fs: all(fs[0].is_parallel(f) for f in fs) and all(
        len(set(f(42, i) for f in fs)) == 1 for i in range(len(fs[0].dom)))
    
    assert eq(Swap(x, x) >> m, m)
    assert eq(x @ e >> m, Id(x), e @ x >> m)
    assert eq(m @ x >> m, x @ m >> m, Function.merge(x, n=3))
    assert eq(Function.merge(x + x), x @ Swap(x, x) @ x >> m @ m)
    assert eq(Swap(x, x).trace(), Id(x))
    assert eq(Function.swap(x, x).trace(), Function.id(x))
    
    T, D = Ty[tuple], Diagram[Function]

    assert eq(D.id(T(x, x)).transpose().inside, Id(x + x))
