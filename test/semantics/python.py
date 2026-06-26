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


def test_FinSet():
    from discopy.markov import Ty, Diagram, Functor
    from discopy.python import finset

    p = finset.Permutation.swap(2, 3)
    assert isinstance(p, finset.Function)
    assert isinstance(p, finset.SymmetricCategory)
    assert p == (3, 4, 0, 1, 2)
    assert p[-1] == 2
    assert p >> p.dagger() == finset.Permutation.id(5)
    assert p @ finset.Permutation((1, 0)) == (3, 4, 0, 1, 2, 6, 5)
    assert finset.Function.swap(2, 3).inside == dict(enumerate(p))
    assert finset.Permutation((1, 0, 3, 2)).cycles() == ((0, 1), (2, 3))
    assert finset.Permutation.from_cycles([(0, 1), (2, 3)], 4)\
        == (1, 0, 3, 2)
    assert finset.Permutation((1, 0)).is_fixpoint_free_involution()
    assert not finset.Permutation((0,)).is_fixpoint_free_involution()
    assert finset.Permutation.identity(2) == (0, 1)
    assert finset.Permutation((1, 0)).then((1, 0)) == (0, 1)
    assert finset.Permutation((1, 0, 2)).then((1, 2, 0)) == (2, 1, 0)
    assert finset.Permutation((1, 2, 0)).dagger() == (2, 0, 1)
    assert finset.Permutation((1, 0, 2)).conjugate((2, 0, 1))\
        == (2, 1, 0)
    assert finset.Permutation((1, 2, 0)).cycle(1) == (1, 2, 0)
    with raises(ValueError):
        finset.Permutation((0, 0))
    with raises(ValueError):
        finset.Permutation((0,), size=2)
    with raises(ValueError):
        finset.Permutation.from_cycles([(0, 0)], 1)
    with raises(ValueError):
        finset.Permutation.from_cycles([(0, 2)], 2)
    with raises(ValueError):
        finset.Permutation.from_cycles([(0, 1), (1, 2)], 3)
    with raises(ValueError):
        finset.Permutation.from_transpositions([(0, 0)], 2)
    with raises(ValueError):
        finset.Permutation.from_transpositions([(0, 2)], 2)
    with raises(ValueError):
        finset.Permutation.from_transpositions([(0, 1), (1, 2)], 4)
    with raises(ValueError):
        finset.Permutation((0,)).cycle(1)

    x = Ty('x')
    copy, discard, swap = Diagram.copy(x), Diagram.copy(x, 0), Diagram.swap(x, x)
    F = Functor({x: 1}, {}, cod=finset.Function)

    assert F(copy >> discard @ x) == F(Diagram.id(x)) == F(copy >> x @ discard)
    assert F(copy >> copy @ x) == F(Diagram.copy(x, 3)) == F(copy >> x @ copy)
    assert F(copy >> swap) == F(copy)


def test_additive_Function():
    from discopy.interaction import Ty, Diagram
    from discopy.python.additive import Ty as T, Function, Id, Swap, Merge

    X, xs = (int, ), []
    m, e = Function.merge(X, n=2), Function.merge(X, n=0)

    def f_inside(m, n=0):
        xs.append(m)
        return 3 * m + 1 if m % 2 else m // 2, 0 if n == 1 and m == 2 else 1

    f = Function(f_inside, X + X, X + X)
    g = Function(lambda m: m // 2, X, X)

    # This converges if https://en.wikipedia.org/wiki/Collatz_conjecture holds.
    assert f.trace()(42) == 1 and xs == [42, 21, 64, 32, 16, 8, 4, 2]

    eq = lambda *fs: all(fs[0].is_parallel(f) for f in fs) and all(
        len(set(f(42, i) for f in fs)) == 1 for i in range(len(fs[0].dom)))

    assert eq(Swap(X, X) >> m, m)
    assert eq(X @ e >> m, Id(X), e @ X >> m)
    assert eq(m @ X >> m, X @ m >> m, Function.merge(X, n=3))
    assert eq(Function.merge(X + X), X @ Swap(X, X) @ X >> m @ m)

    assert eq(Swap(X, X).trace(), Id(X))  # Yanking
    assert eq((f >> X @ g).trace(), (X @ g >> f).trace())  # Sliding
    assert eq((g @ X >> f).trace(), g >> f.trace())  # Left-naturality
    assert eq((f >> g @ X).trace(), f.trace() >> g)  # Right-naturality
    
    T, D = Ty[tuple], Diagram[Function]

    assert eq(D.id(T(X, X)).transpose().inside, Id(X + X))

def test_list_generic_in_function():
    func = Function(sum, List[int], int)
    assert func([1, 2, 3]) == 6
