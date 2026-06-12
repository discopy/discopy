from __future__ import annotations

import pytest

from discopy.closed import (
    Application,
    Abstraction,
    Variable,
    Ty,
    Functor,
    Box,
    Eval,
    Coeval,
    CombinatorialMap,
)


def test_exp():
    X, Y = Ty("X"), Ty("Y")
    assert X >> Y == Y**X == Y << X
    assert X @ Ty() == X == Ty() @ X


def test_str():
    X, Y = Ty("X"), Ty("Y")
    f = X(lambda x: (X >> Y)(lambda y: y(x)))
    assert str(f) == "X(lambda x: (X >> Y)(lambda y: y(x)))"


def test_term_equality_is_alpha_equivalence():
    X, Y = map(Ty, "XY")
    x, y = Variable(X, "x"), Variable(X, "y")
    assert X(lambda x: x) == X(lambda y: y)
    assert hash(X(lambda x: x)) == hash(X(lambda y: y))
    assert X(lambda x: (X >> Y)(lambda f: f(x)))\
        == X(lambda y: (X >> Y)(lambda g: g(y)))
    assert x != y
    assert Abstraction(x, y) != Abstraction(y, x)


def test_python_Functor():
    x, y, z = map(Ty, "xyz")
    f, g = Box("f", y, x >> z), Box("g", x @ y, z)

    from discopy.python import Function

    F = Functor(
        ob={x: complex, y: bool, z: float},
        ar={
            f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda x, y: abs(x + 1j if y else -1j),
        },
        cod=Function,
    )

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.curry().uncurry())(1j, True) == F(g)(1j, True)


def test_python_Func():
    x, y, z = map(Ty, "xyz")
    f, g = Box("f", y, x >> z), Box("g", x @ y, z)

    from discopy.python import Function

    ob = lambda typ: {"x": complex, "y": bool, "z": float}[str(typ)]

    F = Functor(
        ob=ob,
        ar={
            f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda x, y: abs(x + 1j if y else -1j),
        },
        cod=Function,
    )

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.curry().uncurry())(1j, True) == F(g)(1j, True)


