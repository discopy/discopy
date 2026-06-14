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
    Substitution,
    pack,
    unpack,
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


def test_substitution_under_abstraction():
    X = Ty("X")
    x, y, z = (Variable(X, name) for name in "xyz")
    assert Substitution({x: z})(Abstraction(x, x)) == Abstraction(x, x)
    assert Substitution({y: z})(Abstraction(x, y)) == Abstraction(x, z)


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


def test_pack_unpack_terms():
    X, Y = map(Ty, "XY")
    x, y = Variable(X, "x"), Variable(Y, "y")

    pair = pack(x, y)
    assert pair.dom == X @ Y
    assert pair.cod == X @ Y
    assert pair.to_diagram().dom == X @ Y
    assert pair.to_diagram().cod == X @ Y
    assert pair.to_map() == CombinatorialMap.id(X @ Y)

    swap = unpack(pair, lambda a, b: pack(b, a))
    assert swap.dom == X @ Y
    assert swap.cod == Y @ X
    assert swap.to_diagram().dom == X @ Y
    assert swap.to_diagram().cod == Y @ X
    assert swap.to_map().dom == X @ Y
    assert swap.to_map().cod == Y @ X

    assert unpack(pair, lambda a, b: pack(a, b))\
        == unpack(pair, lambda c, d: pack(c, d))


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
