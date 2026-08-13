from __future__ import annotations

from discopy.closed import *


def test_exp():
    X, Y = Ty('X'), Ty('Y')
    assert X >> Y == Y ** X == Y << X
    assert X @ Ty() == X == Ty() @ X


def test_str():
    X, Y = Ty("X"), Ty("Y")
    f = X(lambda x: (X >> Y)(lambda y: y(x)))
    assert str(f) == "X(lambda x: (X >> Y)(lambda y: y(x)))"


def test_python_Functor():
    x, y, z = map(Ty, "xyz")
    f, g = Box('f', y, x >> z), Box('g', x @ y, z)

    from discopy.python import Function
    F = Functor(
        ob_map={x: complex, y: bool, z: float},
        ar_map={f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda x, y: abs(x + 1j if y else -1j)},
        cod=Function)

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.curry().uncurry())(1j, True) == F(g)(1j, True)


def test_context_dom():
    """
    `Context.dom` instantiates `category.ob` before calling `.tensor`, so
    it works both for an empty context (regression test for #549) and for
    a non-empty one.
    """
    X = Ty('X')
    assert Context([]).dom == Ty()
    assert Context([Variable('x', X)]).dom == X


def test_discard():
    """ A discard in a closed diagram is a Discard, not a Copy with n=0. """
    x = Ty('x')
    assert Diagram.discard(x) == Copy(x, 0) == Discard(x)
    assert isinstance(Diagram.discard(x), Discard)
    from discopy import cat, closed  # noqa: F401  (used by eval)
    assert eval(repr(Discard(x))) == Discard(x)
    assert Diagram.discard(x @ x) == Discard(x) @ Discard(x)


def test_Substitution():
    """ https://github.com/discopy/discopy/issues/492 """
    X, Y = Ty("X"), Ty("Y")
    x, y = Variable("x", X), Variable("y", X)
    f, c = (X >> Y)("f"), Y("c")

    assert Substitution({x: y})(c) == c
    assert Substitution({x: y})(f(x)) == f(y)
    assert Substitution({x: y})(Application(f, x, left=True)).left
    assert Substitution({x: y})(Abstraction(x, f(x))) == Abstraction(x, f(x))
