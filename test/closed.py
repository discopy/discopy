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


def test_abstraction_eval_context():
    """
    Both branches of `Abstraction.eval` curry on the right, so an
    abstraction applied to an argument sharing a free variable evaluates
    to a diagram with the type of the term (regression test for #562).
    """
    X, Y = Ty("X"), Ty("Y")
    x, f = Variable('x', X), Variable('f', X >> Y)
    g = Constant('g', X >> (X >> Y))
    t = Abstraction(x, Abstraction(f, f(x))(g(x)))
    assert t.eval().dom == t.dom and t.eval().cod == t.cod

    from discopy.python import Function
    F = Functor(ob_map={X: int, Y: str}, ar_map={}, cod=Function)
    F.ar_map[g] = Function(
        lambda: lambda n: lambda m: f"{n}|{m}", (), F(g.cod))
    assert F(t.eval())()(7) == "7|7"


def test_abstraction_eval_left():
    """ A left abstraction evaluates as its right counterpart WLOG. """
    X, Y = Ty("X"), Ty("Y")
    x, f = Variable('x', X), Variable('f', X >> Y)
    assert Abstraction(x, f(x), left=True).eval()\
        == Abstraction(x, f(x)).eval()


def test_Substitution():
    """ https://github.com/discopy/discopy/issues/492 """
    X, Y = Ty("X"), Ty("Y")
    x, y = Variable("x", X), Variable("y", X)
    f, c = (X >> Y)("f"), Y("c")

    assert Substitution({x: y})(c) == c
    assert Substitution({x: y})(f(x)) == f(y)
    assert Substitution({x: y})(Application(f, x, left=True))\
        == Application(f, y, left=True)
    assert Substitution({x: y})(Abstraction(x, f(x))) == Abstraction(x, f(x))
