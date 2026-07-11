from __future__ import annotations

from discopy import biclosed
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
        ob={x: complex, y: bool, z: float},
        ar={f: lambda y: lambda x: abs(x) ** 2 if y else 0,
            g: lambda x, y: abs(x + 1j if y else -1j)},
        cod=Function)

    assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
    assert F(g.curry().uncurry())(1j, True) == F(g)(1j, True)


def test_from_biclosed():
    x, y = biclosed.Ty("x"), biclosed.Ty("y")
    X, Y = Ty("x"), Ty("y")
    assert Ty.from_biclosed(x << y) == Ty.from_biclosed(y >> x) == Y >> X
    assert Ty.from_biclosed(x @ (x >> y)) == X @ (X >> Y)

    g, a, h = (y << x)("g"), x("a"), (x >> y)("h")
    assert g(a).to_closed() == TermBase.from_biclosed(g(a))\
        == Constant("g", X >> Y)(Constant("a", X))
    assert a(h, left=True).to_closed()\
        == Constant("h", X >> Y)(Constant("a", X))
    assert TermBase.from_biclosed(x(lambda v: g(v)))\
        == X(lambda v: Constant("g", X >> Y)(v))


def test_normal_form():
    X, Y = Ty("X"), Ty("Y")
    f, a = (X >> Y)("f"), X("a")
    var = Variable("v", X)
    assert f.normal_form() == f
    assert a.normal_form() == a
    assert var.normal_form() == var
    assert X(lambda z: f(z))(a).normal_form() == f(a)
    assert X(lambda z: X(lambda w: f(w))(z)).normal_form()\
        == X(lambda z: f(z))


def test_Substitution():
    X, Y = Ty("X"), Ty("Y")
    f, a = (X >> Y)("f"), X("a")
    v, w = Variable("v", X), Variable("w", X)
    sub = Substitution({v: a})
    assert sub(f) == f
    assert sub(v) == a and sub(w) == w
    assert sub(f(v)) == f(a)
    assert sub(Abstraction(v, f(v))) == Abstraction(v, f(v))
    assert sub(Abstraction(w, f(v))) == Abstraction(w, f(a))


def test_discard_and_nonlinear_eval():
    x, y = Ty("x"), Ty("y")
    assert Diagram.discard(x) == Copy(x, 0) == Discard(x)
    assert not Copy(x).is_linear

    g = (x >> (x >> y))("g")
    term = x(lambda v: g(v)(v))
    diagram = term.eval()
    assert diagram.dom == Ty() and diagram.cod == x >> y
    assert not diagram.arg.is_linear

    shared_abstraction = x(lambda v: x(lambda w: g(w)(v))(v))
    diagram = shared_abstraction.eval()
    assert diagram.dom == Ty() and diagram.cod == x >> y
