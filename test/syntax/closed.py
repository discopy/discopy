from __future__ import annotations

from pytest import raises

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
        ob_map={x: complex, y: bool, z: float},
        ar_map={f: lambda y: lambda x: abs(x) ** 2 if y else 0,
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

    g, y = (X >> (X >> Y))("g"), Variable("y", X)
    term = X(lambda x: X(lambda y: g(x)(y)))(y)
    var = Variable("y_", X)
    assert term.normal_form() == Abstraction(var, g(y)(var))

    a = Variable("a", X)
    assert X(lambda x: f(x))(a).normal_form() == f(a)
    with raises(ValueError, match="free-variable context"):
        X(lambda x: Y("c"))(a).normal_form()

    h, x, y = (X >> (X >> Y))("h"), Variable("x", X), Variable("y", X)
    exchange = Abstraction(x, h(x)(y))(a)
    assert exchange.freevars == [y, a] and exchange.dom == X @ X
    with raises(ValueError, match="free-variable context"):
        exchange.normal_form()

    duplicate = X(lambda x: h(x)(x))(X("a"))
    with raises(ValueError, match="duplicate an argument"):
        duplicate.normal_form()


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
    with raises(TypeError):
        Substitution({a: v})
    with raises(ValueError):
        Substitution({v: Y("b")})


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


def test_eval_with_context_and_composite_binders():
    X, Y, Z = map(Ty, "XYZ")
    x, y = Variable("x", X), Variable("y", Y)
    g, h = (Y >> X)("g"), (Y >> (X >> Z))("h")

    for left in [False, True]:
        abstraction = Abstraction(x, h(y)(x), left=left)
        argument = g(y)
        application = argument(abstraction, left=True)\
            if left else abstraction(argument)
        assert application.overlap
        assert (application.eval().dom, application.eval().cod)\
            == (application.dom, application.cod)

    XY, var = X @ Y, Variable("var", X @ Y)
    for term in [
            Abstraction(var, (XY >> Z)("f")(var)),
            Abstraction(var, Z("z"))]:
        assert (term.eval().dom, term.eval().cod) == (term.dom, term.cod)

    nested = X(lambda x: (X >> Z)(lambda f: f(x)))
    diagram, drawing = nested.eval(), nested.eval().to_drawing()
    assert (drawing.dom, drawing.cod)\
        == (diagram.dom.to_drawing(), diagram.cod.to_drawing())


def test_Application_context_order_is_stable():
    X, Y, Z, A, B = map(Ty, "XYZAB")
    x, y, z = Variable("x", X), Variable("y", Y), Variable("z", Z)
    func = (X >> (Y >> (A >> B)))("f")(x)(y)
    args = (Y >> (Z >> A))("a")(y)(z)
    term = func(args)

    assert term.overlap
    assert term.freevars == [x, y, z]
    assert term.dom == X @ Y @ Z
    assert (term.eval().dom, term.eval().cod) == (term.dom, term.cod)
