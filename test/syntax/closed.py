from __future__ import annotations

from pytest import raises

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


def test_discard():
    X = Ty("X")
    assert isinstance(Copy(X, 0), Discard)
    assert Diagram.discard(X) == Discard(X)


def test_abstraction_eval_dom_cod():
    X, Y, Z, W = map(Ty, "XYZW")
    h = (X >> (Z >> Y))("h")
    k = (X >> (Z >> (W >> Y)))("k")
    x, w, v = Variable("x", X), Variable("w", Z), Variable("v", W)
    for term in [
            Abstraction(w, h(x)(w)),      # w is not first in freevars
            Abstraction(w, k(x)(w)(v))]:  # w is in the middle of freevars
        diagram = term.eval()
        assert (diagram.dom, diagram.cod) == (term.dom, term.cod)


def test_to_term_round_trip():
    X, Y, Z, W, T = map(Ty, "XYZWT")
    f, g = (X >> Y)("f"), (X >> (X >> Y))("g")
    h = (X >> (Z >> Y))("h")
    k = (X >> (Z >> (W >> Y)))("k")
    x, w, v = Variable("x", X), Variable("w", Z), Variable("v", W)
    for term in [
            X("c"),
            f(x),
            X(lambda a: f(a)),
            Abstraction(w, h(x)(w)),
            Abstraction(w, k(x)(w)(v)),
            X(lambda z: g(z)(z)),                          # copy
            g(x)(x),                                       # copy with free x
            X(lambda a: (X >> Y)(lambda fn: fn(a))),       # nested
            (T >> T)(lambda fn: T(lambda a: fn(fn(a)))),   # Church numeral
            ((X >> X)(lambda a: a))(X(lambda b: b)),       # beta redex
            X(lambda z: (X >> ((X >> Y) >> Y))("m")(z)(    # abstraction with
                X(lambda u: g(u)(z))))]:                   # shared context
        diagram = term.eval()
        assert (diagram.dom, diagram.cod) == (term.dom, term.cod)
        result = diagram.to_term()
        assert result == term and str(result) == str(term)


def test_to_term_fresh_names():
    X, Y = Ty("X"), Ty("Y")
    diagram = Curry(Eval(X >> Y, left=True))  # no varname attributes
    term = diagram.to_term()
    assert isinstance(term, Abstraction)
    assert term.eval() == diagram


def test_to_term_errors():
    X = Ty("X")
    with raises(ValueError):
        (Box("c", Ty(), X) >> Copy(X)).to_term()  # copy of a non-variable
    with raises(ValueError):
        Swap(X, X).to_term()  # two variables are not a single term


def test_context_image():
    from discopy.python import Function

    X = Ty("X")
    context = Context([Variable("z", X)])
    assert context.dom == X
    F = Functor(ob={X: int}, ar={}, cod=Function)
    assert context.image(F) == F(context.dom)


def test_substitution():
    X, Y = Ty("X"), Ty("Y")
    f = (X >> Y)("f")
    x, z = Variable("x", X), Variable("z", X)
    substitution = Substitution({x: X("c")})
    assert substitution(x) == X("c")
    assert substitution(f) == f
    assert substitution(f(x)) == f(X("c"))
    assert substitution(Abstraction(z, f(z))) == Abstraction(z, f(z))
    # a substitution does not cross a binder for the same variable
    assert substitution(Abstraction(x, f(x))) == Abstraction(x, f(x))
