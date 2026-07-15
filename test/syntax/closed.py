from __future__ import annotations

from pytest import raises

from discopy.closed import *
from discopy.closed import (
    Copy, Discard, Product, Projection, LetStatement, Substitution, let)


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


def test_discard():
    X = Ty("X")
    assert Copy(X, 0) == Discard(X) == Diagram.discard(X)
    assert Box("f", X, X).is_linear and not Diagram.copy(X).is_linear


def test_Product():
    X, Y = Ty("X"), Ty("Y")
    x, y = Variable("x", X), Variable("y", Y)
    assert Product(x, y).dom == X @ Y == Product(x, y).cod
    assert Product(x, y).eval() == Id(X @ Y)
    assert Product().cod == Ty() and Product().eval() == Id(Ty())
    assert Product(x, x).eval() == Diagram.copy(X)
    assert str(Product(x, y)) == "(x, y)"
    assert repr(Product(x)) == f"closed.Product({x!r})"
    with raises(TypeError):
        Product(x, 42)


def test_Projection():
    X, Y = Ty("X"), Ty("Y")
    x, y = Variable("x", X), Variable("y", Y)
    first, second = [Projection(Product(x, y), i) for i in (0, 1)]
    assert (first.cod, second.cod) == (X, Y)
    assert first.eval() == X @ Diagram.discard(Y)
    assert second.eval() == Diagram.discard(X) @ Y
    assert str(second) == "(x, y)[1]"
    assert repr(second) == f"closed.Projection({Product(x, y)!r}, 1)"
    assert second.constants == []
    with raises(IndexError):
        Projection(Product(x, y), 2)
    with raises(TypeError):
        Projection(x, "0")


def test_LetStatement():
    X, Y = Ty("X"), Ty("Y")
    f, x = Constant("f", X >> Y), Variable("x", X)
    once = let(f(x), lambda y: Product(y, y))
    assert once.dom == X and once.cod == Y @ Y
    assert once.constants == [f]
    assert str(once) == "let((X >> Y)('f')(x), lambda y: (y, y))"
    assert once == eval(repr(once), {
        "closed": __import__("discopy").closed, "cat": __import__(
            "discopy").cat})
    with raises(ValueError):
        LetStatement(f(x), (Variable("y", X), ), x)
    with raises(ValueError):
        let(f(x), lambda y, z: y)
    with raises(TypeError):
        LetStatement(f(x), (42, ), x)


def test_let_eval():
    from discopy import python
    X, Y = Ty("X"), Ty("Y")
    f, x = Constant("f", X >> Y), Variable("x", X)
    once = let(f(x), lambda y: Product(y, y))
    twice = Product(f(x), f(x))
    ob = {X: int, Y: int}
    const = python.Function(
        lambda: (lambda n: n + 1), (),
        Functor(ob, {}, cod=python.Function)(f.cod))
    F = Functor(ob, {f: const}, cod=python.Function)
    assert F(once)(41) == F(twice)(41) == (42, 42)
    assert F(Projection(twice, 1))(41) == 42


def test_snake_let():
    X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    state = Constant("state", Ty() >> X @ Y)
    effect = Constant("effect", Y @ Z >> Ty())
    snake = Z(lambda z: let(
        state(), lambda x, y: let(effect(y, z), lambda: x)))
    assert snake.cod == Z >> X
    assert snake.constants == [state, effect]
    diagram = snake.eval()
    assert diagram.dom == Ty() and diagram.cod == Z >> X
    curry, = diagram.boxes
    assert state in curry.arg.boxes and effect in curry.arg.boxes
    snake.eval().to_drawing()


def test_application_overlap():
    X, Y = Ty("X"), Ty("Y")
    g, x = Variable("g", X >> (X >> Y)), Variable("x", X)
    term = g(x)(x)
    diagram = term.eval()
    assert diagram.dom == (X >> (X >> Y)) @ X and diagram.cod == Y


def test_abstraction_in_let():
    X, Y = Ty("X"), Ty("Y")
    f, x = Constant("f", X >> Y), Variable("x", X)
    term = let(f(x), lambda y: Y(lambda w: Product(w, y)))
    assert term.cod == (Y @ Y) << Y
    assert term.eval().dom == X and term.eval().cod == (Y @ Y) << Y


def test_Substitution():
    X, Y = Ty("X"), Ty("Y")
    f = Constant("f", X >> Y)
    x, x_ = Variable("x", X), Variable("x_", X)
    s = Substitution({x: x_})
    assert s(x) == x_ and s(f) == f
    assert s(f(x)) == f(x_)
    assert s(Product(x, x)) == Product(x_, x_)
    assert s(Projection(Product(x, x), 0)) == Projection(Product(x_, x_), 0)
    term = let(f(x), lambda y: Product(y, x))
    y, = term.variables
    assert s(term) == LetStatement(f(x_), (y, ), Product(y, x_))
    abstraction = X(lambda x: f(x))
    assert s(abstraction) == abstraction  # x is bound, not substituted
