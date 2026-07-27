from __future__ import annotations

from pytest import raises

from discopy.closed import *


def test_exp():
    X, Y = Ty('X'), Ty('Y')
    assert X >> Y == Y ** X == Y << X
    assert X @ Ty() == X == Ty() @ X


def test_product():
    X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    assert X * Y == Ty(Product(X, Y))
    assert (X * Y) * Z != X * (Y * Z) != X.product(Y, Z)
    assert (X * Y).is_product and (X * Y).factors == (X, Y)
    assert eval(str(X * Y)) == X * Y
    assert Pack(X * Y).dagger() == Unpack(X * Y)
    assert Unpack(X * Y).dagger() == Pack(X * Y)
    with raises(TypeError):
        Pack(X @ Y)
    with raises(TypeError):
        Unpack(X @ Y)


def test_strictification():
    X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    F = Functor({typ: typ @ typ for typ in (X, Y, Z)}, {})
    assert F((X * Y) * Z) == (X @ X * (Y @ Y)) * (Z @ Z)
    from discopy.python import Function
    G = Functor({X: int, Y: bool, Z: float}, {}, cod=Function)
    assert G((X * Y) * Z) == (int, bool, float) == G(X.product(Y, Z))
    packed = G(Pack(X * Y))
    assert packed.dom == packed.cod == (int, bool)
    assert packed(5, True) == (5, True)


def test_tuple():
    X, Y = Ty("X"), Ty("Y")
    x, y = Variable("x", X), Variable("y", Y)
    assert Tuple(x, y).cod == X * Y
    assert Tuple(x, Tuple(y, x)).cod == X * (Y * X)
    assert Tuple(x, y).eval() == Pack(X * Y)
    assert Tuple(x, x).eval() == Copy(X) >> Pack(X * X)


def test_projection():
    X, Y = Ty("X"), Ty("Y")
    x, y = Variable("x", X), Variable("y", Y)
    assert Projection(Tuple(x, y), 1).eval() == Discard(X) @ Y
    f = (X >> X * Y)("f")
    assert Projection(f(x), 1).eval()\
        == f @ X >> Diagram.ev(X * Y, X) >> Unpack(X * Y) >> Discard(X) @ Y
    with raises(TypeError):
        Projection(x, 0)
    with raises(IndexError):
        Projection(Tuple(x, y), 2)


def test_let():
    X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    f, g = (X >> Y)("f"), (Y >> Z)("g")
    x = Variable("x", X)
    t = let(f(x), lambda y: g(y))
    assert t.freevars == [x] and t.cod == Z
    assert t.eval() == f @ X >> Diagram.ev(Y, X) >> g @ Y >> Diagram.ev(Z, Y)
    assert Substitution({x: x})(t) == t

    both = let(Tuple(f(x), x), lambda y, z: Tuple(z, y))
    assert both.cod == X * Y and both.eval().dom == X

    with raises(ValueError):
        Let(f(x), (x, ), x)
    with raises(ValueError):
        Let(f(x), (Variable("y", Z), ), x)
    with raises(ValueError):
        let(f(x), lambda y, z: y)


def test_let_shared():
    X = Ty("X")
    x = Variable("x", X)
    effect = (X >> Ty())("effect")
    t = let(effect(x), lambda: x)
    assert t.cod == X and t.freevars == [x]
    assert t.eval().dom == X and t.eval().cod == X


def test_substitution():
    X, Y = Ty("X"), Ty("Y")
    f = (X >> Y)("f")
    x, z = Variable("x", X), Variable("z", X)
    s = Substitution({x: z})
    assert s(f) == f and s(x) == z
    assert s(f(x)) == f(z)
    assert s(Tuple(x, f(x))) == Tuple(z, f(z))
    assert s(Projection(Tuple(x, x), 0)) == Projection(Tuple(z, z), 0)
    assert s(Abstraction(x, f(x))) == Abstraction(x, f(x))
    assert s(let(f(x), lambda y: y)) == let(f(z), lambda y: y)


def test_compact_str():
    E = Ty("E")
    query, feed_forward = (E >> E)("query"), (E >> E)("feed_forward")
    x = Variable("x", E)
    t = let(query(x), lambda q: feed_forward(q))
    assert str(t) == "let(query(x), lambda q: feed_forward(q))"
    assert eval(str(t), dict(
        let=let, query=query, feed_forward=feed_forward, x=x)) == t


def test_to_term():
    X, Y = Ty("X"), Ty("Y")
    f, g = Box("f", X, Y @ Y), Box("g", Y @ Y, Y)
    diagram = Diagram.copy(X) >> f @ Diagram.discard(X) >> g
    t = diagram.to_term()
    assert str(t) == "let(f(x0), lambda x1, x2: g(Tuple(x1, x2)))"
    x0 = Variable("x0", X)
    assert eval(str(t), dict(
        let=let, Tuple=Tuple, x0=x0,
        f=(X >> Y @ Y)("f"), g=(Y.product(Y) >> Y)("g"))) == t

    assert Copy(X).to_term() == Tuple(x0, x0)
    assert Box("h", X, Y).to_term() == (X >> Y)("h")(x0)


def test_to_term_round_trip():
    from discopy.python import Function
    X, Y = Ty("X"), Ty("Y")
    f, g = Box("f", X, Y @ Y), Box("g", Y @ Y, Y)
    diagram = Diagram.copy(X) >> f @ Diagram.discard(X) >> g
    term = diagram.to_term()
    F = Functor({X: int, Y: int}, {
        f: Function(lambda n: (n, n + 1), (int,), (int, int)),
        g: Function(lambda a, b: a * b, (int, int), (int,))}, cod=Function)
    constant_f, constant_g = term.constants
    G = Functor({X: int, Y: int}, {
        constant_f: Function(
            lambda: lambda n: (n, n + 1), (),
            Function.exp((int, int), (int,))),
        constant_g: Function(
            lambda: lambda a, b: a * b, (),
            Function.exp((int,), (int, int)))}, cod=Function)
    assert F(diagram)(3) == G(term)(3) == 12


def test_python_let():
    from discopy.python import Function
    x = Ty("x")
    f, g = (x >> x)("f"), (x.product(x) >> x)("g")
    v = Variable("v", x)
    t = let(Tuple(f(v), f(v)), lambda a, b: g(Tuple(a, b)))
    F = Functor({x: int}, {
        f: Function(lambda: lambda n: n + 1, (), Function.exp((int,), (int,))),
        g: Function(lambda: lambda a, b: a * b, (),
                    Function.exp((int,), (int, int)))}, cod=Function)
    assert F(t)(3) == 16


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
