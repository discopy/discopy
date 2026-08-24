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
    assert Tuple().cod == Ty() and Tuple().eval() == Id(Ty())


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
    with raises(ValueError):
        let(f(x), lambda *ys: ys[0])
    with raises(ValueError):
        let(f(x), lambda **ys: x)


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


def test_substitution_capture():
    X = Ty("X")
    x, y, z = (Variable(name, X) for name in "xyz")
    g = (X >> X)("g")
    t = let(g(z), lambda y: Tuple(y, x))
    assert Substitution({x: z})(t) == let(g(z), lambda y: Tuple(y, z))
    with raises(ValueError):
        Substitution({x: y})(t)
    with raises(ValueError):
        Substitution({x: y})(Abstraction(y, Tuple(y, x)))


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
    effect = Box("effect", X, Ty())
    assert effect.to_term() == (X >> Ty())("effect")(x0)
    assert effect.to_term().cod == Ty()


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


def test_Application_without_freevars():
    """ A closed application of constants has an empty domain, see #542. """
    X, Y = Ty('X'), Ty('Y')
    f, x = (X >> Y)('f'), X('x')
    assert f(x).freevars == [] and f(x).dom == Ty() and f(x).cod == Y
    assert f(x).eval() == f.eval() @ x.eval() >> Diagram.ev(Y, X)


def test_Application_freevars_order():
    """ Free variables keep first-occurrence order rather than going through a
    set, whose iteration order depends on hashing, see #543. """
    A, B, C, W, Z = map(Ty, "ABCWZ")
    f, F = (A >> (B >> (C >> W)))('f'), (W >> (A >> Z))('F')
    body = F(f(A('a'))(B('b'))(C('c')))(A('a'))
    assert body.freevars == [] and body.dom == Ty()

    t = A(lambda a: B(lambda b: C(lambda c: F(f(a)(b)(c))(a))))
    inside = t.body.body.body
    assert [x.name for x in inside.freevars] == ['a', 'b', 'c']
    assert inside.dom == A @ B @ C
    assert t.cod == A >> (B >> (C >> Z))


def test_Abstraction_of_unused_variable():
    """ Abstracting a variable absent from the body discards it, see #541. """
    X, Y = Ty('X'), Ty('Y')
    h = (X >> Y)('h')
    t = X(lambda x: h)
    assert t.freevars == [] and t.cod == X >> (X >> Y)
    curry, = t.eval().boxes
    assert curry.arg == Discard(X) >> h.eval()


def test_Abstraction_eval_preserves_dom_and_cod():
    """ Nested abstractions used to curry the wrong wire, see #544. """
    A, B, C, Z = map(Ty, "ABCZ")
    g, h = (A >> (B >> Z))('g'), (A >> (B >> (C >> Z)))('h')
    gg = (A >> (A >> Z))('gg')
    for t in [A(lambda a: g(a)),
              A(lambda a: B(lambda b: g(a)(b))),
              A(lambda a: B(lambda b: C(lambda c: h(a)(b)(c)))),
              A(lambda a: g),
              A(lambda a: gg(a)(a))]:
        assert (t.dom, t.cod) == (t.eval().dom, t.eval().cod)


def test_Abstraction_eval_curries_the_right_wire():
    """
    Binders of the same type have the same `dom` and `cod` whichever wire
    is curried, so only the diagram tells them apart, see #544.
    """
    A, Z = Ty('A'), Ty('Z')
    g = (A >> (A >> Z))('g')
    swapped, straight = (A(lambda a: A(lambda b: g(a)(b))),
                         A(lambda a: A(lambda b: g(b)(a))))
    assert (swapped.dom, swapped.cod) == (straight.dom, straight.cod)
    assert swapped.eval() != straight.eval()


def test_nonlinear_eval():
    """ A repeated variable is copied, an unused one is discarded. """
    X, Y = Ty('X'), Ty('Y')
    g = (X >> (X >> Y))('g')
    copied, = X(lambda x: g(x)(x)).eval().boxes
    assert not copied.arg.is_linear
    assert any(isinstance(box, Copy) for box in copied.arg.boxes)

    discarded, = Y(lambda y: X(lambda x: g(x)(x))).eval().boxes
    assert any(isinstance(box, Discard) for box in discarded.arg.boxes)
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


def test_draw_copy_and_swap():
    """
    `closed.Diagram.to_drawing` routes through `closed.Functor` to get
    `Curry` and `Eval` right, which used to drag in the markov, symmetric
    and balanced branches calling `copy`, `merge`, `swap`, `braid` and
    `twist` on a `Drawing` that has none of them, see issues #491 and #548.

    Falling through draws them the way markov and symmetric diagrams are
    drawn today, so the closed drawing is the *same* drawing, not merely
    one that does not raise.
    """
    from discopy import markov, symmetric
    x, mx, sx = Ty('x'), markov.Ty('x'), symmetric.Ty('x')

    assert (Copy(x) >> Box('f', x @ x, x)).to_drawing()\
        == (markov.Copy(mx) >> markov.Box('f', mx @ mx, mx)).to_drawing()
    assert (Swap(x, x) >> Box('g', x @ x, x)).to_drawing()\
        == (symmetric.Swap(sx, sx)
            >> symmetric.Box('g', sx @ sx, sx)).to_drawing()
    assert (Copy(x) >> Swap(x, x) >> Box('h', x @ x, x)).to_drawing()\
        == (markov.Copy(mx) >> markov.Swap(mx, mx)
            >> markov.Box('h', mx @ mx, mx)).to_drawing()
    assert Diagram.discard(x).to_drawing()\
        == markov.Diagram.discard(mx).to_drawing()

    # A non-linear term evaluates to such a diagram, so it draws too.
    X = Ty('X')
    assert X(lambda x: (X >> X)(lambda f: f(x))).eval().to_drawing()
