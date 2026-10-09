from __future__ import annotations

from pytest import raises

from discopy import biclosed
from discopy.closed import *
from discopy.utils import dumps, loads


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


def test_product_str_round_trip():
    X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    for typ in (X.product(Y, Z), X.product()):
        assert eval(str(typ)) == typ
    assert eval(str(Product())) == Product()


def test_pack_to_hypergraph():
    X, Y = Ty("X"), Ty("Y")
    for box in (Pack(X * Y), Unpack(X * Y)):
        hypergraph = box.to_hypergraph()
        assert hypergraph.dom == box.dom and hypergraph.cod == box.cod
        assert hypergraph.to_diagram() == box


def test_pack_to_tree():
    X, Y = Ty("X"), Ty("Y")
    for box in (Pack(X * Y), Unpack(X * Y)):
        assert loads(dumps(box)) == box


def test_term_to_tree():
    """
    ``Tuple``, ``Projection`` and ``Let`` round-trip through their own
    ``to_tree``/``from_tree`` for terms with no ``Variable`` or ``Constant``
    leaf: those two inherit ``Box.from_tree`` from ``biclosed`` and its
    ``name``/``dom``/``cod`` kwargs don't match either of their
    constructors, a pre-existing gap this test does not cover, see
    https://github.com/discopy/discopy/pull/489#discussion_r3896298502.
    """
    empty = Tuple()
    assert loads(dumps(empty)) == empty
    nested = Tuple(Tuple(), Tuple())
    assert loads(dumps(nested)) == nested
    assert loads(dumps(Projection(nested, 0))) == Projection(nested, 0)
    let_term = Let(Tuple(), (), Tuple())
    assert loads(dumps(let_term)) == let_term


def test_strictification():
    X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    F = Functor({typ: typ @ typ for typ in (X, Y, Z)}, {})
    assert F((X * Y) * Z) == (X @ X * (Y @ Y)) * (Z @ Z)
    from discopy.python import Function, Ty as PyTy
    G = Functor({X: int, Y: bool, Z: float}, {}, cod=Function)
    assert G((X * Y) * Z) == PyTy(int, bool, float) == G(X.product(Y, Z))
    packed = G(Pack(X * Y))
    assert packed.dom == packed.cod == PyTy(int, bool)
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


def test_let_shadowing():
    X = Ty("X")
    x, y = Variable("x", X), Variable("y", X)
    f, g2 = (X >> X)("f"), ((X * X) >> X)("g2")
    term = Tuple(g2(Tuple(x, y)), let(f(y), lambda x: Tuple(x, y)))
    renamed = Tuple(g2(Tuple(x, y)), let(f(y), lambda x_: Tuple(x_, y)))
    assert term.eval() == renamed.eval()


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
    y_ = Variable("y_", X)
    assert Substitution({x: y})(t) == let(g(z), lambda y_: Tuple(y_, y))
    assert Substitution({x: y})(Abstraction(y, Tuple(y, x)))\
        == Abstraction(y_, Tuple(y_, y))


def test_substitution_bind_ignores_unused_replacements():
    X = Ty("X")
    x, y, z = (Variable(name, X) for name in "xyz")
    term = Abstraction(y, x)
    assert Substitution({z: y})(term) == term


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


def test_to_compact():
    w, x, y, z = map(Ty, "wxyz")
    f = Box("f", x @ y, z)

    for left in (True, False):
        source = f.curry(left=left)
        target = source.to_compact()
        expected = (f >> Coeval(source.cod, left=left)).trace(
            left=not left)
        assert target == expected
        assert (target.dom, target.cod) == (source.dom, source.cod)

    h = Box("h", w @ x @ y, z)
    for left in (True, False):
        source = h.curry(n=2, left=left)
        assert source.to_compact() == (
            h >> Coeval(source.cod, left=left)).trace(
                n=2, left=not left)

    g = Box("g", z << y, x)
    assert (f.curry() >> g).to_compact() == f.curry().to_compact() >>\
        g.to_compact()

    nested = f.curry().curry().to_compact().to_map()
    assert sum(isinstance(box, Coeval) for box in nested.boxes) == 2
    assert not any(isinstance(box, Curry) for box in nested.boxes)

    identity = x(lambda variable: variable)
    application = identity(x("a"))
    for term in (identity, application):
        result = term.to_compact().to_map()
        assert not any(isinstance(box, Curry) for box in result.boxes)
        assert term.to_map().to_compact() == result


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


def test_is_linear():
    """ Bubbles, sums and terms are linear when their insides are. """
    x, y = Ty("x"), Ty("y")
    f, nonlinear = Box("f", x, y), Copy(x) >> Box("g", x @ x, x)
    assert f.curry().is_linear and (f + f).is_linear
    assert (Box("h", x @ y, x @ y) @ x).trace().is_linear
    assert not nonlinear.is_linear
    assert not nonlinear.curry().is_linear
    assert not (nonlinear @ x).trace().is_linear
    assert not (nonlinear + nonlinear).is_linear

    g, a = (x >> (x >> y))("g"), x("a")
    assert g(a)(a).is_linear and x(lambda v: g(v)(a)).is_linear
    for term in [x(lambda v: g(v)(v)), x(lambda v: g(a)(a))]:
        assert not term.is_linear and not term.eval().is_linear


def test_eval_in_context():
    """
    A term evaluated in a context discards the variables it does not use
    and permutes the others, so that the diagram has the context as domain.
    """
    X, Y = Ty("X"), Ty("Y")
    x, y, f = Variable("x", X), Variable("y", Y), (X >> (Y >> Y))("f")
    assert x.eval(context=[y, x]) == Diagram.swap(Y, X) >> X @ Discard(Y)
    assert f.eval(context=[y]) == Discard(Y) >> f
    assert f(x)(y).eval(context=[y, x]) == Diagram.swap(Y, X)\
        >> f(x)(y).eval()
    for context in ([x, y], [y, x], [x, y, Variable("z", X)]):
        diagram = f(x)(y).eval(context=context)
        assert diagram.dom == Ty().tensor(*[v.cod for v in context])
        assert diagram.cod == Y


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


def test_compose():
    """ Terms of function types compose, the first one applied first. """
    X, Y, Z = map(Ty, "XYZ")
    t, u, v = (X >> Y)("t"), (Y >> Z)("u"), (Z >> X)("v")
    assert t.compose() == t and t.compose(u) == X(lambda x: u(t(x)))
    assert t.compose(u, v) == X(lambda x: v(u(t(x))))
    free = Variable("x", X >> Y)
    assert free.compose(u, v).freevars == [free]
    for left, right in [(u, t), (X("a"), t), (t, u)]:
        with raises(AxiomError):
            left.compose(right) if right is not u else left >> right
    with raises(AxiomError):
        t.compose(u, t)
