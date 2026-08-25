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
