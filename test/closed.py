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
        assert target == source.to_map().to_compact()
        assert not any(isinstance(box, Curry) for box in target.boxes)
        assert any(isinstance(box, Coeval) for box in target.boxes)
        assert (target.dom, target.cod) == (source.dom, source.cod)

    h = Box("h", w @ x @ y, z)
    for left in (True, False):
        source = h.curry(n=2, left=left)
        assert not any(
            isinstance(box, Curry) for box in source.to_compact().boxes)

    nested = f.curry().curry().to_compact()
    assert sum(isinstance(box, Coeval) for box in nested.boxes) == 2
    assert not any(isinstance(box, Curry) for box in nested.boxes)

    identity = x(lambda variable: variable)
    application = identity(x("a"))
    for term in (identity, application):
        result = term.to_compact()
        assert not any(isinstance(box, Curry) for box in result.boxes)


def test_Application_without_freevars():
    """ A closed application of constants has an empty domain, see #542. """
    X, Y = Ty('X'), Ty('Y')
    f, x = (X >> Y)('f'), X('x')
    assert f(x).freevars == [] and f(x).dom == Ty() and f(x).cod == Y
    assert f(x).eval() == f.eval() @ x.eval() >> Diagram.ev(Y, X)


def test_Application_freevars_order():
    """ Free variables keep first-occurrence order and closed terms are
    linear, see #543 and the first-order terms of `discopy.markov`. """
    A, B, C, W, Z = map(Ty, "ABCWZ")
    f, F = (A >> (B >> (C >> W)))('f'), (W >> (A >> Z))('F')
    body = F(f(A('a'))(B('b'))(C('c')))(A('a'))
    assert body.freevars == [] and body.dom == Ty()

    t = A(lambda a: B(lambda b: C(lambda c: f(a)(b)(c))))
    inside = t.body.body.body
    assert [x.name for x in inside.freevars] == ['a', 'b', 'c']
    assert inside.dom == A @ B @ C
    assert t.cod == A >> (B >> (C >> W))

    with raises(ValueError):
        A(lambda a: B(lambda b: C(lambda c: F(f(a)(b)(c))(a))))


def test_Abstraction_of_unused_variable():
    """ Closed terms are linear: abstracting a variable absent from the
    body raises, discarding is the business of `discopy.markov`. """
    X, Y = Ty('X'), Ty('Y')
    h = (X >> Y)('h')
    with raises(ValueError):
        X(lambda x: h)


def test_Abstraction_eval_preserves_dom_and_cod():
    """ Nested abstractions used to curry the wrong wire, see #544. """
    A, B, C, Z = map(Ty, "ABCZ")
    g, h = (A >> (B >> Z))('g'), (A >> (B >> (C >> Z)))('h')
    gg = (A >> (A >> Z))('gg')
    del gg
    for t in [A(lambda a: g(a)),
              A(lambda a: B(lambda b: g(a)(b))),
              A(lambda a: B(lambda b: C(lambda c: h(a)(b)(c))))]:
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


def test_linear_eval():
    """ A repeated variable raises, see `discopy.markov` for copies. """
    X, Y = Ty('X'), Ty('Y')
    g = (X >> (X >> Y))('g')
    with raises(ValueError):
        X(lambda x: g(x)(x))


def test_abstraction_eval_left():
    """ A left abstraction evaluates as its right counterpart WLOG. """
    X, Y = Ty("X"), Ty("Y")
    x, f = Variable('x', X), Variable('f', X >> Y)
    assert Abstraction(x, f(x), left=True).eval()\
        == Abstraction(x, f(x)).eval()


def test_draw_swap():
    """
    `closed.Diagram.to_drawing` routes through `closed.Functor` to get
    `Curry` and `Eval` right, which used to drag in the symmetric and
    balanced branches calling `swap`, `braid` and `twist` on a `Drawing`
    that has none of them, see issues #491 and #548.

    Falling through draws them the way symmetric diagrams are drawn today,
    so the closed drawing is the *same* drawing, not merely one that does
    not raise.
    """
    from discopy import symmetric
    x, sx = Ty('x'), symmetric.Ty('x')

    assert (Swap(x, x) >> Box('g', x @ x, x)).to_drawing()\
        == (symmetric.Swap(sx, sx)
            >> symmetric.Box('g', sx @ sx, sx)).to_drawing()

    # A term evaluates to such a diagram, so it draws too.
    X = Ty('X')
    assert X(lambda x: (X >> X)(lambda f: f(x))).eval().to_drawing()
