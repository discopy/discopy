from __future__ import annotations

from pytest import raises

from discopy.closed import *


U = Ty(Unitype())


def church(n, X):
    def body(f):
        def inner(x):
            result = x
            for _ in range(n):
                result = f(result)
            return result
        return X(inner)
    return (X >> X)(body)


def test_exp():
    X, Y = Ty('X'), Ty('Y')
    assert X >> Y == Y ** X == Y << X
    assert X @ Ty() == X == Ty() @ X


def test_unitype_hash():
    exp = Exp(U, U)
    assert Unitype() == exp and hash(Unitype()) == hash(exp)
    ob_map = {Unitype(): "unitype"}
    assert ob_map[exp] == "unitype"


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


def test_church_addition():
    X = Ty('X')
    zero, one, two, three = [church(n, X) for n in range(4)]
    N, I = two.cod, X >> X
    add = N(lambda m: N(lambda n: I(lambda f: X(lambda x: m(f)(n(f)(x))))))
    assert add(one)(two).normal_form() == three
    assert add(zero)(two).normal_form() == two == add(two)(zero).normal_form()


def test_church_multiplication():
    X = Ty('X')
    zero, two, three = church(0, X), church(2, X), church(3, X)
    N, I = two.cod, X >> X
    mul = N(lambda m: N(lambda n: I(lambda f: m(n(f)))))
    assert mul(two)(three).normal_form() == church(6, X)
    assert mul(two)(zero).normal_form() == zero


def test_church_exponentiation():
    assert U == U >> U and U.base == U == U.exponent
    two, three = church(2, U), church(3, U)
    N = two.cod
    assert N == U
    exp = N(lambda m: N(lambda n: (U >> U)(lambda f: n(m)(f))))
    assert exp(two)(three).normal_form() == church(8, U)
    assert exp(three)(two).normal_form() == church(9, U)


def test_normal_form_idempotent():
    two, three = church(2, U), church(3, U)
    exp = two.cod(lambda m: two.cod(lambda n: n(m)))
    result = exp(two)(three).normal_form()
    assert result == result.normal_form()
    x, y = Variable('x', U), Variable("x1", U)
    body = y
    for _ in range(8):
        body = x(body)
    assert result == Abstraction(x, Abstraction(y, body))


def test_reduce_budget():
    X = Ty('X')
    h = Variable('h', X >> (X >> X))
    c, d = Variable('c', X), Variable('d', X)
    identity = X(lambda x: x)
    term = h(identity(c))(identity(d))
    scope = (h, c, d)

    tree0 = term.reduce(budget=0)
    assert (tree0.head_cod, tree0.variables, tree0.head) == (X, scope, 0)
    assert tree0.spine == (identity(c), identity(d))
    with raises(ValueError):
        tree0[0]
    with raises(ValueError):
        tree0[1]
    with raises(ValueError):
        tree0.to_term()

    tree_negative = term.reduce(budget=-1)
    assert (tree_negative.head_cod, tree_negative.variables,
            tree_negative.head) == (X, scope, 0)
    with raises(ValueError):
        tree_negative[0]

    tree1 = term.reduce(budget=1)
    assert tree1[0].to_term(len(scope)) == c
    with raises(ValueError):
        tree1[1]
    with raises(ValueError):
        tree1.to_term()

    tree = term.reduce()
    assert tree[0].to_term(len(scope)) == c
    assert tree[1].to_term(len(scope)) == d
    assert tree.to_term(len(scope)) == h(c)(d)

    with raises(ValueError):
        term.normal_form(budget=1)
    assert term.normal_form(budget=2) == h(c)(d) == term.normal_form()


def test_reduce_strategy():
    X = Ty('X')
    h = Variable('h', X >> (X >> X))
    c, d = Variable('c', X), Variable('d', X)
    term = h(X(lambda x: x)(c))(X(lambda x: x)(d))
    scope = (h, c, d)

    tree = term.reduce(budget=1, strategy=RightmostFirst)
    assert tree.strategy.order(tree.spine) == (1, 0)
    with raises(ValueError):
        tree.to_term()
    assert tree[1].to_term(len(scope)) == d
    with raises(ValueError):
        tree[0]

    with raises(NotImplementedError):
        Strategy().order((c, ))
    with raises(TypeError):
        Constant('e', X).reduce()


def test_bohm_tree_cod():
    X = Ty('X')
    two = church(2, X)
    tree = two.reduce()
    assert tree.cod() == two.cod
    assert tree.to_term() == two


def test_bohm_tree_validation_and_equality():
    X, Y = Ty('X'), Ty('Y')
    x, y = Variable('x', X), Variable('y', Y)
    term = (X >> X)(lambda f: X(lambda x: f(x)))
    assert term.reduce() == term.reduce(budget=10)

    with raises(AxiomError):
        BohmTree(X, (x, ), 1, Strategy(), ())
    with raises(AxiomError):
        BohmTree(X, (x, ), 0, Strategy(), (x, ))
    with raises(AxiomError):
        BohmTree(X, (Variable('f', X >> X), ), 0, Strategy(), (y, ))
    with raises(AxiomError):
        BohmTree(Y, (x, ), 0, Strategy(), ())


def test_substitution():
    X = Ty('X')
    c, g, x = Constant('c', X), Variable('g', X >> X), Variable('x', X)
    assert Substitution({x: c})(g(x)) == g(c)
    assert Substitution({x: c})(g(c)) == g(c)
    h = Variable('h', X >> (X >> (X >> X)))
    f, y = Variable('f', X >> X), Variable('y', X)
    y_, y1 = Variable("y'", X), Variable("y1", X)
    term = X(lambda y: f(y))
    assert Substitution({f: h(y)(y_)})(term)\
        == Abstraction(y1, h(y)(y_)(y1))


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
