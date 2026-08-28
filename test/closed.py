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
        # term.to_map() is the lambda-term encoding: the diagram map is
        # explicit
        assert CMap.from_diagram(term).to_compact() == result


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
    F = Functor(ob_map={X: int}, ar_map={}, cod=Function)
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

def church(n, o=Unitype()):
    def body(f):
        def inner(x):
            result = x
            for _ in range(n):
                result = f(result)
            return result
        return o(inner)
    return o(body)

def test_unitype():
    o = Unitype()
    assert o == Ty("o") and o >> o == o == o << o == o ** o
    assert o.is_exp and o.base == o.exponent == o
    assert (o >> Ty("X")) != o  # ordinary exponentials still work
    assert church(2).cod == o and church(2)(church(2)).cod == o

def test_bohm_tree_church_arithmetic():
    o = Unitype()
    add = o(lambda m: o(lambda n: o(lambda f: o(lambda x:
        m(f)(n(f)(x))))))
    mult = o(lambda m: o(lambda n: o(lambda f: m(n(f)))))
    exponent = o(lambda m: o(lambda n: n(m)))

    def tree(term):
        return BohmTree.from_term(term)

    assert tree(add(church(2))(church(3))) == tree(church(5))
    assert hash(tree(church(5))) == hash(tree(add(church(2))(church(3))))
    assert tree(mult(church(2))(church(3))) == tree(church(6))
    assert tree(exponent(church(2))(church(3))) == tree(church(8))
    assert tree(church(0)) == tree(add(church(0))(church(0)))
    assert tree(mult(church(2))(church(0))) == tree(church(0))

def test_bohm_tree_idempotent():
    o = Unitype()
    mult = o(lambda m: o(lambda n: o(lambda f: m(n(f)))))
    for term in [church(0), church(3), mult(church(2))(church(2))]:
        tree = BohmTree.from_term(term)
        assert BohmTree.from_term(tree.to_term()) == tree

def test_bohm_tree_budget():
    o = Unitype()
    term = church(2)(church(2))  # 2 ** 2, needs several beta steps
    assert BohmTree.from_term(term, budget=0) is None
    complete = BohmTree.from_term(term)
    partial = BohmTree.from_term(term, budget=4)
    assert partial is not None and partial != complete
    assert None in partial.args or any(
        arg and None in arg.args for arg in partial.args)
    with raises(ValueError):
        partial.to_term()
    assert BohmTree.from_term(term, budget=100) == complete

def test_bohm_tree_names_and_scope():
    o = Unitype()
    identity = o(lambda u: u)
    tree = BohmTree.from_term(identity)
    assert tree.variables[0].name == "u"  # names are preserved
    assert str(tree.to_term()) == str(identity)
    free = Variable("z", o)
    tree = BohmTree.from_term(identity(free), scope=(free, ))
    assert tree.head == 0 and tree.to_term(scope=(free, )) == free

def test_bohm_tree_constant_head():
    o = Unitype()
    with raises(NotImplementedError):
        BohmTree.from_term(o("a"))

def test_substitution_capture():
    o = Unitype()
    u, v = Variable("u", o), Variable("v", o)
    renamed = Substitution({u: v})(Abstraction(v, u(v)))
    assert renamed.var not in (u, v)
    assert renamed.body == v(renamed.var)
