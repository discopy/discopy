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
