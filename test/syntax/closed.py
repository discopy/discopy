from __future__ import annotations

from pytest import raises

from discopy import monoidal
from discopy.closed import *


class Unitype(Exp):
    "The unitype is its own exponential, i.e. ``U == U >> U``."
    def __init__(self):
        monoidal.Wire.__init__(self, "U")
        self.base = self.exponent = Ty(self)

    def __eq__(self, other):
        return isinstance(other, Unitype) or isinstance(other, Exp)\
            and (other.base, other.exponent) == (self.base, self.exponent)

    def __hash__(self):
        return hash("U")

    def __str__(self):
        return "U"

    def __repr__(self):
        return "Unitype()"


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
    x, y = Variable('x', U), Variable("x'", U)
    body = y
    for _ in range(8):
        body = x(body)
    assert result == Abstraction(x, Abstraction(y, body))


def test_reduce_budget():
    X = Ty('X')
    h = Variable('h', X >> (X >> X))
    c, d = Variable('c', X), Variable('d', X)
    term = h(X(lambda x: x)(c))(X(lambda x: x)(d))
    scope = (h, c, d)
    leaf_c, leaf_d = [BohmTree(X, scope, i, ()) for i in (1, 2)]
    assert term.reduce(budget=0) == BohmTree(X, scope, 0, (None, None))
    assert term.reduce(budget=1) == BohmTree(X, scope, 0, (leaf_c, None))
    assert term.reduce() == BohmTree(X, scope, 0, (leaf_c, leaf_d))
    with raises(ValueError):
        term.reduce(budget=1).to_term()
    with raises(ValueError):
        term.normal_form(budget=1)
    assert term.normal_form(budget=2) == h(c)(d) == term.normal_form()


def test_reduce_strategy():
    X = Ty('X')
    h = Variable('h', X >> (X >> X))
    c, d = Variable('c', X), Variable('d', X)
    term = h(X(lambda x: x)(c))(X(lambda x: x)(d))
    scope = (h, c, d)

    class RightmostFirst(LeftmostOutermost):
        def arguments(self, terms, variables):
            return tuple(reversed(
                [self(term, variables) for term in reversed(terms)]))

    assert term.reduce(budget=1, strategy=RightmostFirst)\
        == BohmTree(X, scope, 0, (None, BohmTree(X, scope, 2, ())))
    with raises(NotImplementedError):
        Strategy()(c, scope)
    with raises(TypeError):
        Constant('e', X).reduce()


def test_bohm_tree_ty():
    X = Ty('X')
    two = church(2, X)
    tree = two.reduce()
    assert tree.ty() == two.cod
    assert tree.to_term() == two


def test_substitution():
    X = Ty('X')
    c, g, x = Constant('c', X), Variable('g', X >> X), Variable('x', X)
    assert Substitution({x: c})(g(x)) == g(c)
    assert Substitution({x: c})(g(c)) == g(c)
    h = Variable('h', X >> (X >> (X >> X)))
    f, y = Variable('f', X >> X), Variable('y', X)
    y_, y__ = Variable("y'", X), Variable("y''", X)
    term = X(lambda y: f(y))
    assert Substitution({f: h(y)(y_)})(term)\
        == Abstraction(y__, h(y)(y_)(y__))
