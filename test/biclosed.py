import sys
from discopy.biclosed import *
from discopy import cat
from discopy.axioms import Related, assert_axioms
from pytest import raises


def test_Over():
    x, y = Ty('x'), Ty('y')
    assert isinstance(x.over_factory(x, y), cat.Ob)
    assert not isinstance(x.over_factory(x, y), Ty)
    assert x.over(y) == x << y
    assert isinstance(x ** y, Ty)
    assert not isinstance((x ** y).inside[0], Ty)
    assert isinstance(x << y, Ty)
    assert not isinstance((x << y).inside[0], Ty)
    assert (x << y).inside == (Over(x, y), )
    assert repr(Over(x, y))\
        == "biclosed.Over(biclosed.Ty(cat.Ob('x')), biclosed.Ty(cat.Ob('y')))"
    assert {Over(x, y): 42}[Over(x, y)] == 42
    assert Over(x, y) != Under(x, y)


def test_Under():
    x, y = Ty('x'), Ty('y')
    assert isinstance(y.under_factory(y, x), cat.Ob)
    assert not isinstance(y.under_factory(y, x), Ty)
    assert y.under(x) == x >> y
    assert isinstance(x >> y, Ty)
    assert not isinstance((x >> y).inside[0], Ty)
    assert (x >> y).inside == (Under(y, x), )
    assert repr(Under(x, y))\
        == "biclosed.Under(biclosed.Ty(cat.Ob('x')), biclosed.Ty(cat.Ob('y')))"
    assert {Under(x, y): 42}[Under(x, y)] == 42
    assert Under(x, y) != Over(x, y)


def test_Term():
    x, y = Ty('x'), Ty('y')
    assert isinstance((x << y).inside[0], cat.Ob)
    assert not isinstance((x << y).inside[0], Ty)
    f, g = (x << y)("f"), (y >> x)("g")
    a = y("a")

    assert isinstance(f, TermBase)
    assert f(a).cod == x
    assert f(a).eval() == f @ a >> Eval(x << y)
    assert a(g, left=True).cod == x
    assert a(g, left=True).eval() == a @ g >> Eval(y >> x)

    var = Variable('var', y)
    assert Abstraction(var, f(var)).cod == x << y
    assert Abstraction(var, var(g, left=True), left=True).cod == y >> x


def test_Variable_atomic():
    """
    A variable stands for exactly one wire, so a non-atomic codomain is
    rejected at construction rather than silently binding the last wire
    instead of the whole variable (regression test for #609).
    """
    x, y = Ty('x'), Ty('y')
    with raises(ValueError):
        Variable('v', x @ y)
    with raises(ValueError):
        Variable('v', Ty())
    assert Variable('v', x).cod == x


def test_Term_str():
    X, Y = Ty('X'), Ty('Y')
    f, g = (Y << X)("f"), (X >> Y)("g")
    x, y = X("x"), Variable("y", X)
    assert str(f(x)) == "(Y << X)('f')(X('x'))"
    assert str(x(g, left=True)) == "X('x')((X >> Y)('g'), left=True)"
    assert str(X(lambda y: f(y))) == "X(lambda y: (Y << X)('f')(y))"
    assert str(f(y)) == "(Y << X)('f')(y)"


def test_Term_linear_planar():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = (x << y)("f"), (y >> x)("g")
    fvar = Variable('fvar', x << y)
    gvar = Variable('gvar', y >> x)
    h = ((x << y) << y)("h")
    var = Variable('var', y)

    with raises(ValueError):
        h(var)(var)
    with raises(ValueError):
        z(lambda u, left=True: f(var))

    eta_over = Abstraction(var, fvar(var))
    eta_under = Abstraction(var, var(gvar, left=True), left=True)
    assert (eta_over.dom, eta_over.cod) == (x << y, x << y)
    assert (eta_under.dom, eta_under.cod) == (y >> x, y >> x)


def test_Application_freevars_order():
    """The free variables of a term are ordered like the wires of its dom."""
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    a, b = Variable('a', y), Variable('b', x)
    func = a((y >> (z << x))("h"), left=True)

    under = b((x >> (y >> z))("k"), left=True)
    for term in [func(b), a(under, left=True)]:
        assert term.dom == Ty().tensor(*[var.cod for var in term.freevars])
        assert term.freevars == [a, b]


def test_Abstraction_well_typed():
    """Abstraction agrees with eval on both dom and cod, nesting included."""
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    a, b = Variable('a', y), Variable('b', x)
    term = a((y >> (z << x))("h"), left=True)(b)

    inner = Abstraction(b, term)
    assert (inner.dom, inner.cod) == (y, z << x)
    outer = Abstraction(a, inner, left=True)
    assert (outer.dom, outer.cod) == (Ty(), y >> (z << x))
    for abstraction in [inner, outer]:
        evaluated = abstraction.eval()
        assert (evaluated.dom, evaluated.cod) == (abstraction.dom,
                                                  abstraction.cod)

    with raises(ValueError):
        Abstraction(a, term)


def test_to_rigid():
    from discopy import rigid

    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    diagram = Id(x << y) @ f >> Diagram.ev(x, y, left=True)
    assert Diagram.to_rigid(x) == rigid.Ty('x')
    x_, y_ = rigid.Ty('x'), rigid.Ty('y')
    f_ = rigid.Box('f', x_, y_)
    assert Diagram.to_rigid(diagram)\
        == rigid.Id(x_ @ y_.l) @ f_ >> rigid.Id(x_) @ rigid.Cup(y_.l, y_)


def test_to_compact():
    x, y, z = map(Ty, "xyz")
    f = Box("f", x @ y, z)
    for left in (True, False):
        source = f.curry(left=left)
        assert source.to_compact() == (
            f.to_map() >> CMap.ev(z, y if left else x, left).dagger()).trace(
                left=not left)
        assert source.to_map().to_compact() == source.to_compact()
        assert not any(isinstance(box, Curry)
                       for box in source.to_compact().boxes)


def test_alpha_eq():
    X, Y = Ty("X"), Ty("Y")
    f, g, h = (Y << X)("f"), (X >> Y)("g"), ((Y << X) << X)("h")
    x, y = Variable("x", X), Variable("y", X)
    assert X(lambda x: f(x)).alpha_eq(X(lambda y: f(y)), X(lambda z: f(z)))
    assert not X(lambda x: f(x)).alpha_eq(X(lambda y: f(y)), f(x))
    assert f(x).alpha_eq(f(x)) and not f(x).alpha_eq(f(y))
    assert X(lambda x, left=True: x(g, left=True)).alpha_eq(
        X(lambda y, left=True: y(g, left=True)))
    assert not X(lambda x: f(x)).alpha_eq(
        X(lambda x, left=True: x(g, left=True)))
    assert not X(lambda x: x).alpha_eq(Y(lambda y: y))
    assert X(lambda x: X(lambda y: h(x)(y))).alpha_eq(
        X(lambda y: X(lambda x: h(y)(x))))
    assert not X(lambda x: X(lambda y: h(x)(y))).alpha_eq(
        X(lambda x: X(lambda y: ((Y << X) << X)("h_")(x)(y))))


def test_alpha_eq_under_restores_the_scopes():
    X, Y = Ty("X"), Ty("Y")
    f, x = (Y << X)("f"), Variable("x", X)
    scopes = [{x: 0}, {}]
    assert X(lambda x: f(x)).alpha_eq_under(
        scopes, [X(lambda y: f(y))], depth=1)
    assert scopes == [{x: 0}, {}]


def test_alpha_completeness():
    X, Y = Ty("X"), Ty("Y")
    shape, other = (Y << X, [8, 0, 0], [X]), [8, 1, 0]
    build = Canonical[TermBase].build
    assert TermBase.alpha_completeness(build(shape, None, "yz"))
    assert TermBase.alpha_completeness(build(shape, other, "yz"))
    first, second, canonical, _ = build(shape, other, "yz")
    assert first != second and not TermBase.alpha_completeness(
        Canonical[TermBase]([first, second, canonical, canonical]))


def test_related():
    X, Y = Ty("X"), Ty("Y")
    shape = (Y << X, [8, 0, 0], [X])
    first, second, canonical, _ = Canonical[TermBase].build(
        shape, [8, 1, 0], "yz")
    terms = Related[TermBase]((first, canonical, second))
    assert TermBase.symmetry(terms) and TermBase.transitivity(terms)
    assert not TermBase.transitivity(
        Related[TermBase]((first, second, canonical)))
    assert TermBase.equivalent(first, canonical)
    assert not TermBase.equivalent(first, second)


def test_generate():
    X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    term = TermBase.generate(Y << X, [8, 0, 0], [X], "xy")
    assert term == X(lambda x0: (Y << X)("c0")(x0))
    assert term.alpha_eq(TermBase.generate(Y << X, [8, 0, 0], [X], "z"))
    spine = TermBase.generate((Z << Y) << X, [8, 8], [X], "x")
    assert spine == X(lambda x0: Y(lambda x1: ((Z << Y) << X)("c0")(x0)(x1)))
    assert spine.alpha_eq(TermBase.generate((Z << Y) << X, [8, 8], [X], "yz"))


def test_Sampler():
    X, Y = Ty("X"), Ty("Y")
    x0, x1 = Variable("x0", X), Variable("x1", Y)
    sampler = Sampler(TermBase, iter([]), [X], "x")
    assert sampler.spine(Y, (x0, x1)) == ((Y << Y) << X)("c0")(x0)(x1)
    assert sampler.variable(X) == Variable("v1", X)
    assert sampler.bound(1, Y) == x1 and sampler.choose("abc") == "a"
    assert [leaf() for leaf in sampler.leaves(X, (x0, ), True)] == [x0]
    assert sampler.leaves(Y, (x0, ), True) == []
    assert len(sampler.splits((x0, ), (x1, ), True)) == 5
    split = sampler.split(Y, ((), (), False), ((), (), True), False, 0)
    assert isinstance(split(), Application)
    assert Sampler(TermBase, iter([1]), [X], "x").term(Y) == Variable("v0", Y)


def test_deep_terms():
    X, limit = Ty("X"), sys.getrecursionlimit()
    sys.setrecursionlimit(100_000)
    try:
        term = TermBase.generate(X, [2, 0, 0] * 6000, [X], "x")
        assert term.alpha_eq(TermBase.generate(X, [2, 0, 0] * 6000, [X], "y"))
    finally:
        sys.setrecursionlimit(limit)
    depth = 0
    while isinstance(term, Application):
        depth, term = depth + 1, term.args
    assert depth == 6000


def test_axioms():
    assert_axioms(TermBase)

