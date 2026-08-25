from discopy.biclosed import *
from discopy import cat
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
    f = (x << y)("f")
    fvar = Variable('fvar', x << y)
    gvar = Variable('gvar', y >> x)
    h = ((x << y) << y)("h")
    var = Variable('var', y)

    with raises(ValueError):
        h(var)(var)
    with raises(ValueError):
        z(lambda u, left=True: f(var))
    with raises(ValueError):
        Abstraction(var, var(gvar, left=True))
    with raises(ValueError):
        Abstraction(var, fvar(var), left=True)

    eta_over = Abstraction(var, fvar(var))
    eta_under = Abstraction(var, var(gvar, left=True), left=True)
    assert (eta_over.dom, eta_over.cod) == (x << y, x << y)
    assert (eta_under.dom, eta_under.cod) == (y >> x, y >> x)


def test_Ty_call_errors():
    x = Ty('x')
    with raises(NotImplementedError):
        x(lambda u, left=1: u)
    with raises(NotImplementedError):
        x(lambda u, v: u)
    with raises(ValueError):
        x(42)


def test_Term_context_order_and_composite_abstraction():
    X, Y, Z = map(Ty, "XYZ")
    x = Variable("x", X)
    f, g = Variable("f", X >> Y), Variable("g", Y << X)

    assert g(x).freevars == [g, x]
    assert g(x).dom == (Y << X) @ X
    assert x(f, left=True).freevars == [x, f]
    assert x(f, left=True).dom == X @ (X >> Y)

    XY, var = X @ Y, Variable("var", X @ Y)
    over, under = Variable("over", Z << XY), Variable("under", XY >> Z)
    abstractions = [
        Abstraction(var, over(var)),
        Abstraction(var, var(under, left=True), left=True)]
    assert [term.dom for term in abstractions] == [over.cod, under.cod]
    assert abstractions[0].variables == [var, over]
    for term in abstractions:
        assert (term.eval().dom, term.eval().cod) == (term.dom, term.cod)


def test_term_Functor():
    x, y = Ty('x'), Ty('y')
    f, a = (y << x)("f"), x("a")
    g, b = (y << x)("g"), x("b")
    var = Variable("v", x)
    F = Functor(ob_map={x: x, y: y}, ar_map={f: g, a: b})

    assert F(f(a)) == g(b)
    assert F(var) == var
    assert F(Abstraction(var, f(var))) == Abstraction(var, g(var))

    h, c = (x >> y)("h"), (x >> y)("k")
    G = Functor(ob_map={x: x, y: y}, ar_map={h: c, a: b})
    assert G(a(h, left=True)) == b(c, left=True)


def test_Abstraction_eval():
    x, y = Ty('x'), Ty('y')
    f, g = (y << x)("f"), (x >> y)("g")
    assert x(lambda v: f(v)).eval()\
        == (f @ Id(x) >> Diagram.ev(y, x, left=True)).curry(left=True)
    assert x(lambda v, left=True: v(g, left=True)).eval()\
        == (Id(x) @ g >> Diagram.ev(y, x, left=False)).curry(left=False)


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
