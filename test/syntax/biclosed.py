from discopy.biclosed import *
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
    f, g = (x << y)("f"), (y >> x)("g")
    fvar = Variable('fvar', x << y)
    gvar = Variable('gvar', y >> x)
    h = ((x << y) << y)("h")
    var = Variable('var', y)

    with raises(ValueError):
        h(var)(var)
    with raises(ValueError):
        z(lambda u, left=True: f(var))
    with raises(ValueError):
        Abstraction(var, fvar(var))
    with raises(ValueError):
        Abstraction(var, var(gvar, left=True), left=True)


def test_InternalLanguage_errors():
    x = Ty('x')
    with raises(NotImplementedError):
        x(lambda u, left=1: u)
    with raises(NotImplementedError):
        x(lambda u, v: u)
    with raises(ValueError):
        x(42)


def test_term_Functor():
    x, y = Ty('x'), Ty('y')
    f, a = (y << x)("f"), x("a")
    g, b = (y << x)("g"), x("b")
    var = Variable("v", x)
    F = Functor(ob={x: x, y: y}, ar={f: g, a: b})

    assert F(f(a)) == g(b)
    assert F(var) == var
    assert F(Abstraction(var, f(var))) == Abstraction(var, g(var))

    h, c = (x >> y)("h"), (x >> y)("k")
    G = Functor(ob={x: x, y: y}, ar={h: c, a: b})
    assert G(a(h, left=True)) == b(c, left=True)


def test_Abstraction_eval():
    x, y = Ty('x'), Ty('y')
    f, g = (y << x)("f"), (x >> y)("g")
    assert x(lambda v: f(v)).eval()\
        == (f @ Id(x) >> Diagram.ev(y, x, left=True)).curry(left=True)
    assert x(lambda v, left=True: v(g, left=True)).eval()\
        == (Id(x) @ g >> Diagram.ev(y, x, left=False)).curry(left=False)


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
