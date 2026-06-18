from discopy.biclosed import *
from pytest import raises


def test_Over():
    x, y = Ty('x'), Ty('y')
    assert repr(Over(x, y))\
        == "biclosed.Over(biclosed.Ty(cat.Ob('x')), biclosed.Ty(cat.Ob('y')))"
    assert {Over(x, y): 42}[Over(x, y)] == 42
    assert Over(x, y) != Under(x, y)


def test_Under():
    x, y = Ty('x'), Ty('y')
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
    assert str(f(x)) == "f(x)"
    assert str(x(g, left=True)) == "x(g, left=True)"
    assert str(X(lambda y: f(y))) == "X(lambda y: f(y))"
    assert str(f(y)) == "f(y)"


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
