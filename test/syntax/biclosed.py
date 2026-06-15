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
    f, g = Constant(Box('f', y, x), left=True), Constant(Box('g', y, x))
    a = Constant(Box('a', Ty(), y))

    assert isinstance(f, TermBase)
    assert (f << a).typ == x
    assert (f << a).eval()\
        == f.inside.curry(left=True) @ a.inside >> Eval(x << y, left=True)
    assert (a >> g).typ == x
    assert (a >> g).eval()\
        == a.inside @ g.inside.curry(left=False) >> Eval(y >> x, left=False)

    var = Variable('var', y)
    assert Abstraction(var, f << var).typ == x << y
    assert Abstraction(var, var >> g, left=True).typ == y >> x


def test_Term_str():
    X, Y = Ty('X'), Ty('Y')
    f, g = Box('f', X, Y), Box('g', X, Y)
    x, y = Box('x', Ty(), X), Variable("y", X)
    assert str(Constant(f, left=True) << Constant(x)) ==\
        "Constant(f, left=True) << Constant(x)"
    assert str(Constant(x) >> Constant(g)) == "Constant(x) >> Constant(g)"
    assert str(X(lambda y: Constant(f, left=True) << y)) ==\
        "X(lambda y: Constant(f, left=True) << y)"
    assert str(Constant(f, left=True) << y) == "Constant(f, left=True) << y"


def test_Term_linear_planar():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Constant(Box('f', y, x), left=True), Constant(Box('g', y, x))
    fvar = Variable('fvar', x << y)
    gvar = Variable('gvar', y >> x)
    h = Constant(Box('h', y, (x << y)), left=True)
    var = Variable('var', y)

    with raises(ValueError):
        h << var << var
    with raises(ValueError):
        z(lambda u, left=True: f << var)
    with raises(ValueError):
        Abstraction(var, var >> gvar)
    with raises(ValueError):
        Abstraction(var, fvar << var, left=True)


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
