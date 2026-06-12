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
    f, g, a = Constant(x << y, 'f'), Constant(y >> x, 'g'), Constant(y, 'a')

    assert isinstance(f, TermBase)
    assert (f << a).cod == x
    assert (f << a).to_diagram()\
        == Box('f', Ty(), x << y) @ Box('a', Ty(), y)\
        >> Eval(x << y, left=True)
    assert (a >> g).cod == x
    assert (a >> g).to_diagram()\
        == Box('a', Ty(), y) @ Box('g', Ty(), y >> x)\
        >> Eval(y >> x, left=False)

    var = Variable(y, 'var')
    assert Abstraction(var, f << var).cod == x << y
    assert Abstraction(var, var >> g, left=True).cod == y >> x


def test_Term_str():
    x, y = Ty('x'), Ty('y')
    f, g, a = Constant(x << y, 'f'), Constant(y >> x, 'g'), Constant(y, 'a')
    var = Variable(y, 'var')
    terms = [f << a, a >> g, y(lambda u: f << u),
             y(lambda u, left=True: u >> g), f << var]
    env = locals()
    assert all(eval(str(term), env) == term for term in terms)


def test_Term_str_constants():
    N, S = Ty("N"), Ty("S")
    Alice_loves_Bob = N("Alice") >> (((N >> S) << N)("loves") << N("Bob"))
    assert str(Alice_loves_Bob) == (
        "N('Alice') >> (((N >> S) << N)('loves') << N('Bob'))")


def test_Term_linear_planar():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Constant(x << y, 'f'), Constant(y >> x, 'g')
    fvar = Variable(x << y, 'fvar')
    gvar = Variable(y >> x, 'gvar')
    h = Constant((x << y) << y, 'h')
    var = Variable(y, 'var')

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
