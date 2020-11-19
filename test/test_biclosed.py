from pytest import raises
from discopy.biclosed import *


def test_Over():
    x, y = Ty('x'), Ty('y')
    assert repr(Over(x, y)) == "Over(Ty('x'), Ty('y'))"
    assert {Over(x, y): 42}[Over(x, y)] == 42
    assert Over(x, y) != Under(x, y)


def test_Under():
    x, y = Ty('x'), Ty('y')
    assert repr(Under(x, y)) == "Under(Ty('x'), Ty('y'))"
    assert {Under(x, y): 42}[Under(x, y)] == 42
    assert Under(x, y) != Over(x, y)


def test_Diagram():
    x, y = Ty('x'), Ty('y')
    assert Diagram.id(x) == Id(x)
    assert Diagram.ba(x, x >> y) == BA(x >> y)
    assert Diagram.fa(x << y, y) == FA(x << y)
    with raises(AxiomError):
        Diagram.ba(x, y >> x)
    with raises(AxiomError):
        Diagram.fa(y << x, y)

def test_BA():
    x, y = Ty('x'), Ty('y')
    with raises(TypeError):
        BA(x << y)
    assert repr(BA(x >> y)) == "BA(Under(Ty('x'), Ty('y')))"


def test_FA():
    x, y = Ty('x'), Ty('y')
    with raises(TypeError):
        FA(x >> y)
    assert repr(FA(x << y)) == "FA(Over(Ty('x'), Ty('y')))"


def test_FC():
    x, y = Ty('x'), Ty('y')
    with raises(TypeError):
        FC(x >> y, y >> x)
    with raises(TypeError):
        FC(x << y, y >> x)


def test_Functor():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    IdF = Functor(lambda x: x, lambda f: f)
    assert IdF(x >> y << x) == x >> y << x
    assert IdF(Curry(f)) == Curry(f)
    assert IdF(FC(x << y, y << x)) == FC(x << y, y << x)


def test_biclosed2rigid():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    diagram = Id(x << y) @ f >> FA(x << y)
    assert biclosed2rigid(x) == rigid.Ty('x')
    x_, y_ = rigid.Ty('x'), rigid.Ty('y')
    f_ = rigid.Box('f', x_, y_)
    assert biclosed2rigid(diagram)\
        == rigid.Id(x_ @ y_.l) @ f_ >> rigid.Id(x_) @ rigid.Cup(y_.l, y_)
    assert biclosed2rigid(Curry(BA(x >> y))).normal_form()\
        == rigid.Cap(y_, y_.l) @ rigid.Id(x_)
    assert biclosed2rigid(Curry(FA(x << y), left=True)).normal_form()\
        == rigid.Id(y_) @ rigid.Cap(x_.r, x_)
    assert biclosed2rigid(FC(x << y, y << x))\
        == rigid.Id(x_) @ rigid.Cup(y_.l, y_) @ rigid.Id(x_.l)
