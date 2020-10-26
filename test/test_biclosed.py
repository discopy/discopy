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
    assert Diagram.ba(x, x >> y) == BA(x, x >> y)
    assert Diagram.fa(x << y, y) == FA(x << y, y)


def test_BA():
    x, y = Ty('x'), Ty('y')
    with raises(AxiomError):
        BA(x, x << y)
    repr(BA(x, x >> y)) == "BA(Ty('y'), Over(Ty('x'), Ty('y')))"


def test_FA():
    x, y = Ty('x'), Ty('y')
    with raises(AxiomError):
        FA(x >> y, x)
    repr(FA(x << y, y)) == "FA(Under(Ty('x'), Ty('y')), Ty('y'))"
