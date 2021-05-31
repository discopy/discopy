from pytest import raises
from discopy.hypergraph import *


def test_pushout():
    with raises(ValueError):
        pushout(1, 1, [0], [0, 1])


def test_Diagram_init():
    x, y = types('x y')
    with raises(ValueError):
        Diagram(x, x, [], [])
    with raises(AxiomError):
        Diagram(x, y, [], [0, 0])


def test_Diagram_str():
    x, y = types('x y')
    assert str(Swap(x, y)) == "Swap(x, y)"
    assert str(Spider(1, 0, x @ y))\
        == "Id(x) @ Spider(1, 0, y) >> Spider(1, 0, x)"


def test_Diagram_then():
    x, y = types('x y')
    with raises(AxiomError):
        Id(x) >> Id(y)


def test_Diagram_tensor():
    assert Id().tensor(Id(), Id()) == Id().tensor() == Id()


def test_Diagram_getitem():
    with raises(NotImplementedError):
        Spider(1, 2, Ty('x'))[0]


def test_Diagram_bijection():
    with raises(ValueError):
        Spider(1, 2, Ty('x')).bijection


def test_Box():
    box = Box('box', Ty('x'), Ty('y'))
    assert box == box and box == box @ Id() and box != 1


def test_AxiomError():
    x, y = types('x y')
    with raises(AxiomError):
        Cup(x @ y, x @ y)
    with raises(AxiomError):
        Cap(x @ y, x @ y)
