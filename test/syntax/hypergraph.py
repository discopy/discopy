from pytest import raises

from discopy.hypergraph import *


def test_pushout():
    with raises(ValueError):
        pushout(1, 1, [0], [0, 1])


def test_Diagram_init():
    x, y = map(Ty, "xy")
    with raises(ValueError):
        Diagram(x, x, [], [])
    with raises(AxiomError):
        Diagram(x, y, [], [0, 0])


def test_Diagram_str():
    x, y = map(Ty, "xy")
    assert str(Swap(x, y)) == "Swap(x, y)"
    assert str(spiders(1, 0, x @ y))\
        == "x @ Spider(1, 0, y) >> Spider(1, 0, x)"


def test_Diagram_repr():
    x, y = map(Ty, "xy")
    assert repr(spiders(1, 0, x @ y))\
        == "hypergraph.Diagram("\
           "dom=frobenius.Ty(frobenius.Ob('x'), frobenius.Ob('y')), "\
           "cod=frobenius.Ty(), boxes=[], wires=[0, 1])"

def test_Diagram_then():
    x, y = map(Ty, "xy")
    with raises(AxiomError):
        Id(x) >> Id(y)


def test_Diagram_tensor():
    assert Id().tensor(Id(), Id()) == Id().tensor() == Id()


def test_Diagram_getitem():
    with raises(NotImplementedError):
        spiders(1, 2, Ty('x'))[0]


def test_Diagram_bijection():
    with raises(ValueError):
        spiders(1, 2, Ty('x')).bijection


def test_Box():
    box = Box('box', Ty('x'), Ty('y'))
    assert box == box and box == box @ Id() and box != 1


def test_AxiomError():
    x, y = map(Ty, "xy")
    with raises(AxiomError):
        cups(x @ y, x @ y)
    with raises(AxiomError):
        caps(x @ y, x @ y)


def test_cups():
    x = Ty('x')
    assert Diagram.cups(x, x).make_monogamous().dagger()\
        == Diagram.caps(x, x).make_monogamous()
    assert Diagram.caps(x, x).make_monogamous().dagger()\
        == Diagram.cups(x, x).make_monogamous()
    assert Cap(x, x).downgrade() == frobenius.Cap(x, x)
