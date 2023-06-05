from pytest import raises

from discopy.hypergraph import *
from discopy.frobenius import Ty, Box, Cap, Hypergraph as H

def test_pushout():
    with raises(ValueError):
        pushout(1, 1, [0], [0, 1])


def test_Hypergraph_init():
    x, y = map(Ty, "xy")
    with raises(ValueError):
        H(x, x, (), ())
    with raises(AxiomError):
        H(x, y, (), (0, 0))


def test_Hypergraph_str():
    x, y = map(Ty, "xy")
    assert str(H.swap(x, y)) == "Swap(x, y)"
    assert str(H.spiders(1, 0, x @ y))\
        == "Spider(1, 0, x) @ y >> Spider(1, 0, y)"


def test_Hypergraph_repr():
    x, y = map(Ty, "xy")
    assert repr(H.spiders(1, 0, x @ y))\
        == "frobenius.Hypergraph("\
           "dom=frobenius.Ty(frobenius.Ob('x'), frobenius.Ob('y')), "\
           "cod=frobenius.Ty(), boxes=(), wires=(0, 1))"


def test_Hypergraph_hash():
    x, y = map(Ty, "xy")
    assert hash(H.id(x @ y)) == hash(H.id(x) @ H.id(y))


def test_Hypergraph_then():
    x, y = map(Ty, "xy")
    with raises(AxiomError):
        H.id(x) >> H.id(y)


def test_Hypergraph_tensor():
    Id = H.id
    assert Id().tensor(Id(), Id()) == Id().tensor() == Id()


def test_Hypergraph_getitem():
    with raises(NotImplementedError):
        H.spiders(1, 2, Ty('x'))[0]


def test_Hypergraph_bijection():
    with raises(ValueError):
        H.spiders(1, 2, Ty('x')).bijection


def test_Hypergraph_rotate():
    assert H.id() == \
           H.id().rotate(left=False).rotate(left=True)


def test_Box():
    box = Box('box', Ty('x'), Ty('y')).to_hypergraph()
    assert box == box and box == box @ H.id() and box != 1


def test_AxiomError():
    x, y = map(Ty, "xy")
    with raises(AxiomError):
        H.cups(x @ y, x @ y)
    with raises(AxiomError):
        H.caps(x @ y, x @ y)


def test_cups():
    x = Ty('x')
    assert H.cups(x, x).make_monogamous().dagger()\
        == H.caps(x, x).make_monogamous()
    assert H.caps(x, x).make_monogamous().dagger()\
        == H.cups(x, x).make_monogamous()
    assert H.caps(x, x).to_diagram() == Cap(x, x)
