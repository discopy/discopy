from pytest import raises

from discopy.utils import AxiomError
from discopy.pivotal import *


def test_trace():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_to_hypergraph():
    x, y = map(Ty, "xy")
    f = Box('f', x, y)
    assert f.transpose().to_hypergraph() == f.to_hypergraph().transpose()
