from pytest import raises

from discopy.utils import AxiomError
from discopy.pivotal import *
from discopy.axioms import assert_axioms


def test_trace():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_axioms():
    assert_axioms(Ty, Diagram)
