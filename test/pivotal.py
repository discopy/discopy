from pytest import raises

from discopy.utils import AxiomError
from discopy.pivotal import *


def test_trace():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_Sum():
    f = Box('f', 'x', 'x')
    assert Sum([f]) == f
    assert isinstance(f + f, Sum) and (f + f).terms == (f, f)
