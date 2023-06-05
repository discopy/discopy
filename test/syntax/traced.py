from pytest import raises

from discopy.traced import *
from discopy.utils import AxiomError


def test_trace_repr():
    assert repr(Box('f', 'x', 'x').trace()) == "traced.Trace(f, left=False)"


def test_trace_error():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_trace_dagger():
    f = Box('f', 'x', 'x')
    assert f.trace().dagger() == f.dagger().trace()
