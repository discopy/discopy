from pytest import raises

from discopy.traced import *


def test_trace():
    assert repr(Box('f', 'x', 'x').trace()) == "traced.Trace(f, left=False)"
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()
