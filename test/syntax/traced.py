from pytest import raises
from discopy.traced import *


def test_trace():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()
