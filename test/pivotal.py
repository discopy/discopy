from pytest import raises

from discopy.utils import AxiomError
from discopy.pivotal import *


def test_trace():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_strategy():
    from hypothesis import find

    from discopy import axioms

    axioms.assert_strategy_finds(Diagram, Cup, Cap)
    winding = find(Ty.strategy(min_length=1), lambda value: value.inside[0].z)
    assert winding.inside[0].z
