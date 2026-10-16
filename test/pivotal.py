from pytest import raises

from discopy.utils import AxiomError
from discopy.pivotal import *
from discopy.axioms import assert_axioms
from hypothesis import find
from discopy.utils import dumps, loads


def test_trace():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()


def test_axioms():
    assert_axioms(Ty, Diagram)
    for generator in Diagram.generators:
        box = find(generator.strategy(), lambda _: True)
        assert box == loads(dumps(box)) == eval(repr(box), box.environment())
