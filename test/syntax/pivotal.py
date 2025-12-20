from pytest import raises

from discopy.utils import AxiomError
from discopy.pivotal import *


def test_Ob_eq():
    assert Ob('a') == Ob('a').l.r and Ob('a') != 'a'
    assert Ob('a') == cat.Ob('a') and Ob('a', z=1) != cat.Ob('a')


def test_trace():
    with raises(AxiomError):
        Box('f', 'x', 'y').trace()
