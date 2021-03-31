# -*- coding: utf-8 -*-

from pytest import raises
from discopy.symmetric import *

def test_ValueError():
    x = Ty('x')
    with raises(ValueError):
        Diagram(x, x, [], [])
    with raises(ValueError):
        Diagram(x, x @ x, [], [(0, 1), (0, 2)])
    with raises(ValueError):
        Diagram(x, x, [], [(1, 0)])


def test_AxiomError():
    x, y = types('x y')
    with raises(AxiomError):
        Id(x) >> Id(y)


def test_Diagram():
    assert Id(Ty('x')) != "Id(Ty('x'))"
