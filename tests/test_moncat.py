# -*- coding: utf-8 -*-

from pytest import raises
from discopy.moncat import *


def test_Ty():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    assert x @ y != y @ x
    assert x @ Ty() == x == Ty() @ x
    assert (x @ y) @ z == x @ y @ z == x @ (y @ z)


def test_Ty_init():
    assert list(Ty('x', 'y', 'z')) == [Ob('x'), Ob('y'), Ob('z')]


def test_Ty_repr():
    assert repr(Ty('x', 'y')) == "Ty('x', 'y')"


def test_Ty_str():
    str(Ty('x')) == 'x'


def test_Ty_getitem():
    assert Ty('x', 'y', 'z')[:1] == Ty('x')


def test_Ty_pow():
    assert Ty('x') ** 42 == Ty('x') ** 21 @ Ty('x') ** 21


def test_Diagram_init():
    with raises(ValueError) as err:
        Diagram('x', Ty('x'), [], [])
    assert "Domain of type Ty expected, got 'x'" in str(err.value)
    with raises(ValueError) as err:
        Diagram(Ty('x'), Ty('x'), [], [1])
    assert "Boxes and offsets must have the same length." in str(err.value)
    with raises(ValueError) as err:
        Diagram(Ty('x'), Ty('x'), [1], [1])
    assert "Box of type Diagram expected, got 1" in str(err.value)
    with raises(ValueError) as err:
        Diagram(Ty('x'), Ty('x'), [Box('f', Ty('x'), Ty('y'))], [Ty('x')])
    assert "Offset of type int expected, got Ty('x')" in str(err.value)
