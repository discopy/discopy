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


def test_Diagram_offsets():
    assert Diagram(Ty('x'), Ty('x'), [], []).offsets == []


def test_Diagram_repr():
    x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    assert repr(Diagram(x, x, [], [])) == "Id(Ty('x'))"
    f0, f1 = Box('f0', x, y), Box('f1', z, w)
    assert "Diagram(dom=Ty('x'), cod=Ty('y')" in repr(Diagram(x, y, [f0], [0]))
    assert "offsets=[0]" in repr(Diagram(x, y, [f0], [0]))
    assert "Diagram(dom=Ty('x', 'z'), cod=Ty('y', 'w')" in repr(f0 @ f1)
    assert "offsets=[0, 1]" in repr(f0 @ f1)


def test_Diagram_hash():
    assert {Id(Ty('x')): 42}[Id(Ty('x'))] == 42


def test_Diagram_str():
    x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    assert str(Diagram(x, x, [], [])) == "Id(x)"
    f0, f1 = Box('f0', x, y), Box('f1', z, w)
    assert str(Diagram(x, y, [f0], [0])) == "f0"
    assert str(f0 @ Id(z) >> Id(y) @ f1) == "f0 @ Id(z) >> Id(y) @ f1"
    assert str(f0 @ Id(z) >> Id(y) @ f1) == "f0 @ Id(z) >> Id(y) @ f1"


def test_Diagram_matmul():
    assert Id(Ty('x')) @ Id(Ty('y')) == Id(Ty('x', 'y'))
    assert Id(Ty('x')) @ Id(Ty('y')) == Id(Ty('x')).tensor(Id(Ty('y')))


def spiral(n_cups):
    """
    Implements the asymptotic worst-case for normal_form, see arXiv:1804.07832.
    """
    x = Ty('x')  # pylint: disable=invalid-name
    unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
    cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
    result = unit
    for i in range(n_cups):
        result = result >> Id(x ** i) @ cap @ Id(x ** (i + 1))
    result = result >> Id(x ** n_cups) @ counit @ Id(x ** n_cups)
    for i in range(n_cups):
        result = result >>\
            Id(x ** (n_cups - i - 1)) @ cup @ Id(x ** (n_cups - i - 1))
    return result


def test_spiral(n=2):
    spira = spiral(n)
    unit, counit = Box('unit', Ty(), Ty('x')), Box('counit', Ty('x'), Ty())
    assert spira.boxes[0] == unit and spira.boxes[n + 1] == counit
    spira_nf = spira.normal_form()
    assert spira_nf.boxes[-1] == counit and spira_nf.boxes[n] == unit
