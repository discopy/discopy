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
    spiral = build_spiral(n)
    unit, counit = Box('unit', Ty(), Ty('x')), Box('counit', Ty('x'), Ty())
    assert spiral.boxes[0] == unit and spiral.boxes[n + 1] == counit
    spiral_nf = spiral.normal_form()
    assert spiral_nf.boxes[-1] == counit and spiral_nf.boxes[n] == unit
