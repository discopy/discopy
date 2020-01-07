from pytest import raises
from discopy.moncat import *


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
    spiral = spiral(n)
    unit, counit = Box('unit', Ty(), Ty('x')), Box('counit', Ty('x'), Ty())
    assert spiral.boxes[0] == unit and spiral.boxes[n + 1] == counit
    spiral_nf = spiral.normal_form()
    assert spiral_nf.boxes[-1] == counit and spiral_nf.boxes[n] == unit
