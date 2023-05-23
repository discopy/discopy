""" DisCoPy's drawing modules: legacy and grid. """

from discopy.drawing import legacy, grid
from discopy.drawing.grid import (
    Grid,
    Cell,
    Wire,
)
from discopy.drawing.legacy import (
    draw,
    to_gif,
    Equation,
    Node,
    diagram2nx,
    ATTRIBUTES,
    DEFAULT,
    COLORS,
    SHAPES,
)


def spiral(n_cups):
    """
    Implements the asymptotic worst-case for normal_form, see arXiv:1804.07832.
    """
    from discopy.monoidal import Ty, Box, Id
    x = Ty('x')
    unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
    cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
    for box in [unit, counit, cup, cap]:
        box.draw_as_spider, box.color, box.drawing_name = True, "black", ""
    result = unit
    for i in range(n_cups):
        result = result >> Id(x ** i) @ cap @ Id(x ** (i + 1))
    result = result >> Id(x ** n_cups) @ counit @ Id(x ** n_cups)
    for i in range(n_cups):
        result = result >>\
            Id(x ** (n_cups - i - 1)) @ cup @ Id(x ** (n_cups - i - 1))
    return result
