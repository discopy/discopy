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
    diagramize,
    diagram2nx,
    nx2diagram,
    ATTRIBUTES,
    DEFAULT,
    COLORS,
    SHAPES,
)
