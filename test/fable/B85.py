"""B85: frame_dual_rail is not idempotent, Drawing.then/tensor alias their operand and draw(show=False) leaks a figure (discopy/drawing/drawing.py:378-388, 774, 869; backend.py:1357).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import matplotlib
import matplotlib.pyplot as plt

from discopy import ribbon
from discopy.drawing import Drawing
from discopy.monoidal import Box, Ty

matplotlib.use('Agg')

x, y = Ty('x'), Ty('y')
f = Box('f', x, y)


def test_b85_plain_drawing_renders_the_same_twice_control():
    d = (f >> Box('g', y, x)).to_drawing()
    assert d._repr_svg_() == d._repr_svg_()


def test_b85_dual_rail_drawing_renders_the_same_twice():
    X = ribbon.Ty('x')
    d = ribbon.Braid(X, X).trace(left=False).to_ribbons().to_drawing()
    assert d._repr_svg_() == d._repr_svg_()


def test_b85_then_does_not_alias_its_operand():
    g = f.to_drawing()
    assert (Drawing.id(x) >> g) is not g


def test_b85_tensor_does_not_alias_its_operand():
    g = f.to_drawing()
    assert (Drawing.id() @ g) is not g


def test_b85_draw_without_a_path_leaves_no_open_figure():
    plt.close('all')
    f.draw(show=False)
    assert plt.get_fignums() == []
