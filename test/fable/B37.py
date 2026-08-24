"""B37: drawing a Drawing mutates it, so two equal drawings differ after draw() (discopy/drawing/drawing.py).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import matplotlib

from discopy.monoidal import Box, Ty

matplotlib.use('Agg')


def test_b37_draw_does_not_mutate(tmp_path):
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', y, z)
    drawn, fresh = (f >> g).to_drawing(), (f >> g).to_drawing()
    assert drawn == fresh
    drawn.draw(path=str(tmp_path / 'x.png'))
    assert drawn == fresh
