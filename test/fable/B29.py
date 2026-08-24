# -*- coding: utf-8 -*-
"""B29: validate_attributes raises TypeError and dagger asserts on drawings with two or more boxes (discopy/drawing/drawing.py:325, discopy/drawing/drawing.py:897).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy.monoidal import Box, Ty


def test_b29_validate_attributes():
    x, y = Ty('x'), Ty('y')
    Box('f', x, y).to_drawing().validate_attributes()


def test_b29_dagger_two_boxes():
    x, y, z = Ty('x'), Ty('y'), Ty('z')
    f, g = Box('f', x, y), Box('g', y, z)
    drawing = (f >> g).to_drawing()
    daggered = drawing.dagger()
    assert daggered.dom == drawing.cod
