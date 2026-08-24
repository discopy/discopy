# -*- coding: utf-8 -*-
"""B17: Braid.rotate is wrong for dagger braids (discopy/ribbon.py:198).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy.ribbon import Box, Braid, Diagram, Ty


def test_b17_rotate_convention():
    x, y = Ty('x'), Ty('y')
    c = Braid(x, y)
    assert c.rotate().dom == c.cod.r and c.rotate().cod == c.dom.r
    b = Braid(x, y).dagger()
    assert b.rotate().dom == b.cod.r and b.rotate().cod == b.dom.r


def test_b17_rotate_diagram_rescans():
    x, y = Ty('x'), Ty('y')
    f = Box('f', Ty('z'), y @ x)
    d = (f >> Braid(x, y).dagger()).rotate()
    Diagram(d.inside, d.dom, d.cod)
