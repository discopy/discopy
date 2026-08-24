# -*- coding: utf-8 -*-
"""B27: Id(x).width raises instead of returning the number of wires (discopy/monoidal.py:962).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy.monoidal import Id, Ty


def test_b27_id_width():
    assert Id(Ty('x')).width == 1
