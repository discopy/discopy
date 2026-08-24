# -*- coding: utf-8 -*-
"""B18: rigid Box.dagger drops the winding number z (discopy/rigid.py:645).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy.rigid import Box, Ty


def test_b18_dagger_keeps_z():
    f = Box('f', Ty('a'), Ty('b'))
    assert f.r.dagger().z == 1


def test_b18_double_dagger():
    f = Box('f', Ty('a'), Ty('b'))
    assert f.r[::-1][::-1] == f.r
