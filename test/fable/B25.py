# -*- coding: utf-8 -*-
"""B25: heterogeneous memory feedback is broken and Diagram.discard returns an uninitialised object (discopy/feedback.py:348, discopy/feedback.py:666, discopy/markov.py:217).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy.feedback import Box, Diagram, Ty


def test_b25_heterogeneous_memory_feedback():
    x, y, m, n = map(Ty, 'xymn')
    f = Box('f', x @ (m @ n).delay(), y @ m @ n)
    fb = f.feedback(mem=m @ n)
    assert fb.dom == x and fb.cod == y


def test_b25_discard_builds():
    x = Ty('x')
    d = Diagram.discard(x)
    assert d.dom == x
