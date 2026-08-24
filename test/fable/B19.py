# -*- coding: utf-8 -*-
"""B19: an oversized or negative trace silently succeeds or recurses forever (discopy/utils.py:571).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import sys

import pytest

from discopy.symmetric import Box, Ty
from discopy.utils import AxiomError


def test_b19_oversized_trace_raises():
    a, b = Ty('a'), Ty('b')
    g = Box('g', a @ b, a @ b)
    with pytest.raises(AxiomError):
        g.trace(n=5)


def test_b19_negative_trace_raises():
    a, b = Ty('a'), Ty('b')
    g = Box('g', a @ b, a @ b)
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(100)
    try:
        with pytest.raises(AxiomError):
            g.trace(n=-1)
    finally:
        sys.setrecursionlimit(limit)
