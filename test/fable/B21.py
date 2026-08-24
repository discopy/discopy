# -*- coding: utf-8 -*-
"""B21: Bubble.dagger crashes and Curry, Trace and Twist fail loads(dumps) (discopy/cat.py:737).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
from discopy import balanced, cat, closed, symmetric
from discopy.utils import dumps, loads


def test_b21_bubble_dagger():
    b = cat.Box('f', cat.Ob('x'), cat.Ob('y')).bubble()
    assert b.dagger().dagger() == b


def test_b21_curry_roundtrip():
    x, y = closed.Ty('x'), closed.Ty('y')
    obj = closed.Box('g', x, y).curry()
    assert loads(dumps(obj)) == obj


def test_b21_trace_roundtrip():
    x = symmetric.Ty('x')
    obj = symmetric.Box('g', x, x).trace(left=False)
    loaded = loads(dumps(obj))
    assert loaded == obj and loaded.left == obj.left


def test_b21_twist_roundtrip():
    obj = balanced.Twist(balanced.Ty('x'))
    assert loads(dumps(obj)) == obj
