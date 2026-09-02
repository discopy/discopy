# -*- coding: utf-8 -*-
"""B70: closed.Eval defaults right-handed against Diagram.ev, Curry never checks n, Eval accepts a contradicting left (discopy/biclosed.py:366, :425).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import pytest

from discopy import biclosed, closed
from discopy.utils import AxiomError


def test_b70_closed_eval_default_matches_ev():
    X, Y = closed.Ty('X'), closed.Ty('Y')
    assert closed.Eval(X >> Y) == closed.Diagram.ev(Y, X)


def test_b70_oversized_curry_raises():
    x, z = biclosed.Ty('x'), biclosed.Ty('z')
    with pytest.raises(AxiomError):
        biclosed.Curry(biclosed.Box('f', x, z), n=5)


def test_b70_eval_with_contradicting_left_is_rejected_or_stable():
    x, y = biclosed.Ty('x'), biclosed.Ty('y')
    try:
        box = biclosed.Eval(x << y, left=False)
    except (AxiomError, ValueError):
        return
    assert biclosed.Functor.id(biclosed.Diagram)(box) == box
