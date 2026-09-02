# -*- coding: utf-8 -*-
"""B52: hopf.Functor sends the twist to the inverse of the trace of the braid (discopy/hopf.py:938).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np

from discopy import ribbon
from discopy.hopf import Algebra, Double, Functor, Intertwiner, Representation


def _functor():
    D3 = Double(Algebra.cyclic(3))
    omega = np.exp(2j * np.pi / 3)
    x = ribbon.Ty('x')
    F = Functor({x: Representation[D3].anyon(1, omega)}, {}, cod=Intertwiner[D3])
    return x, F


def _value(F, diagram):
    return complex(F(diagram).eval(dtype=complex))


def test_b52_twist_is_right_trace_of_braid():
    x, F = _functor()
    twist, trace = _value(F, ribbon.Twist(x)), _value(F, ribbon.Braid(x, x).trace())
    assert np.isclose(twist, trace), (twist, trace)


def test_b52_twist_is_left_trace_of_braid():
    x, F = _functor()
    twist = _value(F, ribbon.Twist(x))
    trace = _value(F, ribbon.Braid(x, x).trace(left=True))
    assert np.isclose(twist, trace), (twist, trace)
