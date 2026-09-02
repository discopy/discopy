# -*- coding: utf-8 -*-
"""B53: Representation.l is not the dual that the cup convention pairs, cups(V.l, V) and caps(V, V.l) are not intertwiners (discopy/hopf.py:768).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np

from discopy.hopf import Algebra, Intertwiner, Representation

T = Algebra.taft(3)
V = Representation[T].regular()
N = T.dim
COUNIT = T.counit.eval(dtype=complex).array.reshape(N)
COMULT = T.comult.eval(dtype=complex).array.reshape(N, N, N)


def _action(rep):
    return rep.action.eval(dtype=complex).array.reshape(N, N, N)


def _cup_is_linear(left, right):
    image = Intertwiner[T].cups(left, right).eval(dtype=complex).array.reshape(N, N)
    lhs = np.einsum('hpq,pio,qjw,ow->hij', COMULT, _action(left), _action(right), image)
    rhs = np.einsum('h,ij->hij', COUNIT, image)
    return np.allclose(lhs, rhs, atol=1e-6)


def _cap_is_linear(left, right):
    image = Intertwiner[T].caps(left, right).eval(dtype=complex).array.reshape(N, N)
    lhs = np.einsum('hpq,pio,qjw,ij->how', COMULT, _action(left), _action(right), image)
    rhs = np.einsum('h,ow->how', COUNIT, image)
    return np.allclose(lhs, rhs, atol=1e-6)


def test_b53_left_cup_is_an_intertwiner():
    assert _cup_is_linear(V.l, V)


def test_b53_left_cap_is_an_intertwiner():
    assert _cap_is_linear(V, V.l)


def test_b53_right_dual_control():
    "Passing control: V.r pairs correctly in both orientations."
    assert _cup_is_linear(V, V.r) and _cap_is_linear(V.r, V)
