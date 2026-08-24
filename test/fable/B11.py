"""B11: subs and lambdify drop box attributes (discopy/quantum/gates.py:588).
Asserts the correct behaviour, red while the bug is live — issue #606."""

import numpy as np
import sympy

from discopy.quantum.gates import CRz, Sqrt, scalar

phi = sympy.Symbol('phi')


def test_b11_subs_keeps_distance():
    assert CRz(phi, distance=2).subs(phi, 0.5).distance == 2


def test_b11_lambdify_keeps_distance():
    assert CRz(phi, distance=2).lambdify(phi)(0.5).distance == 2


def test_b11_subs_keeps_is_mixed():
    assert scalar(2 * phi, is_mixed=True).subs(phi, 0.5).is_mixed


def test_b11_sqrt_dagger_conjugates():
    assert np.allclose(
        Sqrt(1j).dagger().eval().array, np.conj(1j ** 0.5))
