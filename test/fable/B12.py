"""B12: the pure-branch gradient of U1 is wrong (discopy/quantum/gates.py:649).
Asserts the correct behaviour, red while the bug is live — issue #606."""

import numpy as np
import sympy

from discopy.quantum.gates import U1


def test_b12_u1_pure_grad():
    phi = sympy.Symbol('phi')
    grad = U1(phi).grad(phi, mixed=False)
    value = grad.lambdify(phi)(0.3).eval().array.reshape(2, 2)
    eps = 1e-6
    finite_difference = (
        U1(0.3 + eps).eval().array - U1(0.3 - eps).eval().array
    ).reshape(2, 2) / (2 * eps)
    assert np.allclose(value, finite_difference, atol=1e-4)
