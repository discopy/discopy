"""B59: Scalar.grad drops is_mixed, so the derivative of a mixed scalar is squared under mixed evaluation (discopy/quantum/gates.py:741-744).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np
from sympy.abc import phi

from discopy.quantum import scalar

SCALAR = scalar(2 * phi, is_mixed=True)


def test_b59_grad_keeps_is_mixed():
    assert SCALAR.grad(phi).is_mixed


def test_b59_grad_mixed_eval():
    assert np.isclose(SCALAR.grad(phi).eval(mixed=True).array, 2)
