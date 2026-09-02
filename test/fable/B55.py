"""B55: circuit.Box(is_mixed=False).dagger() and .rotate() come back mixed, so the dagger of a classical box is not transposed and a quantum one cannot be evaluated (discopy/quantum/circuit.py:853-863).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np

from discopy.quantum import Box, bit, qubit
from discopy.tensor import Tensor

QUANTUM = Box('f', qubit ** 2, qubit, data=list(range(8)), is_mixed=False)
CLASSICAL = Box('g', bit, bit, data=[1, 2, 3, 4], is_mixed=False)


def test_b55_dagger_keeps_is_mixed():
    assert QUANTUM.dagger().is_mixed is False


def test_b55_rotate_keeps_is_mixed():
    assert QUANTUM.rotate().is_mixed is False


def test_b55_classical_dagger_is_the_transpose():
    result, expected = CLASSICAL.dagger().eval(), CLASSICAL.eval().dagger()
    assert np.allclose(result.array, expected.array), (
        result.array, expected.array)
    assert isinstance(result, Tensor)


def test_b55_quantum_dagger_evaluates():
    assert QUANTUM.dagger().eval().is_close(QUANTUM.eval().dagger())
