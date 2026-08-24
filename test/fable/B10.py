"""B10: Controlled.array ignores dagger and conjugate (discopy/quantum/gates.py:534).
Asserts the correct behaviour, red while the bug is live — issue #606."""

import numpy as np

from discopy.quantum.gates import Controlled, S


def test_b10_controlled_dagger_array():
    CS = Controlled(S)
    assert np.allclose(
        CS.dagger().eval().array.reshape(4, 4),
        CS.eval().array.reshape(4, 4).conj().T)


def test_b10_controlled_dagger_unitary():
    CS = Controlled(S)
    assert np.allclose(
        (CS >> CS.dagger()).eval().array.reshape(4, 4), np.eye(4))
