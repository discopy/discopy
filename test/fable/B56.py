"""B56: to_tk drops a permutation over mixed bit/qubit types, which Swap(bit, bit) @ qubit is since #594 (discopy/quantum/tk.py:303-316).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np
import pytest

pytest.importorskip("pytket")

from discopy.quantum import (  # noqa: E402
    ClassicalGate, Discard, H, Ket, Measure, Swap, bit, qubit)
from discopy.quantum.tk import mockBackend  # noqa: E402

PREPARE = Ket(0, 0, 0) >> H @ qubit ** 2 >> Measure(2) @ qubit


def test_b56_swap_reaches_tket():
    with_swap = (PREPARE >> Swap(bit, bit) @ qubit).to_tk()
    assert repr(with_swap) != repr(PREPARE.to_tk())


def test_b56_eval_through_backend_matches_numpy():
    """A classical gate first, so the swap lands in the post-processing the
    all-bits branch already uses, and the mock counts stay meaningful."""
    post = ClassicalGate('post', bit ** 2, bit ** 2, data=np.eye(4).flatten())
    c = PREPARE >> post @ qubit >> Swap(bit, bit) @ qubit
    backend = mockBackend({(0, 0): 512, (1, 0): 512})
    expected = (c >> bit ** 2 @ Discard()).eval().array.real
    result = c.eval(backend=backend).array
    assert np.allclose(result, expected), (result, expected)
