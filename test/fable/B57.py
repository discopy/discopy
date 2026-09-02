"""B57: to_tk drops Discard(bit), so a backend evaluation keeps the discarded bit and post-processes at the wrong offset (discopy/quantum/tk.py:298-302).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np
import pytest

pytest.importorskip("pytket")

from discopy.quantum import Bits, Discard, H, Ket, Measure, bit  # noqa: E402
from discopy.quantum.tk import mockBackend  # noqa: E402
from discopy.tensor import Dim  # noqa: E402

CIRCUIT = Ket(0, 0) >> H @ H >> Measure(2) >> Discard(bit) @ bit
BACKEND = mockBackend({(0, 0): 256, (0, 1): 256, (1, 0): 256, (1, 1): 256})


def test_b57_backend_eval_has_the_circuit_cod():
    assert CIRCUIT.eval(backend=BACKEND).cod == Dim(2)


def test_b57_marginal_matches_numpy():
    c = CIRCUIT >> Bits(0).dagger()
    result, expected = c.eval(backend=BACKEND).array, c.eval().array.real
    assert result.shape == expected.shape and np.allclose(result, expected), (
        result, expected)
