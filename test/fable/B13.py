# -*- coding: utf-8 -*-
"""B13: gate2zx mistranslates CRx and CU1 (discopy/quantum/zx.py:357-362).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import numpy as np

from discopy.quantum.gates import CRx, CRz, CU1
from discopy.quantum.zx import circuit2zx


def _phase_normalised(matrix):
    matrix = np.asarray(matrix, dtype=complex).reshape(4, 4)
    first = matrix.flat[np.flatnonzero(np.abs(matrix) > 1e-9)[0]]
    return matrix / first


def _check(gate):
    via_zx = np.asarray(circuit2zx(gate).to_pyzx().to_matrix())
    direct = np.asarray(gate.eval().array).reshape(4, 4)
    assert np.allclose(
        _phase_normalised(via_zx), _phase_normalised(direct), atol=1e-8)


def test_b13_crz_control():
    _check(CRz(0.3))


def test_b13_crx():
    _check(CRx(0.3))


def test_b13_cu1():
    _check(CU1(0.25))
