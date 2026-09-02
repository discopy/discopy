"""B58: gate2zx tests CRz/CRx/CU1 by name before the distance != 1 decomposition, so non-adjacent controlled rotations translate wrong or crash (discopy/quantum/zx.py:359-373).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np
import pytest

pytest.importorskip("pyzx")

from discopy.quantum import CRz, Controlled, Rz  # noqa: E402
from discopy.quantum.gates import U1  # noqa: E402
from discopy.quantum.zx import circuit2zx  # noqa: E402


def unitary(gate):
    n = len(gate.dom)
    return gate.eval().array.reshape(2 ** n, 2 ** len(gate.cod)).T


def same_up_to_phase(a, b):
    a, b = np.asarray(a), np.asarray(b)
    i = np.argmax(np.abs(b))
    return np.allclose(a * (b.flat[i] / a.flat[i]), b, atol=1e-8)


def test_b58_distance_minus_one_matrix():
    gate = Controlled(Rz(0.3), distance=-1)
    matrix = circuit2zx(gate).to_pyzx().to_matrix()
    assert same_up_to_phase(matrix, unitary(gate))


def test_b58_distance_two_has_three_wires():
    diagram = circuit2zx(Controlled(Rz(0.3), distance=2))
    assert len(diagram.dom) == len(diagram.cod) == 3


def test_b58_controlled_u1_at_distance_two_translates():
    diagram = circuit2zx(Controlled(U1(0.3), distance=2))
    assert len(diagram.dom) == len(diagram.cod) == 3


def test_b58_control_adjacent_crz_matches():
    """Passing control: the adjacent CRz is translated right."""
    matrix = circuit2zx(CRz(0.3)).to_pyzx().to_matrix()
    assert same_up_to_phase(matrix, unitary(CRz(0.3)))
