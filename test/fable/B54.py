"""B54: Controlled.l and .r are the conjugate rather than the transpose, and ControlledRotation.rotate crashes (discopy/quantum/gates.py:487, 630).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
from discopy.quantum import CRz, CX, Controlled, H, Ket, qubit


def test_b54_l_is_the_transpose():
    assert CRz(0.3).l.eval().is_close(CRz(0.3).eval().l)


def test_b54_r_is_the_transpose():
    assert CRz(0.3).r.eval().is_close(CRz(0.3).eval().r)


def test_b54_circuit_l_is_the_transpose():
    c = Ket(0, 0) >> H @ qubit >> CRz(0.3)
    assert c.l.eval().is_close(c.eval().l)


def test_b54_controlled_rotation_rotate_builds():
    assert isinstance(CRz(0.5).rotate(), Controlled)


def test_b54_control_cx_is_invisible():
    """Passing control: CX is real-symmetric so conjugate == transpose."""
    assert CX.l.eval().is_close(CX.eval().l)
