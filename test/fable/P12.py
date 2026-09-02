"""P12: converters agree — eval, to_tk, to_pyzx and to_tn give the same numbers on circuits with a mixed-type permutation, a bit discard and non-adjacent controlled rotations.
Miniature of the property over curated examples; red while its bullets (B56, B57, B58) are live — issue #699.
"""
import numpy as np
import pytest

pytest.importorskip("pytket")
pytest.importorskip("pyzx")
tn = pytest.importorskip("tensornetwork")

from discopy.quantum import (  # noqa: E402
    Bits, Circuit, ClassicalGate, Controlled, Discard, H, Ket, Measure, Rx,
    Rz, Swap, bit, qubit)
from discopy.quantum.tk import mockBackend  # noqa: E402
from discopy.quantum.zx import circuit2zx  # noqa: E402

PURE = [
    ("perm, CRz(-1), CRx(2)",
     Circuit.permutation([2, 0, 1])
     >> Controlled(Rz(0.3), distance=-1) @ qubit
     >> Controlled(Rx(0.4), distance=2)),
    ("H, CRz(2), CRz(-2)",
     H @ qubit ** 2 >> Controlled(Rz(0.3), distance=2)
     >> Controlled(Rz(0.5), distance=-2)),
]
PREPARE = Ket(0, 0, 0) >> H @ qubit ** 2 >> Measure(2) @ qubit
POST = ClassicalGate('post', bit ** 2, bit ** 2, data=np.eye(4).flatten())
MIXED = [
    ("bit swap beside a qubit",
     PREPARE >> POST @ qubit >> Swap(bit, bit) @ qubit >> bit ** 2 @ Discard(),
     {(0, 0): 512, (1, 0): 512}),
    ("bit discard then classical box",
     Ket(0, 0) >> H @ H >> Measure(2) >> Discard(bit) @ bit
     >> Bits(0).dagger(),
     {(0, 0): 256, (0, 1): 256, (1, 0): 256, (1, 1): 256}),
]


def unitary(circuit):
    n = len(circuit.dom)
    return circuit.eval().array.reshape(2 ** n, 2 ** len(circuit.cod)).T


def same_up_to_phase(a, b):
    a, b = np.asarray(a), np.asarray(b)
    i = np.argmax(np.abs(b))
    return np.allclose(a * (b.flat[i] / a.flat[i]), b, atol=1e-8)


def check(failures, label, law):
    try:
        if not law():
            failures.append(f"{label}: disagree")
    except Exception as error:
        failures.append(f"{label}: raised {type(error).__name__}")


def test_p12():
    failures = []
    for label, c in PURE:
        check(failures, f"{label} to_tk", lambda: Circuit.from_tk(
            c.to_tk()).eval().is_close(c.init_and_discard().eval()))
        check(failures, f"{label} to_pyzx", lambda: same_up_to_phase(
            circuit2zx(c).to_pyzx().to_matrix(), unitary(c)))
        check(failures, f"{label} to_tn", lambda: c.eval(
            contractor=tn.contractors.auto).is_close(c.eval()))
    for label, c, counts in MIXED:
        check(failures, f"{label} to_tk", lambda: np.allclose(
            c.eval(backend=mockBackend(counts)).array,
            c.eval().array.real))
        check(failures, f"{label} to_tn", lambda: c.eval(
            contractor=tn.contractors.auto, mixed=True).is_close(c.eval()))
    assert not failures, failures
