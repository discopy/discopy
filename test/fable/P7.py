"""P7: representation round-trips agree numerically, zx and tket.
Miniature of the property over curated examples; red while its bullets (B13, B22, B23) are live — issue #606.
"""
import numpy
import pytket as tk

from discopy.quantum import Circuit, Ket, H, Measure, zx
from discopy.quantum.gates import CRz, CRx, CU1, GATES


def _equal_up_to_global_phase(left, right):
    left, right = numpy.asarray(left, complex), numpy.asarray(right, complex)
    i = numpy.argmax(numpy.abs(right))
    if abs(left.flat[i]) < 1e-9:
        return numpy.allclose(left, right)
    return numpy.allclose(left / left.flat[i] * right.flat[i], right)


def _zx_matches(gate):
    matrix = zx.circuit2zx(gate).to_pyzx().to_matrix()
    expected = numpy.asarray(gate.eval().array, complex).reshape(4, 4)
    return _equal_up_to_global_phase(matrix, expected)


def _from_tk_cu1():
    circuit = tk.Circuit(2)
    circuit.CU1(0.5, 0, 1)
    return Circuit.from_tk(circuit) is not None


def _from_tk_measure_evals():
    back = Circuit.from_tk((Ket(0) >> H >> Measure()).to_tk())
    return back.eval() is not None


def test_p7():
    cases = [
        ("zx round-trip CRz(0.3)", lambda: _zx_matches(CRz(0.3))),
        ("zx round-trip CRx(0.3)", lambda: _zx_matches(CRx(0.3))),
        ("zx round-trip CU1(0.25)", lambda: _zx_matches(CU1(0.25))),
        ("GATES['CCX'].to_tk()", lambda: GATES['CCX'].to_tk() is not None),
        ("Circuit.from_tk of tk CU1", _from_tk_cu1),
        ("from_tk circuit with Measure evaluates", _from_tk_measure_evals),
    ]
    failures = []
    for label, law in cases:
        try:
            if not law():
                failures.append(f"{label}: law violated")
        except Exception as error:
            failures.append(f"{label}: raised {type(error).__name__}")
    assert not failures, failures
