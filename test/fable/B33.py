"""B33: to_tn crashes on bit discards and non-destructive measures, measure() crashes on classical circuits (discopy/quantum/circuit.py:441-485,399).
Asserts the correct behaviour, red while the bug is live — issue #606.
"""
import numpy as np
import tensornetwork as tn

from discopy.quantum import Bits, Copy, Discard, Measure, bit


def test_b33_to_tn_bit_discard():
    circuit = Bits(0) >> Discard(bit)
    expected = circuit.eval()
    result = circuit.eval(contractor=tn.contractors.auto)
    assert np.allclose(np.asarray(result.array), np.asarray(expected.array))


def test_b33_to_tn_nondestructive_measure():
    expected = Measure(destructive=False).eval()
    result = Measure(destructive=False).eval(contractor=tn.contractors.auto)
    assert np.allclose(np.asarray(result.array), np.asarray(expected.array))


def test_b33_measure_classical_circuit():
    array = Copy().measure()
    assert np.asarray(array) is not None
