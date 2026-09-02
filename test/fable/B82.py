"""B82: rebuilds — a Sum has no hypergraph or map and dies on 'zero', QuantumGate rejects the data circuit.Box accepts, a string-phased Spider has no dagger, to_rigid drops data (discopy/hypergraph.py, quantum/gates.py:237, frobenius.py:272, rigid.py:876).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np
import pytest

from discopy import biclosed, frobenius, symmetric
from discopy.quantum.gates import QuantumGate, qubit
from discopy.utils import AxiomError

x = symmetric.Ty('x')
f, g = symmetric.Box('f', x, x), symmetric.Box('g', x, x)


def test_b82_sum_to_map_raises_a_meaningful_error():
    with pytest.raises((AxiomError, NotImplementedError)):
        (f + g).to_map()


def test_b82_sum_to_hypergraph_raises_a_meaningful_error():
    with pytest.raises((AxiomError, NotImplementedError)):
        (f + g).to_hypergraph()


def test_b82_quantum_gate_accepts_a_2d_array():
    gate = QuantumGate('U', qubit, qubit, data=np.eye(2))
    assert np.allclose(np.asarray(gate.eval().array).reshape(2, 2), np.eye(2))


def test_b82_quantum_gate_accepts_a_symbolic_list():
    sympy = pytest.importorskip("sympy")
    phi = sympy.Symbol('phi')
    gate = QuantumGate('U', qubit, qubit, data=[phi, 0, 0, phi])
    assert phi in gate.free_symbols


def test_b82_string_phased_spider_has_a_dagger():
    spider = frobenius.Spider(1, 1, frobenius.Ty('x'), "$\\phi$")
    assert isinstance(spider.dagger(), frobenius.Spider)


def test_b82_to_rigid_keeps_data():
    a, b = map(biclosed.Ty, "ab")
    assert biclosed.Box('f', a, b, data=42).to_rigid().data == 42
