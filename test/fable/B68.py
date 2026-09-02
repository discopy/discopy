"""B68: Channel.then compares flattened dimensions only, CQ.__str__ prints a classical type as Q(...) and Ket indexes with booleans (discopy/quantum/channel.py:177-182, 96-100; gates.py:339).
Asserts the correct behaviour, red while the bug is live — issue #699.
"""
import numpy as np
import pytest

from discopy.quantum import Ket
from discopy.quantum.channel import C, Channel, Q
from discopy.tensor import Dim
from discopy.utils import AxiomError


def test_b68_then_rejects_mismatched_cq_types():
    with pytest.raises(AxiomError):
        Channel.id(Q(Dim(2))) >> Channel.id(C(Dim(2, 2)))


def test_b68_str_of_classical_type():
    assert str(C(Dim(2))).startswith('C(')


def test_b68_ket_true_is_ket_one_or_raises():
    try:
        ket = Ket(True)
    except (TypeError, ValueError):
        return
    assert ket == Ket(1)
    assert np.allclose(ket.eval().array, Ket(1).eval().array)


def test_b68_numpy_bool_bitstring():
    ket = Ket(*(np.array([1, 0]) > 0))
    assert np.allclose(ket.eval().array, Ket(1, 0).eval().array)
