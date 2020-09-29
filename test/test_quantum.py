# -*- coding: utf-8 -*-

from pytest import raises
from discopy.quantum import *


def test_QuantumMap():
    def pure(circuit):
        return CQMap.pure(circuit.pure_eval())
    assert pure(Ket(0)).is_causal
    assert pure(Ket(0, 1)).is_causal
    assert pure(H).is_causal
    assert pure(CX).is_causal
    assert pure(Ket(0, 0, 0) >> H @ CX >> CX @ X).is_causal
    assert pure(Ket(1, 0) >> CX) == pure(Ket(1, 0)) >> pure(CX)
    assert pure(Ket(1)) @ pure(Ket(0)) == pure(Ket(1, 0))
