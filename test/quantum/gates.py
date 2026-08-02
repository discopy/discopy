# -*- coding: utf-8 -*-
from pytest import raises

from discopy.quantum import bit, qubit
from discopy.quantum.gates import (
    CRx, CRz, CU1, Controlled, Encode, Measure, Rz, Sqrt, X)


def test_Encode_types():
    assert Encode().dom == bit and Encode().cod == qubit
    for destructive in (True, False):
        for override_bits in (True, False):
            measure = Measure(
                destructive=destructive, override_bits=override_bits)
            assert measure.dagger().dom == measure.cod
            assert measure.dagger().cod == measure.dom
    assert (Measure(destructive=False) >> Encode(constructive=False)).cod\
        == qubit


def test_Controlled_with_distance():
    assert Controlled(X, distance=-1).with_distance(1) == Controlled(X)
    assert CRz(0.25, distance=-1).with_distance(1) == CRz(0.25)
    for factory in (CRz, CRx, CU1):
        for distance in (-2, -1, 2, 3):
            assert factory(0.25, distance=distance).array is not None


def test_Sqrt():
    assert Sqrt(4).array == 2
    assert Sqrt(0).array == 0
    for undefined in (-1, -2.5, 1j, 1 + 1j):
        with raises(ValueError):
            Sqrt(undefined)
    assert Sqrt(2).dagger() == Sqrt(2)
