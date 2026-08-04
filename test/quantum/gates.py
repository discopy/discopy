# -*- coding: utf-8 -*-
import numpy

from discopy.quantum import bit, qubit
from discopy.quantum.gates import (
    CRx, CRz, CU1, Controlled, Encode, Measure, Sqrt, X)


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
    for data in (4, 2, -1, -2.5, 1j, 1 + 1j):
        assert numpy.isclose(
            (Sqrt(data) >> Sqrt(data).dagger()).eval().array, abs(data))
        assert Sqrt(data).dagger().dagger() == Sqrt(data)
    assert Sqrt(-1).dagger() != Sqrt(-1)
    assert numpy.isclose(Sqrt(-1).dagger().array, Sqrt(-1).array.conjugate())
    assert repr(Sqrt(-1).dagger()).endswith("[::-1]")
