# -*- coding: utf-8 -*-

from pytest import raises
from discopy.quantum import *


def test_QuantumMap():
    f, v = X.eval(), Ket(0).eval()
    assert PureMap(v >> f) == PureMap(v) >> PureMap(f)
    assert PureMap(f) >> Discard(Dim(2)) == Discard(Dim(2))
    assert PureMap(v) >> Discard(Dim(2)) == QuantumMap.id(Dim(1))
