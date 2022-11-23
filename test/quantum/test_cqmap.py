# -*- coding: utf-8 -*-

from pytest import raises
from discopy.quantum import *
from discopy.quantum.cqmap import *


def test_CQ():
    assert C(Dim(2, 3)).l == C(Dim(2, 3)).r == C(Dim(3, 2))


def test_CQMap():
    with raises(ValueError):
        CQMap(CQ(), CQ())
    dim = C(Dim(2))
    assert CQMap.id(C(Dim(2, 2)))\
        == CQMap.id(C()).tensor(CQMap.id(dim), CQMap.id(dim))
    assert CQMap.id(C()) + CQMap.id(C()) == CQMap(C(), C(), 2)
    with raises(AxiomError):
        CQMap.id(C()) + CQMap.id(dim)
    assert CQMap.id(dim).then(CQMap.id(dim), CQMap.id(dim)) == CQMap.id(dim)
    assert CQMap.id(dim).dagger() == CQMap.id(dim)
    assert CQMap.swap(dim, C()) == CQMap.id(dim)
    assert CQMap.cups(C(), C()) == CQMap.caps(C(), C()) == CQMap.id(C())
    assert CQMap.id(C()).tensor(CQMap.id(C()), CQMap.id(C())).utensor == 1


def test_Functor():
    x = circuit.Ty('x')
    f = circuit.Box('f', x, x)
    f.array = [1]
    functor = Functor({x: CQ()}, {})
    assert repr(functor) == "cqmap.Functor(ob={x: CQ()}, ar={})"
    assert functor(f) == CQMap(dom=CQ(), cod=CQ(), array=[1])
    assert functor(sqrt(4)) == CQMap(dom=CQ(), cod=CQ(), array=[4])


def test_CQMap_measure():
    import numpy as np
    array = np.zeros((2, 2, 2, 2, 2))
    array[0, 0, 0, 0, 0] = array[1, 1, 1, 1, 1] = 1
    assert np.all(CQMap.measure(Dim(2), destructive=False).array == array)
    assert CQMap.encode(Dim(1)) == CQMap.measure(Dim(1)) == CQMap.id(C())
    assert CQMap.measure(Dim(2, 2))\
        == CQMap.measure(Dim(2)) @ CQMap.measure(Dim(2))
