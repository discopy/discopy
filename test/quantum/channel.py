# -*- coding: utf-8 -*-

from pytest import raises
from discopy.quantum import *
from discopy.quantum.cqmap import *


def test_CQ():
    assert C(Dim(2, 3)).l == C(Dim(2, 3)).r == C(Dim(3, 2))


def test_Channel():
    with raises(ValueError):
        Channel(CQ(), CQ())
    dim = C(Dim(2))
    assert Channel.id(C(Dim(2, 2)))\
        == Channel.id(C()).tensor(Channel.id(dim), Channel.id(dim))
    assert Channel.id(C()) + Channel.id(C()) == Channel(C(), C(), 2)
    with raises(AxiomError):
        Channel.id(C()) + Channel.id(dim)
    assert Channel.id(dim).then(Channel.id(dim), Channel.id(dim)) == Channel.id(dim)
    assert Channel.id(dim).dagger() == Channel.id(dim)
    assert Channel.swap(dim, C()) == Channel.id(dim)
    assert Channel.cups(C(), C()) == Channel.caps(C(), C()) == Channel.id(C())
    assert Channel.id(C()).tensor(Channel.id(C()), Channel.id(C())).utensor == 1


def test_Functor():
    x = circuit.Ty('x')
    f = circuit.Box('f', x, x)
    f.array = [1]
    functor = Functor({x: CQ()}, {})
    assert repr(functor) == "cqmap.Functor(ob={x: CQ()}, ar={})"
    assert functor(f) == Channel(dom=CQ(), cod=CQ(), array=[1])
    assert functor(sqrt(4)) == Channel(dom=CQ(), cod=CQ(), array=[4])


def test_Channel_measure():
    import numpy as np
    array = np.zeros((2, 2, 2, 2, 2))
    array[0, 0, 0, 0, 0] = array[1, 1, 1, 1, 1] = 1
    assert np.all(Channel.measure(Dim(2), destructive=False).array == array)
    assert Channel.encode(Dim(1)) == Channel.measure(Dim(1)) == Channel.id(C())
    assert Channel.measure(Dim(2, 2))\
        == Channel.measure(Dim(2)) @ Channel.measure(Dim(2))
