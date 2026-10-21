# -*- coding: utf-8 -*-
from discopy.utils import AxiomError
from pytest import raises

from discopy.quantum import *
from discopy.quantum.channel import *


def test_CQ():
    assert C(Dim(2, 3)).l == C(Dim(2, 3)).r == C(Dim(3, 2))


def test_Channel():
    dim = C(Dim(2))
    assert Channel.id(C(Dim(2, 2)))\
        == Channel.id(C()).tensor(Channel.id(dim), Channel.id(dim))
    assert Channel.id(C()) + Channel.id(C()) == Channel(2, C(), C())
    with raises(AxiomError):
        Channel.id(C()) + Channel.id(dim)
    assert Channel.id(dim).then(Channel.id(dim), Channel.id(dim)) == Channel.id(dim)
    assert Channel.id(dim).dagger() == Channel.id(dim)
    assert Channel.swap(dim, C()) == Channel.id(dim)
    assert Channel.cups(C(), C()) == Channel.caps(C(), C()) == Channel.id(C())
    assert Channel.id(C()).tensor(Channel.id(C()), Channel.id(C())).array == 1


def test_Functor():
    f = circuit.Box('f', circuit.Ty(), circuit.Ty(), data=[1])
    functor = Functor({}, {}, dtype=complex)
    assert functor(f) == Channel[complex](dom=CQ(), cod=CQ(), array=[1])
    assert functor(sqrt(4)) == Channel[complex](dom=CQ(), cod=CQ(), array=[4])


def test_Channel_measure():
    import numpy as np
    array = np.zeros((2, 2, 2, 2, 2))
    array[0, 0, 0, 0, 0] = array[1, 1, 1, 1, 1] = 1
    assert np.all(Channel.measure(Dim(2), destructive=False).array == array)
    assert Channel.encode(Dim(1)) == Channel.measure(Dim(1)) == Channel.id(C())
    assert Channel.measure(Dim(2, 2))\
        == Channel.measure(Dim(2)) @ Channel.measure(Dim(2))
    array = np.zeros((3, 3, 3))
    for i in range(3):
        array[i, i, i] = 1
    assert np.all(Channel.measure(Dim(3)).array == array)


def test_CQ_str():
    assert str(C(Dim(2))) == "C(Dim(2))"
    assert str(Q(Dim(2))) == "Q(Dim(2))"
    assert str(C(Dim(2)) @ Q(Dim(3))) == "C(Dim(2)) @ Q(Dim(3))"
    assert str(CQ()) == "CQ()"


def test_Channel_dtype_is_preserved():
    dim = C(Dim(2))
    assert Channel[float].cups(dim, dim).dtype == float
    assert Channel[float].discard(Q(Dim(2))).dtype == float


def test_Channel_tensor():
    left, right = Channel.measure(Dim(2)), Channel.id(C(Dim(3)) @ Q(Dim(2)))
    result = left @ right
    assert result.dom == left.dom @ right.dom
    assert result.cod == left.cod @ right.cod
    assert result.array.shape == \
        result.dom.to_dim().inside + result.cod.to_dim().inside
    assert Channel.id(left.dom) @ Channel.id(right.dom)\
        == Channel.id(left.dom @ right.dom)
    assert (left @ right).then(
        Channel.id(left.cod) @ Channel.id(right.cod)) == left @ right


def test_Measure_override_bits_evaluates():
    import numpy as np
    from discopy.quantum import Measure
    channel = Measure(override_bits=True).eval(mixed=True)
    assert channel.dom == C(Dim(2)) @ Q(Dim(2)) and channel.cod == C(Dim(2))
    assert np.allclose(channel.array, np.tensordot(
        np.ones(2), Channel.measure(Dim(2)).array, 0))
