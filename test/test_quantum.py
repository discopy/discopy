# -*- coding: utf-8 -*-

from pytest import raises
from unittest.mock import Mock
from discopy.quantum import *
from discopy import tk


def test_index2bitstring():
    with raises(ValueError):
        index2bitstring(1, 0)


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
    assert CQMap.id(C()).tensor(CQMap.id(C()), CQMap.id(C())).data == 1


def test_CQMapFunctor():
    assert repr(CQMapFunctor({}, {})) == "CQMapFunctor(ob={}, ar={})"


def test_CQMap_measure():
    import numpy as np
    array = np.zeros((2, 2, 2, 2, 2))
    array[0, 0, 0, 0, 0] = array[1, 1, 1, 1, 1] = 1
    assert np.all(CQMap.measure(Dim(2), destructive=False).array == array)
    assert CQMap.encode(Dim(1)) == CQMap.measure(Dim(1)) == CQMap.id(C())
    assert CQMap.measure(Dim(2, 2))\
        == CQMap.measure(Dim(2)) @ CQMap.measure(Dim(2))


def test_BitsAndQubits():
    with raises(TypeError):
        qubit @ Ty('x')


def test_Circuit_repr():
    assert repr(X >> Y)\
        == "Circuit(dom=qubit, cod=qubit, boxes=[X, Y], offsets=[0, 0])"


def test_Circuit_permutation():
    x = qubit
    assert Circuit.swap(x, x ** 2)\
        == Circuit.swap(x, x) @ Id(x) >> Id(x) @ Circuit.swap(x, x)\
        == Circuit.permutation([2, 0, 1])


def test_Circuit_eval():
    with raises(TypeError):
        Box('f', qubit, qubit).eval()
    assert MixedState().eval() == Discard().eval().dagger()


def test_Circuit_to_tk():
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(1) >> Id(1) @ bell_effect)[::-1]
    tk_circ = snake.to_tk()
    assert repr(tk_circ).split('.')[2:-2] == [
        'H(1)',
        'CX(1, 2)',
        'CX(0, 1)',
        'Measure(1, 1)',
        'H(0)',
        'Measure(0, 0)',
        'post_select({0: 0, 1: 0})']
    assert np.isclose(tk_circ.scalar, 2)
    assert repr((CX >> Measure(2) >> Swap(bit, bit)).to_tk())\
        == "tk.Circuit(2, 2).CX(0, 1).Measure(1, 0).Measure(0, 1)"
    assert repr((Bits(0) >> Id(bit) @ Bits(0)).to_tk())\
        == "tk.Circuit(0, 2)"
    assert repr((Bra(0) @ Bits(0) >> Bits(0) @ Id(bit)).to_tk())\
        == "tk.Circuit(1, 3).Measure(0, 1).post_select({1: 0})"


def test_tk_err():
    with raises(TypeError):
        Circuit.from_tk("foo")
    with raises(NotImplementedError):
        QuantumGate("foo", 1, [1, 2, 3, 4]).to_tk()
    with raises(NotImplementedError):
        Bits(1).to_tk()
    with raises(NotImplementedError):
        Circuit.from_tk(tk.Circuit(3).CSWAP(0, 1, 2))


def test_Circuit_from_tk():
    def back_n_forth(f):
        return Circuit.from_tk(f.to_tk())

    m = Measure(1, destructive=False, override_bits=True)
    assert back_n_forth(m) == m
    assert back_n_forth(CRz(0.5)) == CRz(0.5)
    assert Id(qubit @ bit)\
        == back_n_forth(Swap(qubit, bit)) == back_n_forth(Swap(bit, qubit))


def test_Circuit_get_counts():
    assert Id(1).get_counts() == {(): 1.0}


def test_Circuit_get_counts_snake():
    compilation = Mock()
    compilation.apply = lambda x: x
    backend = Mock()
    backend.get_counts.return_value = {
        (0, 0): 240, (0, 1): 242, (1, 0): 271, (1, 1): 271}
    scaled_bell = Circuit.caps(qubit, qubit)
    snake = scaled_bell @ Id(1) >> Id(1) @ scaled_bell[::-1]
    result = np.round(snake.eval(
        backend, compilation=compilation, measure_all=True).array)
    assert result == 1


def test_Circuit_get_counts_empty():
    backend = Mock()
    backend.get_counts.return_value = {}
    with raises(RuntimeError):
        Id(1).get_counts(backend)


def test_Circuit_measure():
    assert Id(0).measure() == 1
    assert all(Bits(0).measure(mixed=True) == np.array([1, 0]))


def test_Box():
    with raises(TypeError):
        Box('f', Ty('x'), bit)
    with raises(TypeError):
        Box('f', bit, Ty('x'))


def test_pure_Box():
    with raises(ValueError):
        Box('f', bit, qubit, is_mixed=False)


def test_Swap():
    assert Swap(bit, qubit).is_mixed
    assert Swap(bit, bit).eval(mixed=True) == CQMap.swap(C(Dim(2)), C(Dim(2)))


def test_Discard():
    assert Discard().dagger() == MixedState()
    assert MixedState().dagger() == Discard()


def test_Measure():
    assert Measure(destructive=False, override_bits=True).dagger()\
        == Encode(constructive=False, reset_bits=True)
    assert Encode().dagger() == Measure()


def test_QuantumGate():
    assert repr(X) == "X"
    assert repr(QuantumGate("s", 0, [1])) == "QuantumGate('s', n_qubits=0, array=[1])"


def test_ClassicalGate():
    f = ClassicalGate('f', 1, 1, [0, 1, 1, 0])
    assert repr(f.dagger())\
        == "ClassicalGate('f', n_bits_in=1, n_bits_out=1, array=[0, 1, 1, 0]).dagger()"


def test_Bits():
    assert repr(Bits(0).dagger()) == "Bits(0).dagger()"
    assert Bits(0).dagger().dagger() == Bits(0)


def test_Rx():
    assert repr(Rx(0.4)) == "Rx(0.4)"
    assert Rx(0).eval() == Rx(0).dagger().eval() == Id(1).eval()


def test_Rz():
    assert Rz(0).eval() == Id(1).eval()


def test_CRz():
    assert CRz(0).eval() == Id(2).eval()


def test_CircuitFunctor():
    x, y = Ty('x'), Ty('y')
    f = rigid.Box('f', x, y)
    ob, ar = {x: qubit, y: bit}, {f: Measure()}
    assert repr(CircuitFunctor(ob, ar))\
        == "CircuitFunctor(ob={Ty('x'): qubit, Ty('y'): bit}, "\
           "ar={Box('f', Ty('x'), Ty('y')): Measure()})"


def test_IQPAnsatz():
    with raises(ValueError):
        IQPansatz(10, np.array([]))
