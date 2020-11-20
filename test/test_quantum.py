# -*- coding: utf-8 -*-

from pytest import raises
from unittest.mock import Mock
from discopy.quantum.cqmap import *
from discopy.quantum.circuit import *
from discopy.quantum.gates import *
from discopy.quantum import tk


def test_index2bitstring():
    with raises(ValueError):
        index2bitstring(1, 0)
    assert index2bitstring(42, 8) == (0, 0, 1, 0, 1, 0, 1, 0)


def test_bitstring2index():
    assert bitstring2index((0, 0, 1, 0, 1, 0, 1, 0)) == 42


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
    with raises(AttributeError):
        Box('f', qubit, qubit).eval()
    assert MixedState().eval() == Discard().eval().dagger()


def test_Circuit_cups_and_caps():
    assert Circuit.cups(bit, bit) == Match() >> Discard(bit)
    assert Circuit.caps(bit, bit) == MixedState(bit) >> Copy()
    with raises(ValueError):
        Circuit.cups(Ty('x'), Ty('x').r)


def test_Circuit_to_tk():
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(1) >> Id(1) @ bell_effect)[::-1]
    tk_circ = snake.to_tk()
    assert repr(tk_circ) ==\
        'tk.Circuit(3, 2)'\
        '.H(1)'\
        '.CX(1, 2)'\
        '.CX(0, 1)'\
        '.Measure(1, 1)'\
        '.H(0)'\
        '.Measure(0, 0)'\
        '.post_select({0: 0, 1: 0})'\
        '.scale(2)'
    assert np.isclose(tk_circ.scalar, 2)
    assert repr((CX >> Measure(2) >> Swap(bit, bit)).to_tk())\
        == "tk.Circuit(2, 2).CX(0, 1).Measure(1, 0).Measure(0, 1)"
    assert repr((Bits(0) >> Id(bit) @ Bits(0)).to_tk())\
        == "tk.Circuit(0, 2)"
    assert repr((Bra(0) @ Bits(0) >> Bits(0) @ Id(bit)).to_tk())\
        == "tk.Circuit(1, 3).Measure(0, 1)"\
           ".post_select({1: 0}).post_process(Swap(bit, bit))"


def test_Sum_from_tk():
    assert Circuit.from_tk(*(X + X).to_tk()) == (X + X).init_and_discard()
    assert Circuit.from_tk() == Sum([], qubit ** 0, qubit ** 0)


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
    assert back_n_forth(m) == m.init_and_discard()
    assert back_n_forth(CRz(0.5)) ==\
        Ket(0) @ Ket(0) >> CRz(0.5) >> Discard() @ Discard()
    assert Id(qubit @ bit).init_and_discard()\
        == back_n_forth(Swap(qubit, bit)) == back_n_forth(Swap(bit, qubit))


def test_ClassicalGate_to_tk():
    post = ClassicalGate('post', n_bits_in=2, n_bits_out=0, data=[0, 0, 0, 1])
    assert (post[::-1] >> Swap(bit, bit)).to_tk().post_processing\
        == post[::-1] >> Swap(bit, bit)
    circuit = sqrt(2) @ Ket(0, 0) >> H @ Rx(0) >> CX >> Measure(2) >> post
    assert Circuit.from_tk(circuit.to_tk())[-1] == post


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
    assert not Id(1).get_counts(backend)


def test_Circuit_measure():
    assert Id(0).measure() == 1
    assert all(Bits(0).measure(mixed=True) == np.array([1, 0]))


def test_Bra_and_Measure_to_tk():
    c = Circuit(
        dom=qubit ** 0, cod=bit, boxes=[
            Ket(0), Rx(0.552), Rz(0.512), Rx(0.917), Ket(0, 0, 0), H, H, H,
            CRz(0.18), CRz(0.847), CX, H, sqrt(2), Bra(0, 0), Ket(0),
            Rx(0.446), Rz(0.256), Rx(0.177), CX, H, sqrt(2), Bra(0, 0),
            Measure()],
        offsets=[
            0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 2,
            3, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    assert repr(c.to_tk()) ==\
        "tk.Circuit(5, 5)"\
        ".Rx(0.892, 0)"\
        ".H(1)"\
        ".H(2)"\
        ".H(3)"\
        ".Rx(1.104, 4)"\
        ".Rz(0.512, 0)"\
        ".CRz(0.36, 1, 2)"\
        ".Rz(1.024, 4)"\
        ".Rx(0.354, 0)"\
        ".CRz(1.694, 2, 3)"\
        ".Rx(1.834, 4)"\
        ".Measure(2, 4)"\
        ".CX(0, 1)"\
        ".CX(3, 4)"\
        ".Measure(4, 1)"\
        ".Measure(1, 3)"\
        ".H(0)"\
        ".H(3)"\
        ".Measure(3, 0)"\
        ".Measure(0, 2)"\
        ".post_select({0: 0, 1: 0, 2: 0, 3: 0})"\
        ".scale(2)"


def test_ClassicalGate_eval():
    backend = Mock()
    backend.get_counts.return_value = {
        (0, 0): 256, (0, 1): 256, (1, 0): 256, (1, 1): 256}
    post = ClassicalGate('post', 2, 0, [1, 0, 0, 0])
    assert post.eval(backend) == Tensor(dom=Dim(1), cod=Dim(1), array=[0.25])


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
    assert repr(QuantumGate("s", 0, [1]))\
        == "QuantumGate('s', n_qubits=0, array=[1])"


def test_ClassicalGate():
    f = ClassicalGate('f', 1, 1, [0, 1, 1, 0])
    assert repr(f.dagger())\
        == "ClassicalGate('f', n_bits_in=1, n_bits_out=1, "\
           "data=[0, 1, 1, 0]).dagger()"


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


def test_CU1():
    assert CU1(0).eval() == Id(2).eval()


def test_CRx():
    assert CRx(0).eval() == Id(2).eval()


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


def test_Sum():
    assert not Sum([], qubit, qubit).eval()
    assert Sum([Id(0)]).get_counts() == Id(0).get_counts()
    assert (Id(0) + Id(0)).get_counts()[()] == 2
    assert (Id(0) + Id(0)).eval() == Id(0).eval() + Id(0).eval()
    assert not (X + X).is_mixed and (X >> Bra(0) + Discard()).is_mixed
    assert (Discard() + Discard()).eval()\
        == Discard().eval() + Discard().eval()


def test_subs():
    from sympy.abc import phi
    assert (Rz(phi) + Rz(phi + 1)).subs(phi, 1) == Rz(1) + Rz(2)
    circuit = sqrt(2) @ Ket(0, 0) >> H @ Rx(phi) >> CX >> Bra(0, 1)
    assert circuit.subs(phi, 0.5)\
        == sqrt(2) @ Ket(0, 0) >> H @ Rx(0.5) >> CX >> Bra(0, 1)


def test_grad():
    from sympy.abc import phi
    with raises(NotImplementedError):
        Box('f', qubit, qubit, data=phi).grad(phi)
    with raises(NotImplementedError):
        CRz(phi).grad(phi)
    assert scalar(1).grad(phi) == Sum([], qubit ** 0, qubit ** 0)
    assert (Rz(phi) + Rz(2 * phi)).grad(phi)\
        == Rz(phi).grad(phi) + Rz(2 * phi).grad(phi)
    assert scalar(phi).grad(phi) == scalar(1)
    assert Rz(0).grad(phi) == X.grad(phi) == Sum([], qubit, qubit)


def test_ClassicalGate_grad_subs():
    from sympy.abc import x, y
    s = ClassicalGate('s', 0, 0, [x])
    assert s.grad(x) and not s.subs(x, y).grad(x)


def test_Copy_Match():
    assert Match().dagger() == Copy() and Copy().dagger() == Match()


def test_non_linear_ClassicalGate():
    f = ClassicalGate("f", 2, 2, lambda array: np.sin(array) ** 2)
    state = Bits(0, 0) + Bits(0, 1) + Bits(1, 0) + Bits(1, 1)
    vector = (state >> f).eval().array.flatten()
    assert np.all(vector == 4 * [np.sin(1) ** 2])


def test_non_linear_AxiomError():
    f = ClassicalGate("f", 2, 2, lambda array: np.sin(array) ** 2)
    with raises(AttributeError):
        f.array
    with raises(AxiomError):
        f.eval()
    with raises(AxiomError):
        (f @ f).eval()
    with raises(AxiomError):
        (f >> Discard(bit ** 2)).eval()


def test_Sum_get_counts():
    assert Sum([], qubit, qubit).get_counts() == {}
