# -*- coding: utf-8 -*-

from unittest.mock import Mock
from functools import reduce, partial
import itertools
from pytest import raises
import numpy as np
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


def test_Circuit_ob():
    with raises(AxiomError):
        Ob("x", z=-1)
    with raises(ValueError):
        Ob("x", dim=-1)


def test_Circuit_repr():
    assert repr(X >> Y)\
        == "Circuit(dom=qubit, cod=qubit, boxes=[X, Y], offsets=[0, 0])"


def test_Circuit_permutation():
    x = qubit
    assert Circuit.swap(x, x ** 2)\
        == Circuit.swap(x, x) @ Id(x) >> Id(x) @ Circuit.swap(x, x)\
        == Circuit.permutation([2, 0, 1])


def test_Circuit_eval():
    with raises(KeyError):
        Box('f', qubit, qubit).eval()
    assert MixedState().eval() == Discard().eval().dagger()


def test_Circuit_cups_and_caps():
    assert Circuit.cups(bit, bit) == Match() >> Discard(bit)
    assert Circuit.caps(bit, bit) == MixedState(bit) >> Copy()
    with raises(ValueError):
        Circuit.cups(Ty('x'), Ty('x').r)


def test_Circuit_spiders():
    assert Circuit.spiders(123, 456, qubit ** 0) == Id()
    assert Circuit.spiders(0, 0, qubit) == (sqrt(2) >> Ket(0)
                                            >> H >> H
                                            >> Bra(0) >> sqrt(2))
    assert Circuit.spiders(1, 1, qubit) == Id(qubit)
    assert Circuit.spiders(0, 1, qubit ** 2) == ((sqrt(2) >> Ket(0) >> H)
                                                 @ (sqrt(2) >> Ket(0) >> H))

    mul2 = Circuit(dom=qubit @ qubit @ qubit @ qubit,
                   cod=qubit @ qubit,
                   boxes=[SWAP, CX, Bra(0), CX, Bra(0)],
                   offsets=[1, 0, 1, 1, 2])
    assert Circuit.spiders(2, 1, qubit ** 2) == mul2

    ghz2 = Circuit(dom=Ty(),
                   cod=qubit @ qubit @ qubit @ qubit @ qubit @ qubit,
                   boxes=[sqrt(2), Ket(0), H, Ket(0), CX, Ket(0), CX,
                          sqrt(2), Ket(0), H, Ket(0), CX, Ket(0), CX,
                          SWAP, SWAP, SWAP],
                   offsets=[0, 0, 0, 1, 0, 1, 0, 3, 3, 3, 4, 3, 4,
                            3, 2, 1, 3])
    assert Circuit.spiders(0, 3, qubit ** 2) == ghz2

    assert np.abs(Circuit.spiders(0, 0, qubit).eval().array) == 2

    combos = [(2, 3), (5, 4), (0, 1)]
    for n_legs_in, n_legs_out in combos:
        flat_tensor = np.abs(Circuit.spiders(n_legs_in, n_legs_out, qubit)
                             .eval().array.flatten())
        assert flat_tensor[0] == flat_tensor[-1] == 1


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
        '.scale(4)'
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


def test_bra_ket_inputs():
    bad_inputs = ['0', '1', 2]
    for box in [Bra, Ket]:
        for bad_input in bad_inputs:
            with raises(Exception):
                box(bad_input)


def test_Circuit_from_tk():
    def back_n_forth(f):
        return Circuit.from_tk(f.to_tk())

    m = Measure(1, destructive=False, override_bits=True)
    assert back_n_forth(m) == m.init_and_discard()
    assert back_n_forth(CRx(0.5)) ==\
        Ket(0) @ Ket(0) >> CRx(0.5) >> Discard() @ Discard()
    assert back_n_forth(CRz(0.5)) ==\
        Ket(0) @ Ket(0) >> CRz(0.5) >> Discard() @ Discard()
    assert Id(qubit @ bit).init_and_discard()\
        == back_n_forth(Swap(qubit, bit)) == back_n_forth(Swap(bit, qubit))


def test_ClassicalGate_to_tk():
    post = ClassicalGate('post', 2, 0, data=[0, 0, 0, 1])
    assert (post[::-1] >> Swap(bit, bit)).to_tk().post_processing\
        == post[::-1] >> Swap(bit, bit)
    circuit = sqrt(2) @ Ket(0, 0) >> H @ Rx(0) >> CX >> Measure(2) >> post
    assert Circuit.from_tk(circuit.to_tk())[-1] == post


def test_tk_dagger():
    assert S.dagger().to_tk() == tk.Circuit(1).Sdg(0)
    assert T.dagger().to_tk() == tk.Circuit(1).Tdg(0)


def test_Circuit_get_counts():
    assert Id(1).get_counts() == {(): 1.0}


def test_Circuit_get_counts_snake():
    compilation = Mock()
    compilation.apply = lambda x: x
    backend = tk.mockBackend({
        (0, 0): 240, (0, 1): 242, (1, 0): 271, (1, 1): 271})
    scaled_bell = Circuit.caps(qubit, qubit)
    snake = scaled_bell @ Id(1) >> Id(1) @ scaled_bell[::-1]
    result = np.round(snake.eval(
        backend, compilation=compilation, measure_all=True).array)
    assert result == 1


def test_QuantumGate_repr():
    assert repr(Y.l) == repr(Y.r) == "Y.conjugate()"
    assert repr(Z.l) == repr(Z) == repr(Z.r) == "Z"
    assert repr(S.dagger()) == "S.dagger()"
    assert repr(S.l.dagger()) == "S.conjugate().dagger()"


def test_Circuit_conjugate():
    assert (Rz(0.1) >> H).conjugate() == Rz(-0.1) >> H


def test_Circuit_get_counts_empty():
    backend = tk.mockBackend({})
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
        ".scale(4)"


def test_ClassicalGate_eval():
    backend = tk.mockBackend({
        (0, 0): 256, (0, 1): 256, (1, 0): 256, (1, 1): 256})
    post = ClassicalGate('post', 2, 0, [1, 0, 0, 0])
    assert post.eval(backend) == Tensor(dom=Dim(1), cod=Dim(1), array=[0.25])


def test_Box():
    with raises(TypeError):
        Box('f', rigid.Ty('x'), bit)
    with raises(TypeError):
        Box('f', bit, rigid.Ty('x'))


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
        == "QuantumGate('s', n_qubits=0, array=[1.+0.j])"


def test_ClassicalGate():
    f = ClassicalGate('f', 1, 1, [0, 1, 1, 0])
    assert repr(f.dagger())\
        == "ClassicalGate('f', bit, bit, data=[0, 1, 1, 0]).dagger()"


def test_Digits():
    with raises(TypeError):
        Digits()
    d = Digits(0, 1, 2, dim=3)
    assert d.digits == [0, 1, 2]
    assert d.dagger().dagger() == d


def test_Bits():
    assert repr(Bits(0).dagger()) == "Bits(0).dagger()"
    assert Bits(0).dagger().dagger() == Bits(0)


def test_Rx():
    assert repr(Rx(0.4)) == "Rx(0.4)"
    assert Rx(0).eval() == Rx(0).dagger().eval() == Id(1).eval()


def test_Ry():
    assert Ry(0).eval() == Id(1).eval()


def test_Rz():
    assert Rz(0).eval() == Id(1).eval()


def test_CRz():
    assert CRz(0).eval() == Id(2).eval()


def test_CU1():
    assert CU1(0).eval() == Id(2).eval()


def test_CRx():
    assert CRx(0).eval() == Id(2).eval()


def test_CircuitFunctor():
    x, y = rigid.Ty('x'), rigid.Ty('y')
    f = rigid.Box('f', x, y)
    ob, ar = {x: qubit, y: bit}, {f: Measure()}
    assert repr(Functor(ob, ar))\
        == "circuit.Functor(ob={Ty('x'): qubit, Ty('y'): bit}, "\
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
    assert Sum([Id(1)]).eval() == Id(1).eval()


def test_subs():
    from sympy.abc import phi
    assert (Rz(phi) + Rz(phi + 1)).subs(phi, 1) == Rz(1) + Rz(2)
    circuit = sqrt(2) @ Ket(0, 0) >> H @ Rx(phi) >> CX >> Bra(0, 1)
    assert circuit.subs(phi, 0.5)\
        == sqrt(2) @ Ket(0, 0) >> H @ Rx(0.5) >> CX >> Bra(0, 1)


def test_lambdify():
    from sympy.abc import phi
    assert list(Rz(phi).lambdify(phi)(0).array.flatten()) == [1, 0, 0, 1]


def _to_square_mat(m):
    m = np.asarray(m).flatten()
    return m.reshape(2 * (int(np.sqrt(len(m))), ))


def test_grad_basic():
    from sympy.abc import phi
    assert Rz(0).grad(phi).eval() == 0
    assert CU1(1).grad(phi).eval() == 0
    assert CRz(0).grad(phi).eval() == 0
    assert CRx(1).grad(phi).eval() == 0

    assert scalar(2 * phi).grad(phi).eval() == 2
    assert scalar(1.23).grad(phi).eval() == 0
    assert (scalar(2 * phi) + scalar(3 * phi)).grad(phi).eval() == 5

    assert Measure().grad(phi).eval() == 0
    with raises(NotImplementedError):
        Box("dummy box", qubit, qubit, data=phi).grad(phi)


def _assert_is_close_to_iden(m):
    m = _to_square_mat(m)
    assert np.isclose(np.linalg.norm(m - np.eye(len(m))), 0)


def _assert_is_close_to_0(m):
    if isinstance(m, Circuit):
        assert m.dom == m.cod
        m = m.eval().array
    m = np.asarray(m).flatten()
    assert np.isclose(np.linalg.norm(m), 0)


def test_testing_utils():
    for k in range(1, 4):
        _assert_is_close_to_iden(np.eye(k))
        _assert_is_close_to_iden(Id(k))
        _assert_is_close_to_0(np.zeros(k))
        _assert_is_close_to_0(Ket(*([0] * k)) >> Bra(*([1] * k)))


def test_rot_grad():
    from sympy import pi
    from sympy.abc import phi
    x = 0.7
    crz_diff = CRz(phi).grad(phi, mixed=False).lambdify(phi)(x).eval()
    crz_res = (
        (CRz(phi) >> Z @ Z @ scalar(0.5j * pi))
        + (CRz(phi) >> Id(1) @ Z @ scalar(-0.5j * pi))
    ).lambdify(phi)(x).eval()
    assert np.allclose(crz_diff, crz_res)

    crx_diff = CRx(phi).grad(phi, mixed=False).lambdify(phi)(x).eval()
    crx_res = (
        (CRx(phi) >> Z @ X @ scalar(0.5j * pi))
        + (CRx(phi) >> Id(1) @ X @ scalar(-0.5j * pi))
    ).lambdify(phi)(x).eval()
    assert np.allclose(crx_diff, crx_res)


def test_rot_grad_NotImplemented():
    from sympy.abc import z
    with raises(NotImplementedError):
        CU1(z).grad(z, mixed=True)


def test_ClassicalGate_grad_subs():
    from sympy.abc import x, y
    s = ClassicalGate('s', 0, 0, [x])
    assert s.grad(x) and not s.subs(x, y).grad(x)


def test_Copy_Match():
    assert Match().dagger() == Copy() and Copy().dagger() == Match()


def test_Sum_get_counts():
    assert Sum([], qubit, qubit).get_counts() == {}


def sy_cx(c, t, n):
    """
    A sympy CX factory with arbitrary control and target wires.
    The returned function accepts a tuple of integers (representing the
    bits state) or a binary string. The input is assumed in big endian
    ordering.
    :param c: The index of the control bit.
    :param t: The index of the target bit.
    :param n: The total number of bits.
    """
    assert c != t
    assert c in range(n)
    assert t in range(n)

    import sympy as sy
    x = list(sy.symbols(f'x:{n}'))
    x[t] = (x[c] + x[t]) % 2
    x = sy.Array(x)

    def f(v):
        v = map(int, list(v)) if isinstance(v, str) else v
        v = x.subs(zip(sy.symbols(f'x:{n}'), v))
        return tuple(v)
    return f


def verify_rewire_cx_case(c, t, n):
    ext_cx = partial(rewire, CX)
    op = ext_cx(c, t, dom=qubit**n)
    cx1 = sy_cx(c, t, n)

    for k in range(2**n):
        v = format(k, 'b').zfill(n)
        v = tuple(map(int, list(v)))
        # <f(i)| CX_{c, t} |i>, where f is the classical
        # implementation of CX_{c, t}.
        c = Ket(*v) >> op >> Bra(*cx1(v))
        assert np.isclose(c.eval().array, 1)


def test_rewire():
    ext_cx = partial(rewire, CX)

    assert ext_cx(0, 1) == CX
    assert ext_cx(1, 0) == (SWAP >> CX >> SWAP)
    assert ext_cx(0, 1, dom=qubit**2) == CX
    assert ext_cx(2, 1) == Id(1) @ (SWAP >> CX >> SWAP)
    assert rewire(CZ, 1, 2) == Id(1) @ CZ
    assert rewire(Id(2), 1, 0) == SWAP >> SWAP
    assert rewire(Circuit.cups(qubit, qubit), 1, 2).cod == qubit

    with raises(NotImplementedError):
        # Case cod != qubit**2 and non-contiguous rewiring
        rewire(Circuit.cups(qubit, qubit), 0, 2)
    with raises(ValueError):
        # Case dom != qubit**2
        rewire(X, 1, 2)
    with raises(ValueError):
        ext_cx(0, 0)
    with raises(ValueError):
        ext_cx(0, 1, dom=qubit**0)

    for params in [(0, 2, 3), (2, 0, 3)]:
        verify_rewire_cx_case(*params)


def test_real_amp_ansatz():
    rys_layer = (Ry(0) @ Ry(0))
    step = CX >> rys_layer

    for entg in ('full', 'linear'):
        c = rys_layer >> step
        assert real_amp_ansatz(np.zeros((2, 2)), entanglement=entg) == c
        c = rys_layer >> step >> step
        assert real_amp_ansatz(np.zeros((3, 2)), entanglement=entg) == c

    step = (SWAP >> CX >> SWAP) >> CX >> (Ry(0) @ Ry(0))
    c = rys_layer >> step
    assert real_amp_ansatz(np.zeros((2, 2)), entanglement='circular') == c
    c = rys_layer >> step >> step
    assert real_amp_ansatz(np.zeros((3, 2)), entanglement='circular') == c


def test_Controlled():
    with raises(TypeError):
        Controlled(None)
    with raises(ValueError):
        Controlled(X, distance=0)


def test_adjoint():
    n, s = map(rigid.Ty, 'ns')
    Bob = rigid.Box('Bob', rigid.Ty(), n)
    eats = rigid.Box('eats', rigid.Ty(), n.r @ s)
    diagram = Bob @ eats >> rigid.Cup(n, n.r) @ rigid.Id(s)

    func_ob = {n: qubit, s: qubit}
    func_ar = {Bob: Ket(0), eats: Ket(1, 1)}
    F = Functor(ob=func_ob, ar=func_ar)

    assert F(diagram.transpose_box(0, left=True).normal_form()) == Circuit(
        dom=Ty(), cod=qubit, boxes=[Ket(1, 1), Bra(0)], offsets=[0, 0])

    gates = [
        Bra(0), Ket(0, 0), Rx(0.1), Ry(0.2), Rz(0.3),
        CU1(0.4), CRx(0.5), CRz(0.7), Scalar(1 + 2j), CX,
        Swap(bit, qubit), Copy(), Match()
    ]

    gates_conj = [
        Bra(0), Ket(0, 0), Rx(-0.1), Ry(0.2), Rz(-0.3),
        Swap(qubit, qubit) >> CU1(-0.4) >> Swap(qubit, qubit),
        Controlled(Rx(-0.5), distance=-1),
        Controlled(Rz(-0.7), distance=-1),
        Scalar(1 - 2j),
        Controlled(X, distance=-1),
        Swap(qubit, bit), Copy(), Match()
    ]

    for g, g_conj in zip(gates, gates_conj):
        assert g.conjugate() == g_conj


def test_causal_cx():
    assert np.allclose((CX >> Discard(2)).eval(), Discard(2).eval())


def test_eval_no_qutrits():
    qutrit = Ty(Qudit(3))
    with raises(Exception):
        Box('qutrit box', qutrit, qutrit).to_tn(mixed=True)


def test_grad_unknown_controlled():
    from sympy.abc import phi
    unknown = QuantumGate('gate', 1, data=phi)
    with raises(NotImplementedError):
        Controlled(unknown).grad(phi)


def test_symbolic_controlled():
    from sympy.abc import phi
    crz = lambda x, d: Controlled(Rz(x), distance=d)
    assert np.all(
        crz(phi, -1).eval().array
        == (SWAP >> crz(phi, 1) >> SWAP).eval().array)


def test_controlled_subs():
    from sympy.abc import phi, psi
    assert CRz(phi).subs(phi, 0.1) == CRz(0.1)
    assert CRx(psi).l.subs((phi, 0.1), (psi, 0.2)) == CRx(0.2).l


def test_circuit_chaining():
    circuit = Id(5).CX(0, 2).X(4).CRz(0.2, 4, 2).H(2)
    expected_circuit = Circuit(
        dom=qubit @ qubit @ qubit @ qubit @ qubit,
        cod=qubit @ qubit @ qubit @ qubit @ qubit,
        boxes=[
            Controlled(X, distance=2), X, Controlled(Rz(0.2), distance=-2), H],
        offsets=[0, 4, 2, 2])
    assert circuit == expected_circuit

    circuit = Id(3).CZ(0, 1).CX(2, 0).CCX(1, 0, 2).CCZ(2, 0, 1).X(0).Y(1).Z(2)
    expected_circuit = Circuit(
        dom=qubit @ qubit @ qubit, cod=qubit @ qubit @ qubit,
        boxes=[
            CZ, Controlled(X, distance=-2), Controlled(CX, distance=1),
            Controlled(CZ, distance=-1), X, Y, Z],
        offsets=[0, 0, 0, 0, 0, 1, 2])
    assert circuit == expected_circuit

    circuit = Id(1).Rx(0.1, 0).Ry(0.2, 0).Rz(0.3, 0)
    expected_circuit = Rx(0.1) >> Ry(0.2) >> Rz(0.3)
    assert circuit == expected_circuit

    assert Id(1).S(0) == S

    with raises(ValueError):
        Id(3).CY(0, 0)
    with raises(ValueError):
        Id(1).CRx(0.7, 1, 0)
    with raises(ValueError):
        Id(2).X(999)
