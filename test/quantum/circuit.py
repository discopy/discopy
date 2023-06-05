# -*- coding: utf-8 -*-

import pytest

import numpy as np
import sympy
import torch
from pytest import raises

from discopy.quantum.channel import *
from discopy.quantum.circuit import *
from discopy.quantum.gates import *
from discopy import rigid


def test_index2bitstring():
    with raises(ValueError):
        index2bitstring(1, 0)
    assert index2bitstring(42, 8) == (0, 0, 1, 0, 1, 0, 1, 0)


def test_bitstring2index():
    assert bitstring2index((0, 0, 1, 0, 1, 0, 1, 0)) == 42


def test_Circuit_eval():
    with raises(ValueError):
        Box('f', qubit, qubit).eval()
    assert MixedState().eval() == Discard().eval().dagger()


def test_Circuit_cups_and_caps():
    assert Circuit.cups(bit, bit) == Match() >> Discard(bit)
    assert Circuit.caps(bit, bit) == MixedState(bit) >> Copy()


def test_Circuit_spiders():
    assert Circuit.spiders(123, 456, qubit ** 0) == Id()
    assert Circuit.spiders(0, 0, qubit) == (sqrt(2) >> Ket(0)
                                            >> H >> H
                                            >> Bra(0) >> sqrt(2))
    assert Circuit.spiders(1, 1, qubit) == Id(qubit)
    assert Circuit.spiders(0, 1, qubit ** 2) == ((sqrt(2) >> Ket(0) >> H)
                                                 @ (sqrt(2) >> Ket(0) >> H))

    assert Circuit.spiders(2, 1, qubit ** 2) == Circuit.decode(
        qubit @ qubit @ qubit @ qubit,
        zip([SWAP, CX, Bra(0), CX, Bra(0)], [1, 0, 1, 1, 2]))

    boxes = [
        sqrt(2), Ket(0), H, Ket(0), CX, Ket(0), CX,
        sqrt(2), Ket(0), H, Ket(0), CX, Ket(0), CX,
        SWAP, SWAP, SWAP]
    offsets = [0, 0, 0, 1, 0, 1, 0, 3, 3, 3, 4, 3, 4, 3, 2, 1, 3]
    ghz2 = Circuit.decode(Ty(), zip(boxes, offsets))
    assert Circuit.spiders(0, 3, qubit ** 2).eval() == ghz2.eval()

    assert np.isclose(np.abs(Circuit.spiders(0, 0, qubit).eval().array), 2)

    combos = [(2, 3), (5, 4), (0, 1)]
    for n_legs_in, n_legs_out in combos:
        flat_tensor = np.abs(Circuit.spiders(n_legs_in, n_legs_out, qubit)
                             .eval().array.flatten())
        assert np.isclose(flat_tensor[0], 1) and np.isclose(flat_tensor[-1], 1)


def test_Circuit_to_pennylane(capsys):
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(qubit) >> Bra(0) @ bell_effect)[::-1]
    p_snake = snake.to_pennylane()
    p_snake.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ───────╭●──H─┤0>\n"
         "1: ──H─╭●─╰X────┤0>\n"
         "2: ────╰X───────┤  State\n")

    assert np.allclose(p_snake.eval().numpy(), snake.eval().array)

    p_snake_prob = snake.to_pennylane(probabilities=True)
    snake_prob = (snake >> Measure())

    assert np.allclose(p_snake_prob.eval().numpy(), snake_prob.eval().array)

    no_open_snake = (bell_state @ Ket(0) >> Bra(0) @ bell_effect)[::-1]
    p_no_open_snake = no_open_snake.to_pennylane()
    p_no_open_snake.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ───────╭●──H─┤0>\n"
         "1: ──H─╭●─╰X────┤0>\n"
         "2: ────╰X───────┤0>\n")

    assert np.allclose(p_no_open_snake.eval().numpy(),
                       no_open_snake.eval().array)

    # probabilities should not be normalized if all wires are post-selected
    p_no_open_snake_prob = no_open_snake.to_pennylane(probabilities=True)

    assert np.allclose(p_no_open_snake_prob.eval().numpy(),
                       no_open_snake.eval().array)

    x, y, z = sympy.symbols('x y z')
    symbols = [x, y, z]
    weights = [torch.tensor([1.]), torch.tensor([2.]), torch.tensor([3.])]

    boxes = [
        Ket(0), Rx(0.552), Rz(x), Rx(0.917), Ket(0, 0, 0), H, H, H,
        CRz(0.18), CRz(y), CX, H, sqrt(2), Bra(0, 0), Ket(0),
        Rx(0.446), Rz(0.256), Rx(z), CX, H, sqrt(2), Bra(0, 0)]
    offsets = [
        0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 2, 3, 2, 0, 0, 0, 0, 0, 0, 1, 0]
    var_circ = Circuit.decode(Ty(), zip(boxes, offsets))

    p_var_circ = var_circ.to_pennylane()
    p_var_circ.draw(symbols, weights)

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ──RX(2.80)──RZ(1.61)──RX(18.85)─╭●──H─┤0>\n"
         "1: ──H────────╭●───────────────────╰X────┤0>\n"
         "2: ──H────────╰RZ(1.13)─╭●───────────────┤  State\n"
         "3: ──H──────────────────╰RZ(12.57)─╭●──H─┤0>\n"
         "4: ──RX(3.47)──RZ(6.28)──RX(5.76)──╰X────┤0>\n")

    var_f = var_circ.lambdify(*symbols)
    conc_circ = var_f(*[a.item() for a in weights])

    assert np.allclose(p_var_circ.eval(symbols, weights).numpy(),
                       conc_circ.eval().array)

    p_var_circ_prob = var_circ.to_pennylane(probabilities=True)
    conc_circ_prob = (conc_circ >> Measure())

    assert (np.allclose(p_var_circ_prob.eval(symbols, weights).numpy(),
                        conc_circ_prob.eval().array))


def test_PennyLaneCircuit_mixed_error():
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(qubit) >> Bra(0) @ bell_effect)[::-1]
    snake = (snake >> Measure())
    with raises(ValueError):
        snake.to_pennylane()


def test_PennylaneCircuit_draw(capsys):
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(qubit) >> Bra(0) @ bell_effect)[::-1]
    p_circ = snake.to_pennylane()
    p_circ.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ───────╭●──H─┤0>\n"
         "1: ──H─╭●─╰X────┤0>\n"
         "2: ────╰X───────┤  State\n")


def test_pennylane_ops():
    ops = [X, Y, Z, S, T, H, CX, CZ]

    for op in ops:
        disco = (Id().tensor(*([Ket(0)] * len(op.dom))) >> op).eval().array
        plane = op.to_pennylane().eval().numpy()

        assert np.allclose(disco, plane)


def test_pennylane_parameterized_ops():
    ops = [Rx, Ry, Rz, CRx, CRz]

    for op in ops:
        p_op = op(0.5)
        disco = (Id().tensor(*([Ket(0)] * len(p_op.dom))) >> p_op).eval().array
        plane = p_op.to_pennylane().eval().numpy()

        assert np.allclose(disco, plane, atol=10e-5)


def test_bra_ket_inputs():
    bad_inputs = ['0', '1', 2]
    for box in [Bra, Ket]:
        for bad_input in bad_inputs:
            with raises(Exception):
                box(bad_input)


def test_Circuit_get_counts():
    assert Id(qubit).get_counts() == {(): 1.0}


def test_Circuit_conjugate():
    assert (Rz(0.1) >> H).conjugate() == Rz(-0.1) >> H


def test_Circuit_measure():
    assert Id().measure() == 1
    assert all(Bits(0).measure(mixed=True) == np.array([1, 0]))


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
    assert Swap(bit, bit).eval(mixed=True) == Channel.swap(C(Dim(2)), C(Dim(2)))


def test_Discard():
    assert Discard().dagger() == MixedState()
    assert MixedState().dagger() == Discard()


def test_Measure():
    assert Measure(destructive=False, override_bits=True).dagger()\
        == Encode(constructive=False, reset_bits=True)
    assert Encode().dagger() == Measure()


def test_ClassicalGate():
    f = ClassicalGate('f', bit, bit, [0, 1, 1, 0])
    assert repr(f.dagger())\
        == "quantum.gates.ClassicalGate('f', "\
           "quantum.circuit.Ty(quantum.circuit.Digit(2)), "\
           "quantum.circuit.Ty(quantum.circuit.Digit(2)), "\
           "data=[0, 1, 1, 0]).dagger()"


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
    assert repr(Rx(0.4)) == "quantum.gates.Rx(0.4)"
    assert Rx(0).eval() == Rx(0).dagger().eval() == Id(qubit).eval()


def test_Ry():
    assert Ry(0).eval() == Id(qubit).eval()


def test_Rz():
    assert Rz(0).eval() == Id(qubit).eval()


def test_CRz():
    assert CRz(0).eval() == Id(qubit ** 2).eval()


def test_CU1():
    assert CU1(0).eval() == Id(qubit ** 2).eval()


def test_CRx():
    assert CRx(0).eval() == Id(qubit ** 2).eval()


def test_Sum():
    assert not Sum([], qubit, qubit).eval()
    assert Sum([Id()]).get_counts() == Id().get_counts()
    assert (Id() + Id()).get_counts()[()] == 2
    assert (Id() + Id()).eval() == Id().eval() + Id().eval()
    assert not (X + X).is_mixed and (X >> Bra(0) + Discard()).is_mixed
    assert (Discard() + Discard()).eval()\
        == Discard().eval() + Discard().eval()
    assert Sum([Id(qubit)]).eval() == Id(qubit).eval()


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

    assert scalar(2 * phi).grad(phi).eval().array == 2
    assert scalar(1.23).grad(phi).eval() == 0
    assert (scalar(2 * phi) + scalar(3 * phi)).grad(phi).eval().array == 5

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
        _assert_is_close_to_iden(Id(qubit ** k))
        _assert_is_close_to_0(np.zeros(k))
        _assert_is_close_to_0(Ket(*([0] * k)) >> Bra(*([1] * k)))


def test_rot_grad():
    from sympy import pi
    from sympy.abc import phi
    x = 0.7
    crz_diff = CRz(phi).grad(phi, mixed=False).lambdify(phi)(x).eval()
    crz_res = (
        (CRz(phi) >> Z @ Z @ scalar(0.5j * pi))
        + (CRz(phi) >> Id(qubit) @ Z @ scalar(-0.5j * pi))
    ).lambdify(phi)(x).eval()
    assert np.allclose(crz_diff.array, crz_res.array)

    crx_diff = CRx(phi).grad(phi, mixed=False).lambdify(phi)(x).eval()
    crx_res = (
        (CRx(phi) >> Z @ X @ scalar(0.5j * pi))
        + (CRx(phi) >> Id(qubit) @ X @ scalar(-0.5j * pi))
    ).lambdify(phi)(x).eval()
    assert np.allclose(crx_diff.array, crx_res.array)


def test_rot_grad_NotImplemented():
    from sympy.abc import z
    with raises(NotImplementedError):
        CU1(z).grad(z, mixed=True)


def test_Copy_Match():
    assert Match().dagger() == Copy() and Copy().dagger() == Match()


def test_Sum_get_counts():
    assert Sum([], qubit, qubit).get_counts() == {}


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
    F = rigid.Functor(ob=func_ob, ar=func_ar, cod=Category(Ty, Circuit))

    assert F(diagram.transpose_box(0, left=True).normal_form())\
        == Circuit.decode(Ty(), zip([Ket(1, 1), Bra(0)], [0, 0]))

    gates = [
        Bra(0), Ket(0, 0), Rx(0.1), Ry(0.2), Rz(0.3),
        CU1(0.4), CRx(0.5), CRz(0.7), Scalar(1 + 2j), CX,
        Swap(bit, qubit), Copy(), Match()
    ]

    gates_conj = [
        Bra(0), Ket(0, 0), Rx(-0.1), Ry(0.2), Rz(-0.3),
        Controlled(U1(-0.4), distance=-1),
        Controlled(Rx(-0.5), distance=-1),
        Controlled(Rz(-0.7), distance=-1),
        Scalar(1 - 2j),
        Controlled(X, distance=-1),
        Swap(qubit, bit), Copy(), Match()
    ]

    for g, g_conj in zip(gates, gates_conj):
        assert g.conjugate() == g_conj


def test_causal_cx():
    assert np.allclose((CX >> Discard(2)).eval().array,
                       Discard(2).eval().array)


def test_eval_no_qutrits():
    qutrit = Ty(Qudit(3))
    with raises(Exception):
        Box('qutrit box', qutrit, qutrit).to_tn(mixed=True)


def test_grad_unknown_controlled():
    from sympy.abc import phi
    unknown = QuantumGate('gate', qubit, qubit, data=phi)
    with raises(NotImplementedError):
        Controlled(unknown).grad(phi)


def test_controlled_subs():
    from sympy.abc import phi, psi
    assert CRz(phi).subs(phi, 0.1) == CRz(0.1)
    assert CRx(psi).l.subs((phi, 0.1), (psi, 0.2)) == CRx(0.2).l


def test_circuit_chaining():
    circuit = Id(qubit ** 5).CX(0, 2).X(4).CRz(0.2, 4, 2).H(2)
    boxes = [Controlled(X, distance=2), X, Controlled(Rz(0.2), distance=-2), H]
    offsets = [0, 4, 2, 2]
    expected_circuit = Circuit.decode(qubit ** 5, zip(boxes, offsets))
    assert circuit == expected_circuit

    circuit = Id(qubit ** 3).CZ(0, 1).CX(2, 0).CCX(1, 0, 2).CCZ(2, 0, 1).X(0).Y(1).Z(2)
    boxes = [
        CZ, Controlled(X, distance=-2), Controlled(CX, distance=1),
        Controlled(CZ, distance=-1), X, Y, Z]
    offsets = [0, 0, 0, 0, 0, 1, 2]
    assert circuit == Circuit.decode(qubit ** 3, zip(boxes, offsets))

    circuit = Id(qubit).Rx(0.1, 0).Ry(0.2, 0).Rz(0.3, 0)
    expected_circuit = Rx(0.1) >> Ry(0.2) >> Rz(0.3)
    assert circuit == expected_circuit

    assert Id(qubit).S(0) == S

    with raises(ValueError):
        Id(qubit ** 3).CY(0, 0)
    with raises(IndexError):
        Id(qubit).CRx(0.7, 1, 0)
    with raises(IndexError):
        Id(qubit ** 2).X(999)


@pytest.mark.parametrize('x,y', [(0, 1), (0, 2), (1, 0), (2, 0), (5, 0)])
def test_CX_decompose(x, y):
    n = abs(x - y) + 1
    binary_mat = np.eye(n, dtype=int)
    binary_mat[y] = np.bitwise_xor(binary_mat[x], binary_mat[y])

    N = 1 << n
    unitary_mat = np.zeros(shape=(N, N))
    for i in range(N):
        bits = index2bitstring(i, n)
        v = bitstring2index(binary_mat @ bits % 2)
        unitary_mat[i][v] = 1

    # take transpose because tensor axes follow diagrammatic order
    out = Id(n).CX(x, y).eval().array.reshape(N, N).T
    # but CX matrices are self transpose
    assert (out == out.T).all()
    assert (out == unitary_mat).all()


@pytest.mark.parametrize('x,y', [(0, 1), (0, 2), (1, 0), (2, 0), (5, 0)])
def test_CX_decompose(x, y):
    n = abs(x - y) + 1
    binary_mat = np.eye(n, dtype=int)
    binary_mat[y] = np.bitwise_xor(binary_mat[x], binary_mat[y])

    N = 1 << n
    unitary_mat = np.zeros(shape=(N, N))
    for i in range(N):
        bits = index2bitstring(i, n)
        v = bitstring2index(binary_mat @ bits % 2)
        unitary_mat[i][v] = 1

    # take transpose because tensor axes follow diagrammatic order
    out = Id(n).CX(x, y).eval().array.reshape(N, N).T
    # but CX matrices are self transpose
    assert (out == out.T).all()
    assert (out == unitary_mat).all()


@pytest.mark.parametrize('x,y, z', [(0, 1, 2), (0, 2, 4),
                                    (0, 4, 2), (4, 2, 0),
                                    (0, 4, 1), (4, 0, 1)])
def test_CCX_decompose(x, y, z):

    n = max(x, y, z) - min(x, y, z) + 1
    N = 1 << n

    unitary_mat = np.zeros(shape=(N, N))

    for i in range(N):
        bits = list(index2bitstring(i, n))
        bits[z] = (bits[x] & bits[y]) ^ bits[z]
        v = bitstring2index(bits)
        unitary_mat[i][v] = 1

    # take transpose because tensor axes follow diagrammatic order
    out = Id(n).CCX(x, y, z).eval().array.reshape(N, N).T

    np.set_printoptions(threshold=3000)

    print(unitary_mat.real)

    assert (out == unitary_mat).all()


def test_Circuit_to_pennylane(capsys):
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(1) >> Bra(0) @ bell_effect)[::-1]
    p_snake = snake.to_pennylane()
    p_snake.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ───────╭●──H─┤0>\n"
         "1: ──H─╭●─╰X────┤0>\n"
         "2: ────╰X───────┤  State\n")

    assert np.allclose(p_snake.eval().numpy(), snake.eval().array)

    p_snake_prob = snake.to_pennylane(probabilities=True)
    snake_prob = (snake >> Measure())

    assert np.allclose(p_snake_prob.eval().numpy(), snake_prob.eval().array)

    no_open_snake = (bell_state @ Ket(0) >> Bra(0) @ bell_effect)[::-1]
    p_no_open_snake = no_open_snake.to_pennylane()
    p_no_open_snake.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ───────╭●──H─┤0>\n"
         "1: ──H─╭●─╰X────┤0>\n"
         "2: ────╰X───────┤0>\n")

    assert np.allclose(p_no_open_snake.eval().numpy(),
                       no_open_snake.eval().array)

    # probabilities should not be normalized if all wires are post-selected
    p_no_open_snake_prob = no_open_snake.to_pennylane(probabilities=True)

    assert np.allclose(p_no_open_snake_prob.eval().numpy(),
                       no_open_snake.eval().array)

    x, y, z = sympy.symbols('x y z')
    symbols = [x, y, z]
    weights = [torch.tensor(1.), torch.tensor(2.), torch.tensor(3.)]

    var_circ = Circuit.decode(
        dom=qubit ** 0, boxes_and_offsets=zip(
            [Ket(0), Rx(0.552), Rz(x), Rx(0.917), Ket(0, 0, 0), H, H, H,
             CRz(0.18), CRz(y), CX, H, sqrt(2), Bra(0, 0), Ket(0),
             Rx(0.446), Rz(0.256), Rx(z), CX, H, sqrt(2), Bra(0, 0)],
            [0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 2,
             3, 2, 0, 0, 0, 0, 0, 0, 1, 0]))

    p_var_circ = var_circ.to_pennylane()
    p_var_circ.initialise_concrete_params(symbols, weights)
    p_var_circ.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ──RX(2.80)──RZ(1.61)──RX(18.85)─╭●──H─┤0>\n"
         "1: ──H────────╭●───────────────────╰X────┤0>\n"
         "2: ──H────────╰RZ(1.13)─╭●───────────────┤  State\n"
         "3: ──H──────────────────╰RZ(12.57)─╭●──H─┤0>\n"
         "4: ──RX(3.47)──RZ(6.28)──RX(5.76)──╰X────┤0>\n")

    var_f = var_circ.lambdify(*symbols)
    conc_circ = var_f(*[a.item() for a in weights])

    assert np.allclose(p_var_circ.eval().numpy(),
                       conc_circ.eval().array)

    p_var_circ_prob = var_circ.to_pennylane(probabilities=True)
    p_var_circ_prob.initialise_concrete_params(symbols, weights)
    conc_circ_prob = (conc_circ >> Measure())

    assert (np.allclose(p_var_circ_prob.eval().numpy(),
                        conc_circ_prob.eval().array))


def test_PennyLaneCircuit_mixed_error():
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(1) >> Bra(0) @ bell_effect)[::-1]
    snake = (snake >> Measure())
    with raises(ValueError):
        snake.to_pennylane()


def test_PennylaneCircuit_draw(capsys):
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(1) >> Bra(0) @ bell_effect)[::-1]
    p_circ = snake.to_pennylane()
    p_circ.draw()

    captured = capsys.readouterr()
    assert captured.out == \
        ("0: ───────╭●──H─┤0>\n"
         "1: ──H─╭●─╰X────┤0>\n"
         "2: ────╰X───────┤  State\n")


def test_pennylane_ops():
    ops = [X, Y, Z, S, T, H, CX, CZ]

    for op in ops:
        disco = (Id().tensor(*([Ket(0)] * len(op.dom))) >> op).eval().array
        plane = op.to_pennylane().eval().numpy()

        assert np.allclose(disco, plane)


def test_pennylane_parameterized_ops():
    ops = [Rx, Ry, Rz, CRx, CRz]

    for op in ops:
        p_op = op(0.5)
        disco = (Id().tensor(*([Ket(0)] * len(p_op.dom))) >> p_op).eval().array
        plane = p_op.to_pennylane().eval().numpy()

        assert np.allclose(disco, plane, atol=10e-5)


def test_pennylane_devices():
    bell_state = Circuit.caps(qubit, qubit)
    bell_effect = bell_state[::-1]
    snake = (bell_state @ Id(1) >> Bra(0) @ bell_effect)[::-1]

    # Honeywell backend only compatible when `probabilities=True`
    h_backend = {'backend': 'honeywell.hqs', 'device': 'H1-1E'}
    h_circ = snake.to_pennylane(probabilities=True, backend_config=h_backend)
    assert h_circ._device is not None
    with raises(ValueError):
        h_circ = snake.to_pennylane(backend_config=h_backend)

    # Device must be specified when using Honeywell backend
    h_backend_corrupt = {'backend': 'honeywell.hqs'}
    with raises(ValueError):
        h_circ = snake.to_pennylane(probabilities=True,
                                    backend_config=h_backend_corrupt)

    aer_backend = {'backend': 'qiskit.aer',
                   'device': 'aer_simulator_statevector'}
    aer_circ = snake.to_pennylane(backend_config=aer_backend)
    assert aer_circ._device is not None

    # `aer_simulator` is not compatible with state outputs
    aer_backend_corrupt = {'backend': 'qiskit.aer', 'device': 'aer_simulator'}
    with raises(ValueError):
        aer_circ = snake.to_pennylane(backend_config=aer_backend_corrupt)


def test_pennylane_uninitialized():
    x, y, z = sympy.symbols('x y z')
    var_circ = Circuit.decode(
        dom=qubit ** 0, boxes_and_offsets=zip(
            [Ket(0), Rx(0.552), Rz(x), Rx(0.917), Ket(0, 0, 0), H, H, H,
             CRz(0.18), CRz(y), CX, H, sqrt(2), Bra(0, 0), Ket(0),
             Rx(0.446), Rz(0.256), Rx(z), CX, H, sqrt(2), Bra(0, 0)],
            [0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 2,
             3, 2, 0, 0, 0, 0, 0, 0, 1, 0]))
    p_var_circ = var_circ.to_pennylane()

    with raises(ValueError):
        p_var_circ.draw()

    with raises(ValueError):
        p_var_circ.eval()


def test_pennylane_parameter_reference():
    x = sympy.symbols('x')
    p = torch.nn.Parameter(torch.tensor(1.))

    circ = Rx(x)
    p_circ = circ.to_pennylane()
    p_circ.initialise_concrete_params([x], [p])

    with torch.no_grad():
        p.add_(1.)

    assert p_circ._concrete_params[0][0] == p

    with torch.no_grad():
        p.add_(-2.)

    assert p_circ._concrete_params[0][0] == p


def test_pennylane_gradient_methods():
    x, y, z = sympy.symbols('x y z')
    symbols = [x, y, z]

    var_circ = Circuit.decode(
        dom=qubit ** 0, boxes_and_offsets=zip(
            [Ket(0), Rx(0.552), Rz(x), Rx(0.917), Ket(0, 0, 0), H, H, H,
             CRz(0.18), CRz(y), CX, H, sqrt(2), Bra(0, 0), Ket(0),
             Rx(0.446), Rz(0.256), Rx(z), CX, H, sqrt(2), Bra(0, 0)],
            [0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 2, 2,
             3, 2, 0, 0, 0, 0, 0, 0, 1, 0]))

    for diff_method in ['backprop', 'parameter-shift', 'finite-diff']:

        weights = [torch.tensor(1., requires_grad=True),
                   torch.tensor(2., requires_grad=True),
                   torch.tensor(3., requires_grad=True)]

        p_var_circ = var_circ.to_pennylane(probabilities=True,
                                           diff_method=diff_method)
        p_var_circ.initialise_concrete_params(symbols, weights)

        loss = p_var_circ.eval().norm(dim=0, p=2)
        loss.backward()
        assert weights[0].grad is not None

    for diff_method in ['backprop']:

        weights = [torch.tensor(1., requires_grad=True),
                   torch.tensor(2., requires_grad=True),
                   torch.tensor(3., requires_grad=True)]

        p_var_circ = var_circ.to_pennylane(probabilities=False,
                                           diff_method=diff_method)
        p_var_circ.initialise_concrete_params(symbols, weights)

        loss = p_var_circ.eval().norm(dim=0, p=2)
        loss.backward()
        assert weights[0].grad is not None
