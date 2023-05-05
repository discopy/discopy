# -*- coding: utf-8 -*-


import pytest
import tensornetwork as tn

from discopy.quantum import (
    Circuit, IQPansatz,
    Bra, Copy, CRz, Encode, Id, Ket, Rx, Rz, Match, Measure,
    MixedState, Discard, bit, qubit, sqrt, CX, H, SWAP, X, Y, Z)

mixed_circuits = [
    (Copy() >> Encode(2) >> CX >> Rx(0.3) @ Rz(0.3)
        >> SWAP >> Measure() @ Discard()),
    (Copy() @ Id(bit) >> Id(bit @ bit) @ Copy() >> Encode(4)
        >> H @ qubit ** 3 >> qubit @ CX @ qubit >> qubit ** 3 @ Rz(0.4)
        >> CX @ qubit ** 2 >> qubit ** 2 @ CX >> Rx(0.3) @ qubit ** 3
        >> qubit @ CX @ qubit >> qubit ** 3 @ H >> Measure(4)),
    Ket(0, 0, 0, 0) >> Discard(2) @ Measure(2) @ sqrt(2),
    Circuit.swap(bit, bit) @ (MixedState(2) >> SWAP),
    Measure() >> Copy() >> Match() >> Encode()
]

pure_circuits = [
    H >> X >> Y >> Z,
    CX >> H @ Rz(0.5),
    CRz(0.123) >> Z @ Z,
    CX >> H @ qubit >> Bra(0, 0),
    IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]),
    IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]).l.dagger(),
    Circuit.permutation([1, 2, 0])
]

contractor = tn.contractors.auto

def is_close_smallno(a, b):
    return a.is_close(b, rtol=1.e-15, atol=1.e-15) 


@pytest.mark.parametrize('c', pure_circuits + mixed_circuits)
def test_mixed_eval(c):
    assert is_close_smallno(c.eval(contractor=contractor), c.eval())


@pytest.mark.parametrize('c', pure_circuits)
def test_consistent_eval(c):
    pure_result = c.eval(mixed=False, contractor=contractor)
    mixed_result = c.eval(mixed=True, contractor=contractor)

    doubled_result = (pure_result
                      @ pure_result.conjugate(diagrammatic=False))
    assert is_close_smallno(doubled_result, mixed_result.to_tensor())


@pytest.mark.parametrize('c', mixed_circuits)
def test_pytorch_mixed_eval(c):
    with tn.DefaultBackend('pytorch'):
        assert is_close_smallno(c.eval(contractor=contractor), c.eval())


@pytest.mark.parametrize('c', pure_circuits)
def test_pytorch_pure_eval(c):
    with tn.DefaultBackend('pytorch'):
        assert is_close_smallno(c.eval(contractor=contractor), c.eval())


@pytest.mark.parametrize('c', pure_circuits)
def test_pytorch_consistent_eval(c):
    with tn.DefaultBackend('pytorch'):
        pure_result = c.eval(mixed=False, contractor=contractor)
        mixed_result = c.eval(mixed=True, contractor=contractor)

        doubled_result = (
            pure_result
            @ pure_result.conjugate(diagrammatic=False))
        assert is_close_smallno(doubled_result, mixed_result.to_tensor())
