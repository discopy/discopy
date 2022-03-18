# -*- coding: utf-8 -*-


import numpy as np
import tensornetwork as tn

from discopy.quantum import (
    Circuit, IQPansatz,
    Bra, Copy, CRz, Encode, Id, Ket, Rx, Rz, Match, Measure,
    MixedState, Discard, bit, sqrt, CX, H, SWAP, X, Y, Z)

mixed_circuits = [
    (Copy() >> Encode(2) >> CX >> Rx(0.3) @ Rz(0.3)
        >> SWAP >> Measure() @ Discard()),
    (Copy() @ Id(bit) >> Id(bit @ bit) @ Copy() >> Encode(4)
        >> H @ Id(3) >> Id(1) @ CX @ Id(1) >> Id(3) @ Rz(0.4)
        >> CX @ Id(2) >> Id(2) @ CX >> Rx(0.3) @ Id(3)
        >> Id(1) @ CX @ Id(1) >> Id(3) @ H >> Measure(4)),
    Ket(0, 0, 0, 0) >> Discard(2) @ Measure(2) @ sqrt(2),
    Circuit.swap(bit, bit) @ (MixedState(2) >> SWAP),
    Measure() >> Copy() >> Match() >> Encode()
]

pure_circuits = [
    H >> X >> Y >> Z,
    CX >> H @ Rz(0.5),
    CRz(0.123) >> Z @ Z,
    CX >> H @ Id(1) >> Bra(0, 0),
    IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]),
    IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]).l.dagger(),
    Circuit.permutation([1, 2, 0])
]

contractor = tn.contractors.auto


def test_mixed_eval():
    for c in mixed_circuits:
        assert np.allclose(c.eval(contractor=contractor), c.eval())


def test_pure_eval():
    for c in pure_circuits:
        assert np.allclose(c.eval(contractor=contractor), c.eval())


def test_consistent_eval():
    for c in pure_circuits:
        pure_result = c.eval(mixed=False, contractor=contractor)
        mixed_result = c.eval(mixed=True, contractor=contractor)

        doubled_result = (pure_result
                          @ pure_result.conjugate(diagrammatic=False))
        np.allclose(doubled_result, mixed_result)


def test_pytorch_mixed_eval():
    with tn.DefaultBackend('pytorch'):
        for c in mixed_circuits:
            assert np.allclose(c.eval(contractor=contractor), c.eval())


def test_pytorch_pure_eval():
    with tn.DefaultBackend('pytorch'):
        for c in pure_circuits:
            assert np.allclose(c.eval(contractor=contractor), c.eval())


def test_pytorch_consistent_eval():
    with tn.DefaultBackend('pytorch'):
        for c in pure_circuits:
            pure_result = c.eval(mixed=False, contractor=contractor)
            mixed_result = c.eval(mixed=True, contractor=contractor)

            doubled_result = (
                pure_result
                @ pure_result.conjugate(diagrammatic=False))
            np.allclose(doubled_result, mixed_result)
