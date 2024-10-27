
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from discopy.quantum import (
    Circuit, IQPansatz,
    Bra, Copy, CRz, Encode, Id, Ket, Rx, Rz, Match, Measure,
    MixedState, Discard, bit, qubit, sqrt, CX, H, SWAP, X, Y, Z)


pure_circuits = [
    H >> X >> Y >> Z,
    CX >> H @ Rz(0.5),
    CRz(0.123) >> Z @ Z,
    CX >> H @ qubit >> Bra(0, 0),
    IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]),
    IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]).l.dagger(),
    Circuit.permutation([1, 2, 0])
]

@pytest.mark.parametrize('c', pure_circuits)
def test_quimb_pure_eval(c):
    print(c)
    t = c.to_quimb().contract()
    t = t.data.transpose(*np.argsort(t.inds))

    assert np.allclose(t, c.eval().array), f"{t} != {c.eval().array}"