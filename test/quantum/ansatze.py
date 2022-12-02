import numpy as np

from discopy.quantum.ansatze import IQPansatz, Sim14ansatz, Sim15ansatz, real_amp_ansatz


def test_IQPAnsatz():
    with raises(ValueError):
        IQPansatz(10, np.array([]))


def test_Sim14Ansatz():
    with raises(ValueError):
        Sim14ansatz(10, np.array([]))


def test_Sim15Ansatz():
    with raises(ValueError):
        Sim15ansatz(10, np.array([]))


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
