# -*- coding: utf-8 -*-

import numpy as np
from pytest import raises

from discopy.quantum.ansatze import (
    IQPansatz, Sim14ansatz, Sim15ansatz, Rydberg)


def test_IQPAnsatz():
    with raises(ValueError):
        IQPansatz(10, np.array([]))


def test_Sim14Ansatz():
    with raises(ValueError):
        Sim14ansatz(10, np.array([]))


def test_Sim15Ansatz():
    with raises(ValueError):
        Sim15ansatz(10, np.array([]))


def rydberg_hamiltonian(positions, omega, delta, phase, detunings=None):
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    number = np.array([[0, 0], [0, 1]])
    detunings = len(positions) * [0] if detunings is None else detunings
    drive = omega / 2 * (np.cos(phase) * sigma_x + np.sin(phase) * sigma_y)
    place = lambda op, i: np.kron(np.kron(
        np.eye(2 ** i), op), np.eye(2 ** (len(positions) - i - 1)))
    result = sum(place(drive - (delta + d) * number, i)
                 for i, d in enumerate(detunings))
    for i, j in [(i, j) for i in range(len(positions))
                 for j in range(i + 1, len(positions))]:
        distance = np.linalg.norm(
            np.array(positions[i]) - np.array(positions[j]))
        result = result + 1 / distance ** 6\
            * place(number, i) @ place(number, j)
    return result


def rydberg_exact(positions, duration, omega, delta, phase, detunings=None):
    eigenvalues, eigenvectors = np.linalg.eigh(rydberg_hamiltonian(
        positions, omega, delta, phase, detunings))
    return eigenvectors @ np.diag(
        np.exp(-1j * duration * eigenvalues)) @ eigenvectors.conj().T


def circuit_matrix(circuit):
    dim = 2 ** len(circuit.dom)
    return circuit.eval().array.reshape(dim, dim).T


def test_Rydberg_diagonal():
    positions, duration, delta = [(0, 0), (.5, 0)], 0.5, 2.
    circuit = Rydberg(positions, duration, 0, delta)
    assert np.allclose(circuit_matrix(circuit), rydberg_exact(
        positions, duration, 0, delta, 0))


def test_Rydberg_drive():
    positions, duration, omega, phase = [(0, 0)], 0.7, 2.5, 1.2
    circuit = Rydberg(positions, duration, omega, 0, phase=phase)
    assert np.allclose(circuit_matrix(circuit), rydberg_exact(
        positions, duration, omega, 0, phase))


def test_Rydberg_list_waveform():
    positions, duration = [(0, 0)], 0.7
    omega, phase = [1.5, 2.5], [0.3, 1.2]
    steps, dt = len(omega), duration / len(omega)
    circuit = Rydberg(positions, duration, omega, 0, phase=phase, steps=steps)
    exact = rydberg_exact(positions, dt, omega[1], 0, phase[1])\
        @ rydberg_exact(positions, dt, omega[0], 0, phase[0])
    assert np.allclose(circuit_matrix(circuit), exact)


def test_Rydberg_trotter():
    from discopy.quantum.circuit import Id, qubit
    positions, duration = [(0, 0), (0, .5)], 0.1
    omega, delta = 4., 3.
    exact = rydberg_exact(positions, duration, omega, delta, 0)
    error = lambda steps: np.abs(exact - np.linalg.matrix_power(
        circuit_matrix(Rydberg(positions, duration / steps, omega, delta)),
        steps)).max()
    assert error(100) < 1e-2 < error(1)
    assert error(100) < error(10) < error(1)
    step = Rydberg(positions, duration / 10, omega, delta)
    assert Rydberg(positions, duration, omega, delta, steps=10)\
        == Id(qubit ** 2).then(*10 * [step])


def test_Rydberg_dmm():
    positions, duration, delta = [(0, 0), (1, 0), (0, 1)], 0.5, 1.
    weights, dmm = [0, .5, 1], -2.
    circuit = Rydberg(positions, duration, 0, delta, weights=weights, dmm=dmm)
    assert np.allclose(circuit_matrix(circuit), rydberg_exact(
        positions, duration, 0, delta, 0, [w * dmm for w in weights]))
    omega, dmm, dt = 2., [-1., -3.], duration / 2
    circuit = Rydberg(positions, duration, omega, delta,
                      steps=2, weights=weights, dmm=dmm)
    first, second = (Rydberg(positions, dt, omega, delta,
                             weights=weights, dmm=sample) for sample in dmm)
    assert circuit == first >> second
    assert circuit != second >> first


def test_Rydberg_errors():
    with raises(ValueError):
        Rydberg([(0, 0)], 1, [0.1, 0.2, 0.3], 0, steps=2)
    with raises(ValueError):
        Rydberg([(0, 0), (0, 0)], 1, 0, 0)
    with raises(ValueError):
        Rydberg([(0, 0)], 1, 0, 0, steps=0)
    with raises(ValueError):
        Rydberg([(0, 0), (0, 1, 2)], 1, 0, 0)
    with raises(ValueError):
        Rydberg([(0, 0)], 1, 0, 0, weights=[0, 1])
    with raises(ValueError):
        Rydberg([(0, 0)], 1, 0, 0, weights=[2])
    with raises(ValueError):
        Rydberg([(0, 0)], 1, 0, 0, weights=[1], dmm=1)
    with raises(ValueError):
        Rydberg([(0, 0)], 1, 0, 0, weights=[1], dmm=float("nan"))
    with raises(ValueError):
        Rydberg([(0, 0)], 1, 0, 0, weights=[float("nan")])


def test_Rydberg_steps():
    from discopy.quantum import Rydberg as exported
    assert exported is Rydberg
    with raises(ValueError):
        Rydberg([(0, 0)], 1, 1, 0, steps=True)
