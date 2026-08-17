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


def rydberg_hamiltonian(positions, omega, delta, phase, coupling):
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    number = np.array([[0, 0], [0, 1]])
    drive = omega / 2 * (np.cos(phase) * sigma_x - np.sin(phase) * sigma_y)\
        - delta * number
    place = lambda op, i: np.kron(np.kron(
        np.eye(2 ** i), op), np.eye(2 ** (len(positions) - i - 1)))
    result = sum(place(drive, i) for i in range(len(positions)))
    for i, j in [(i, j) for i in range(len(positions))
                 for j in range(i + 1, len(positions))]:
        distance = np.linalg.norm(
            np.array(positions[i]) - np.array(positions[j]))
        result = result + coupling / distance ** 6\
            * place(number, i) @ place(number, j)
    return result


def rydberg_exact(positions, duration, omega, delta, phase, coupling):
    eigenvalues, eigenvectors = np.linalg.eigh(rydberg_hamiltonian(
        positions, omega, delta, phase, coupling))
    return eigenvectors @ np.diag(
        np.exp(-1j * duration * eigenvalues)) @ eigenvectors.conj().T


def circuit_matrix(circuit):
    dim = 2 ** len(circuit.dom)
    return circuit.eval().array.reshape(dim, dim).T


def test_Rydberg_diagonal():
    positions, duration, delta, coupling = [(0, 0), (1, 0)], 0.5, 2., 64.
    circuit = Rydberg(positions, duration, 0, delta, coupling=coupling)
    assert np.allclose(circuit_matrix(circuit), rydberg_exact(
        positions, duration, 0, delta, 0, coupling))


def test_Rydberg_drive():
    positions, duration, omega, phase = [(0, 0)], 0.7, 2.5, 1.2
    circuit = Rydberg(positions, duration, omega, 0, phase=phase)
    assert np.allclose(circuit_matrix(circuit), rydberg_exact(
        positions, duration, omega, 0, phase, 1))


def test_Rydberg_trotter():
    from discopy.quantum.circuit import Id, qubit
    positions, duration = [(0, 0), (0, 7)], 0.1
    omega, delta, coupling = 4., 3., 5420158.53
    exact = rydberg_exact(positions, duration, omega, delta, 0, coupling)
    error = lambda steps: np.abs(exact - np.linalg.matrix_power(
        circuit_matrix(Rydberg(positions, duration / steps, omega, delta)),
        steps)).max()
    assert error(100) < 1e-2 < error(1)
    assert error(100) < error(10) < error(1)
    step = Rydberg(positions, duration / 10, omega, delta)
    assert Rydberg(positions, duration, omega, delta, steps=10)\
        == Id(qubit ** 2).then(*10 * [step])


def test_Rydberg_errors():
    with raises(ValueError):
        Rydberg([(0, 0)], 1, [0.1, 0.2, 0.3], 0, steps=2)
    with raises(ValueError):
        Rydberg([(0, 0), (0, 0)], 1, 0, 0)
