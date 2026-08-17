# -*- coding: utf-8 -*-

"""
Quantum circuit ansätze.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    IQPansatz
    Sim14ansatz
    Sim15ansatz
    Rydberg

"""

from itertools import combinations
from math import pi

from discopy.matrix import get_backend
from discopy.quantum.circuit import qubit, Circuit, Id


def IQPansatz(n_qubits, params) -> Circuit:
    """
    Build an IQP ansatz on n qubits, if n = 1 returns an Euler decomposition.

    >>> pprint = lambda c: print(str(c.foliation()).replace(' >>', '\\n  >>'))
    >>> pprint(IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]))
    H @ H @ H
      >> CRz(0.1) @ qubit
      >> qubit @ CRz(0.2)
      >> H @ H @ H
      >> CRz(0.3) @ qubit
      >> qubit @ CRz(0.4)
    >>> print(IQPansatz(1, [0.3, 0.8, 0.4]))
    Rx(0.3) >> Rz(0.8) >> Rx(0.4)
    """
    from discopy.quantum.gates import H, Rx, Rz, CRz

    np = get_backend()

    def layer(thetas):
        hadamards = Id().tensor(*(n_qubits * [H]))
        rotations = Id(qubit ** n_qubits).then(*(
            qubit ** i @ CRz(thetas[i]) @ qubit ** (n_qubits - 2 - i)
            for i in range(n_qubits - 1)))
        return hadamards >> rotations
    if n_qubits == 1:
        circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
    elif len(np.shape(params)) != 2\
            or np.shape(params)[1] != n_qubits - 1:
        raise ValueError(
            f"Expected params of shape (depth, {n_qubits - 1})")
    else:
        depth = np.shape(params)[0]
        circuit = Id(qubit ** n_qubits).then(*(
            layer(params[i]) for i in range(depth)))
    return circuit


def Sim14ansatz(n_qubits, params) -> Circuit:
    """
    Builds a modified version of circuit 14 from arXiv:1905.10876

    Replaces circuit-block construction with two rings of CRx gates, in
    opposite orientation.

    >>> pprint = lambda c: print(str(c.foliation()).replace(' >>', '\\n  >>'))
    >>> pprint(Sim14ansatz(3, [[i/10 for i in range(12)]]))
    Ry(0) @ Ry(0.1) @ Ry(0.2)
      >> Controlled(Rx(0.3), distance=2)
      >> Controlled(Rx(0.4), distance=-1) @ qubit
      >> qubit @ Controlled(Rx(0.5), distance=-1)
      >> Ry(0.6) @ Ry(0.7) @ Ry(0.8)
      >> CRx(0.9) @ qubit
      >> Controlled(Rx(1), distance=-2)
      >> qubit @ CRx(1.1)
    >>> print(Sim14ansatz(1, [0.1, 0.2, 0.3]))
    Rx(0.1) >> Rz(0.2) >> Rx(0.3)
    """
    from discopy.quantum.gates import Rx, Ry, Rz

    np = get_backend()

    def layer(thetas):
        sublayer1 = Id().tensor(
            *([Ry(theta) for theta in thetas[:n_qubits]]))

        for i in range(n_qubits):
            src = i
            tgt = (i - 1) % n_qubits
            sublayer1 = sublayer1.CRx(thetas[n_qubits + i], src, tgt)

        sublayer2 = Id().tensor(
            *([Ry(theta) for theta in thetas[2 * n_qubits: 3 * n_qubits]]))

        for i in range(n_qubits, 0, -1):
            src = i % n_qubits
            tgt = (i + 1) % n_qubits
            sublayer2 = sublayer2.CRx(thetas[-i], src, tgt)

        return sublayer1 >> sublayer2

    params_shape = np.shape(params)

    if n_qubits == 1:
        circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
    elif (len(params_shape) != 2) or (params_shape[1] != 4 * n_qubits):
        raise ValueError(
            f"Expected params of shape (depth, {4 * n_qubits})")
    else:
        depth = params_shape[0]
        circuit = Id(qubit ** n_qubits).then(*(
            layer(params[i]) for i in range(depth)))

    return circuit


def Rydberg(positions, duration, omega, delta, phase=0, steps=1,
            coupling=5420158.53) -> Circuit:
    """
    Trotterized evolution under the Rydberg Hamiltonian of `Pasqal's QPU
    <https://docs.pasqal.com/qpu-emulators/emumps/advanced/hamiltonian/>`_.

    An atom at position :math:`\\vec{r}_i` is a qubit with the ground state
    :math:`\\ket{g} = \\ket{0}` and the Rydberg state
    :math:`\\ket{r} = \\ket{1}`, evolving with :math:`\\hbar = 1` under

    .. math::
        H(t) = \\sum_i \\left( \\frac{\\Omega(t)}{2} \\left(
            \\cos\\varphi(t) \\, \\sigma_x^i
            - \\sin\\varphi(t) \\, \\sigma_y^i \\right)
            - \\delta(t) \\, n_i \\right)
        + \\sum_{i < j} \\frac{C_6}{|\\vec{r}_i - \\vec{r}_j|^6} \\, n_i n_j

    where :math:`n = \\ket{r}\\bra{r}` counts the Rydberg state. The evolution
    is cut into :code:`steps` slices of length :code:`dt = duration / steps`
    on which the waveforms are constant, each approximated to first order by
    a layer of drives :math:`R_z(\\varphi) R_x(\\Omega dt) R_z(-\\varphi)`,
    a layer of detunings :math:`U_1(\\delta dt)` and one interaction
    :math:`CU_1(-C_6 dt / r_{ij}^6)` for each pair of atoms, all exact so
    that a single step is the exact evolution whenever the terms commute.

    Parameters
    ----------
    positions : list of tuples of floats
        The coordinates of each atom, in :math:`\\mu m`.
    duration : float
        The evolution time, in :math:`\\mu s`.
    omega, delta, phase : float or list of floats
        The Rabi frequency and detuning in :math:`rad / \\mu s` and the
        laser phase in :math:`rad`, either constant or one sample per step.
    steps : int
        The number of Trotter steps, default is :code:`1`.
    coupling : float
        The interaction coefficient :math:`C_6` in
        :math:`rad \\cdot \\mu m^6 / \\mu s`, default is its value
        :code:`5420158.53` for Pasqal's devices at Rydberg level 70.

    Example
    -------
    >>> from math import pi
    >>> pprint = lambda c: print(str(c.foliation()).replace(' >>', '\\n  >>'))
    >>> pprint(Rydberg([(0, 0), (0, 1), (0, 2)], duration=2 * pi,
    ...                omega=1, delta=1, coupling=64))
    Rx(1) @ Rx(1) @ Rx(1)
      >> U1(1) @ U1(1) @ U1(1)
      >> CU1(-64) @ qubit
      >> Controlled(U1(-1), distance=2)
      >> qubit @ CU1(-64)
    >>> pprint(Rydberg([(0, 0)], duration=2 * pi, omega=1, delta=0,
    ...                phase=[0, pi], steps=2))
    Rx(0.5)
      >> U1(0)
      >> Rz(0.5)
      >> Rx(0.5)
      >> Rz(-0.5)
      >> U1(0)
    """
    from discopy.quantum.gates import Rx, Rz, U1, CU1

    n_atoms, dt = len(positions), duration / steps

    def samples(waveform):
        result = list(waveform)\
            if hasattr(waveform, "__len__") else steps * [waveform]
        if len(result) != steps:
            raise ValueError(f"Expected a number or {steps} samples, "
                             f"got {len(result)}")
        return result

    def strength(source, target):
        squared_distance = sum((x - y) ** 2 for x, y in zip(source, target))
        if not squared_distance:
            raise ValueError(f"Atoms at {source} and {target} coincide")
        return coupling / squared_distance ** 3

    interactions = [(i, j, strength(positions[i], positions[j]))
                    for i, j in combinations(range(n_atoms), 2)]

    def drive(omega, phase):
        pulse = Rx(omega * dt / (2 * pi))
        return pulse if phase == 0\
            else Rz(phase / (2 * pi)) >> pulse >> Rz(-phase / (2 * pi))

    def step(omega, delta, phase):
        result = Id().tensor(*n_atoms * [drive(omega, phase)])
        result >>= Id().tensor(*n_atoms * [U1(delta * dt / (2 * pi))])
        return result.then(*(
            qubit ** i @ CU1(-u * dt / (2 * pi), distance=j - i)
            @ qubit ** (n_atoms - j - 1) for i, j, u in interactions))

    return Id(qubit ** n_atoms).then(*(
        step(*point) for point in zip(
            samples(omega), samples(delta), samples(phase))))


def Sim15ansatz(n_qubits, params) -> Circuit:
    """
    Builds a modified version of circuit 15 from arXiv:1905.10876

    Replaces circuit-block construction with two rings of CNOT gates, in
    opposite orientation.

    >>> pprint = lambda c: print(str(c.foliation()).replace(' >>', '\\n  >>'))
    >>> pprint(Sim15ansatz(3, [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]))
    Ry(0.1) @ Ry(0.2) @ Ry(0.3)
      >> Controlled(X, distance=2)
      >> Controlled(X, distance=-1) @ qubit
      >> qubit @ Controlled(X, distance=-1)
      >> Ry(0.4) @ Ry(0.5) @ Ry(0.6)
      >> CX @ qubit
      >> Controlled(X, distance=-2)
      >> qubit @ CX
    >>> print(Sim15ansatz(1, [0.1, 0.2, 0.3]))
    Rx(0.1) >> Rz(0.2) >> Rx(0.3)
    """
    from discopy.quantum.gates import Rx, Ry, Rz

    np = get_backend()

    def layer(thetas):
        sublayer1 = Id().tensor(
            *([Ry(theta) for theta in thetas[:n_qubits]]))

        for i in range(n_qubits):
            src = i
            tgt = (i - 1) % n_qubits
            sublayer1 = sublayer1.CX(src, tgt)

        sublayer2 = Id().tensor(
            *([Ry(theta) for theta in thetas[n_qubits:]]))

        for i in range(n_qubits, 0, -1):
            src = i % n_qubits
            tgt = (i + 1) % n_qubits
            sublayer2 = sublayer2.CX(src, tgt)

        return sublayer1 >> sublayer2

    params_shape = np.shape(params)

    if n_qubits == 1:
        circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
    elif (len(params_shape) != 2) or (params_shape[1] != 2 * n_qubits):
        raise ValueError(
            f"Expected params of shape (depth, {2 * n_qubits})")
    else:
        depth = params_shape[0]
        circuit = Id(qubit ** n_qubits).then(*(
            layer(params[i]) for i in range(depth)))

    return circuit
