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

"""

from discopy.matrix import get_backend
from discopy.quantum.circuit import qubit, Circuit, Id


def IQPansatz(n_qubits, params) -> Circuit:
    """
    Build an IQP ansatz on n qubits, if n = 1 returns an Euler decomposition.

    >>> pprint = lambda c: print(str(c.foliation()).replace(' >>', '\\n  >>'))
    >>> pprint(IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]))
    H @ H @ H
      >> CRz(0.1) @ qubit
      >> H @ CRz(0.2)
      >> qubit @ H @ H
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
      >> Ry(0.6) @ Controlled(Rx(0.5), distance=-1)
      >> qubit @ Ry(0.7) @ Ry(0.8)
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
      >> Ry(0.4) @ Controlled(X, distance=-1)
      >> qubit @ Ry(0.5) @ Ry(0.6)
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
