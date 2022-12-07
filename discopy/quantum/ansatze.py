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

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        real_amp_ansatz
        random_tiling
"""

import random
from functools import reduce, partial
from itertools import takewhile, chain

from discopy.matrix import get_backend
from discopy.tensor import Tensor
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
            "Expected params of shape (depth, {})".format(n_qubits - 1))
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
            "Expected params of shape (depth, {})".format(4 * n_qubits))
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
            "Expected params of shape (depth, {})".format(2 * n_qubits))
    else:
        depth = params_shape[0]
        circuit = Id(qubit ** n_qubits).then(*(
            layer(params[i]) for i in range(depth)))

    return circuit


def real_amp_ansatz(params: "array", *, entanglement='full'):
    """
    The real-amplitudes 2-local circuit. The shape of the params determines
    the number of layers and the number of qubits respectively (layers, qubit).
    This heuristic generates orthogonal operators so the imaginary part of the
    correponding matrix is always the zero matrix.
    :param params: A 2D numpy array of parameters.
    :param entanglement: Configuration for the entaglement, currently either
    'full' (default), 'linear' or 'circular'.
    """
    from discopy.quantum.gates import CX, Ry, rewire
    np = get_backend()
    ext_cx = partial(rewire, CX)
    assert entanglement in ('linear', 'circular', 'full')
    params = np.asarray(params)
    assert params.ndim == 2
    dom = qubit**params.shape[1]

    def layer(v, is_last=False):
        n = len(dom)
        rys = Id().tensor(*(Ry(v[k]) for k in range(n)))
        if is_last:
            return rys
        if entanglement == 'full':
            cxs = [[ext_cx(k1, k2, dom=dom) for k2 in range(k1 + 1, n)] for
                   k1 in range(n - 1)]
            cxs = reduce(lambda a, b: a >> b, chain(*cxs))
        else:
            cxs = [ext_cx(k, k + 1, dom=dom) for k in range(n - 1)]
            cxs = reduce(lambda a, b: a >> b, cxs)
            if entanglement == 'circular':
                cxs = ext_cx(n - 1, 0, dom=dom) >> cxs
        return rys >> cxs

    circuit = [layer(v, is_last=idx == (len(params) - 1)) for
               idx, v in enumerate(params)]
    circuit = reduce(lambda a, b: a >> b, circuit)
    return circuit


def random_tiling(n_qubits, depth=3, gateset=None, seed=None):
    """ Returns a random Euler decomposition if n_qubits == 1,
    otherwise returns a random tiling with the given depth and gateset.

    >>> from discopy.quantum.gates import CX, H, T, Rx, Rz
    >>> c = random_tiling(1, seed=420)
    >>> print(c)
    Rx(0.0263) >> Rz(0.781) >> Rx(0.273)
    >>> print(random_tiling(2, 2, gateset=[CX, H, T], seed=420).foliation())
    CX >> T @ T
    >>> print(random_tiling(3, 2, gateset=[CX, H, T], seed=420).foliation())
    CX @ T >> H @ H @ H
    >>> print(random_tiling(2, 1, gateset=[Rz, Rx], seed=420).foliation())
    Rz(0.673) @ Rx(0.273)
    """
    from discopy.quantum.gates import H, CX, Rx, Rz, Parametrized
    gateset = gateset or [H, Rx, CX]
    if seed is not None:
        random.seed(seed)
    if n_qubits == 1:
        phases = [random.random() for _ in range(3)]
        return Rx(phases[0]) >> Rz(phases[1]) >> Rx(phases[2])
    result = Id(qubit ** n_qubits)
    for _ in range(depth):
        line, n_affected = Id(), 0
        while n_affected < n_qubits:
            gate = random.choice(
                gateset if n_qubits - n_affected > 1 else [
                    g for g in gateset
                    if g is Rx or g is Rz or len(g.dom) == 1])
            if isinstance(gate, type) and issubclass(gate, Parametrized):
                gate = gate(random.random())
            line = line @ gate
            n_affected += len(gate.dom)
        result = result >> line
    return result
