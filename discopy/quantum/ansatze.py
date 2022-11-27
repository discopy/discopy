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

class IQPansatz(Circuit):
    """
    Build an IQP ansatz on n qubits, if n = 1 returns an Euler decomposition

    >>> pprint = lambda c: print(str(c).replace(' >>', '\\n  >>'))
    >>> pprint(IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]))
    H @ Id(2)
      >> Id(1) @ H @ Id(1)
      >> Id(2) @ H
      >> CRz(0.1) @ Id(1)
      >> Id(1) @ CRz(0.2)
      >> H @ Id(2)
      >> Id(1) @ H @ Id(1)
      >> Id(2) @ H
      >> CRz(0.3) @ Id(1)
      >> Id(1) @ CRz(0.4)
    >>> print(IQPansatz(1, [0.3, 0.8, 0.4]))
    Rx(0.3) >> Rz(0.8) >> Rx(0.4)
    """
    def __init__(self, n_qubits, params):
        from discopy.quantum.gates import H, Rx, Rz, CRz

        def layer(thetas):
            hadamards = Id(0).tensor(*(n_qubits * [H]))
            rotations = Id(n_qubits).then(*(
                Id(i) @ CRz(thetas[i]) @ Id(n_qubits - 2 - i)
                for i in range(n_qubits - 1)))
            return hadamards >> rotations
        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        elif len(Tensor.np.shape(params)) != 2\
                or Tensor.np.shape(params)[1] != n_qubits - 1:
            raise ValueError(
                "Expected params of shape (depth, {})".format(n_qubits - 1))
        else:
            depth = Tensor.np.shape(params)[0]
            circuit = Id(n_qubits).then(*(
                layer(params[i]) for i in range(depth)))
        super().__init__(
            circuit.dom, circuit.cod, circuit.boxes, circuit.offsets)


class Sim14ansatz(Circuit):
    """
    Builds a modified version of circuit 14 from arXiv:1905.10876

    Replaces circuit-block construction with two rings of CRx gates, in
    opposite orientation.

    >>> pprint = lambda c: print(str(c).replace(' >>', '\\n  >>'))
    >>> pprint(Sim14ansatz(3, [[i/10 for i in range(12)]]))
    Ry(0) @ Id(2)
      >> Id(1) @ Ry(0.1) @ Id(1)
      >> Id(2) @ Ry(0.2)
      >> CRx(0.3)
      >> CRx(0.4) @ Id(1)
      >> Id(1) @ CRx(0.5)
      >> Ry(0.6) @ Id(2)
      >> Id(1) @ Ry(0.7) @ Id(1)
      >> Id(2) @ Ry(0.8)
      >> CRx(0.9) @ Id(1)
      >> CRx(1)
      >> Id(1) @ CRx(1.1)
    >>> pprint(Sim14ansatz(1, [0.1, 0.2, 0.3]))
    Rx(0.1)
      >> Rz(0.2)
      >> Rx(0.3)
    """

    def __init__(self, n_qubits, params):
        from discopy.quantum.gates import Rx, Ry, Rz

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

        params_shape = Tensor.np.shape(params)

        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        elif (len(params_shape) != 2) or (params_shape[1] != 4 * n_qubits):
            raise ValueError(
                "Expected params of shape (depth, {})".format(4 * n_qubits))
        else:
            depth = params_shape[0]
            circuit = Id(n_qubits).then(*(
                layer(params[i]) for i in range(depth)))

        super().__init__(
            circuit.dom, circuit.cod, circuit.boxes, circuit.offsets)


class Sim15ansatz(Circuit):
    """
    Builds a modified version of circuit 15 from arXiv:1905.10876

    Replaces circuit-block construction with two rings of CNOT gates, in
    opposite orientation.

    >>> pprint = lambda c: print(str(c).replace(' >>', '\\n  >>'))
    >>> pprint(Sim15ansatz(3, [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]))
    Ry(0.1) @ Id(2)
      >> Id(1) @ Ry(0.2) @ Id(1)
      >> Id(2) @ Ry(0.3)
      >> CX
      >> CX @ Id(1)
      >> Id(1) @ CX
      >> Ry(0.4) @ Id(2)
      >> Id(1) @ Ry(0.5) @ Id(1)
      >> Id(2) @ Ry(0.6)
      >> CX @ Id(1)
      >> CX
      >> Id(1) @ CX
    >>> pprint(Sim15ansatz(1, [0.1, 0.2, 0.3]))
    Rx(0.1)
      >> Rz(0.2)
      >> Rx(0.3)
    """

    def __init__(self, n_qubits, params):
        from discopy.quantum.gates import Rx, Ry, Rz

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

        params_shape = Tensor.np.shape(params)

        if n_qubits == 1:
            circuit = Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
        elif (len(params_shape) != 2) or (params_shape[1] != 2 * n_qubits):
            raise ValueError(
                "Expected params of shape (depth, {})".format(2 * n_qubits))
        else:
            depth = params_shape[0]
            circuit = Id(n_qubits).then(*(
                layer(params[i]) for i in range(depth)))

        super().__init__(
            circuit.dom, circuit.cod, circuit.boxes, circuit.offsets)


def real_amp_ansatz(params: Tensor.np.ndarray, *, entanglement='full'):
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
    ext_cx = partial(rewire, CX)
    assert entanglement in ('linear', 'circular', 'full')
    params = Tensor.np.asarray(params)
    assert params.ndim == 2
    dom = qubit**params.shape[1]

    def layer(v, is_last=False):
        n = len(dom)
        rys = Id(0).tensor(*(Ry(v[k]) for k in range(n)))
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
    >>> print(random_tiling(2, 2, gateset=[CX, H, T], seed=420))
    CX >> T @ Id(1) >> Id(1) @ T
    >>> print(random_tiling(3, 2, gateset=[CX, H, T], seed=420))
    CX @ Id(1) >> Id(2) @ T >> H @ Id(2) >> Id(1) @ H @ Id(1) >> Id(2) @ H
    >>> print(random_tiling(2, 1, gateset=[Rz, Rx], seed=420))
    Rz(0.673) @ Id(1) >> Id(1) @ Rx(0.273)
    """
    from discopy.quantum.gates import H, CX, Rx, Rz, Parametrized
    gateset = gateset or [H, Rx, CX]
    if seed is not None:
        random.seed(seed)
    if n_qubits == 1:
        phases = [random.random() for _ in range(3)]
        return Rx(phases[0]) >> Rz(phases[1]) >> Rx(phases[2])
    result = Id(n_qubits)
    for _ in range(depth):
        line, n_affected = Id(0), 0
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
