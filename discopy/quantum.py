import random as random
from itertools import takewhile

from discopy import monoidal, rigid, Quiver
from discopy.rigid import Ob, Ty, Diagram
from discopy.tensor import np, Dim, Tensor, TensorFunctor


class CQ(Ty):
    def __init__(self, classical=Dim(1), quantum=Dim(1)):
        self.classical, self.quantum = classical, quantum
        types = [Ob("C({})".format(dim)) for dim in classical]\
            + [Ob("Q({})".format(dim)) for dim in quantum]
        super().__init__(*types)

    def __repr__(self):
        if not self:
            return "CQ()"
        if not self.classical:
            return "Q({})".format(repr(self.quantum))
        if not self.quantum:
            return "C({})".format(repr(self.classical))
        return "C({}) @ Q({})".format(repr(self.classical), repr(self.quantum))

    def __str__(self):
        return repr(self)

    def tensor(self, other):
        return CQ(
            self.classical @ other.classical, self.quantum @ other.quantum)


class C(CQ):
    def __init__(self, dim):
        super().__init__(dim, Dim(1))


class Q(CQ):
    def __init__(self, dim):
        super().__init__(Dim(1), dim)


class CQMap(rigid.Box):
    def __init__(self, dom, cod, array):
        data = Tensor(dom.classical @ dom.quantum @ dom.quantum,
                      cod.classical @ cod.quantum @ cod.quantum, array)
        self.array = data.array
        super().__init__("CQMap", dom, cod, data=data)

    def __eq__(self, other):
        return isinstance(other, CQMap)\
            and (self.dom, self.cod) == (other.dom, other.cod)\
            and self.data == other.data

    def __repr__(self):
        return "CQMap(dom={}, cod={}, array={})".format(
            self.dom, self.cod, np.array2string(self.array.flatten()))

    def __str__(self):
        return repr(self)

    @staticmethod
    def id(dom):
        data = Tensor.id(dom.classical @ dom.quantum @ dom.quantum)
        return CQMap(dom, dom, data.array)

    def then(self, other):
        data = self.data >> other.data
        return CQMap(self.dom, other.cod, data.array)

    def dagger(self):
        return CQMap(self.cod, self.dom, self.data.dagger().array)

    def tensor(self, other):
        f = rigid.Box('f', Ty('c00', 'q00', 'q00'), Ty('c10', 'q10', 'q10'))
        g = rigid.Box('g', Ty('c01', 'q01', 'q01'), Ty('c11', 'q11', 'q11'))
        ob = {Ty("{}{}{}".format(a, b, c)):
              z.__getattribute__(y).__getattribute__(x)
              for a, x in zip(['c', 'q'], ['classical', 'quantum'])
              for b, y in zip([0, 1], ['dom', 'cod'])
              for c, z in zip([0, 1], [self, other])}
        ar = {f: self.array, g: other.array}
        permute_above = Diagram.id(f.dom[:1] @ g.dom[:1] @ f.dom[1:2])\
            @ Diagram.swap(g.dom[1:2], f.dom[2:]) @ Diagram.id(g.dom[2:])\
            >> Diagram.id(f.dom[:1]) @ Diagram.swap(g.dom[:1], f.dom[1:])\
            @ Diagram.id(g.dom[1:])
        permute_below =\
            Diagram.id(f.cod[:1]) @ Diagram.swap(f.cod[1:], g.cod[:1])\
            @ Diagram.id(g.cod[1:])\
            >> Diagram.id(f.cod[:1] @ g.cod[:1] @ f.cod[1:2])\
            @ Diagram.swap(f.cod[2:], g.cod[1:2]) @ Diagram.id(g.cod[2:])
        F = TensorFunctor(ob, ar)
        array = F(permute_above >> f @ g >> permute_below).array
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        return CQMap(dom, cod, array)

    @staticmethod
    def swap(left, right):
        data = Tensor.swap(left.classical, right.classical)\
            @ Tensor.swap(left.quantum ** 2, right.quantum ** 2)
        return CQMap(left @ right, right @ left, data.array)

    @staticmethod
    def measure(dim):
        if not dim:
            return CQMap(CQ(), CQ(), np.array(1))
        if len(dim) == 1:
            array = np.array([
                i == j == k
                for i in range(dim[0])
                for j in range(dim[0])
                for k in range(dim[0])])
            return CQMap(Q(dim), C(dim), array)
        return CQMap.measure(dim[:1]) @ CQMap.measure(dim[1:])

    @staticmethod
    def encode(dim):
        return CQMap(C(dim), Q(dim), CQMap.measure(dim).array)

    @staticmethod
    def pure(tensor):
        return CQMap(Q(tensor.dom), Q(tensor.cod),
                     (tensor.conjugate() @ tensor).array)

    def classical(tensor):
        return CQMap(C(tensor.dom), C(tensor.cod), tensor.array)

    @staticmethod
    def discard(dom):
        array = np.tensordot(
            np.ones(dom.classical), Tensor.id(dom.quantum).array, 0)
        return CQMap(dom, CQ(), array)

    def is_close(self, other):
        return isinstance(other, CQMap)\
            and (self.dom, self.cod) == (other.dom, other.cod)\
            and np.allclose(self.array, other.array)

    @property
    def is_causal(self):
        return self.discard(self.dom).is_close(self >> self.discard(self.cod))

    @staticmethod
    def cups(left, right):
        return CQMap.classical(Tensor.cups(left.classical, right.classical))\
            @ CQMap.pure(Tensor.cups(left.quantum, right.quantum))

    @staticmethod
    def caps(left, right):
        return CQMap.cups(left, right).dagger()


class BitsAndQubits(Ty):
    @staticmethod
    def _upgrade(ty):
        return BitsAndQubits(*ty.objects)

    def __repr__(self):
        if not self:
            return "Ty()"
        n_bits = len(list(takewhile(lambda x: x.name == "bit", self.objects)))
        n_qubits = len(list(takewhile(
            lambda x: x.name == "qubit", self.objects[n_bits:])))
        remainder = self.objects[n_bits + n_qubits:]
        left = "" if not n_bits else "bit{}".format(
            " ** {}".format(n_bits) if n_bits > 1 else "")
        middle = "" if not n_qubits else "qubit{}".format(
            " ** {}".format(n_qubits) if n_qubits > 1 else "")
        right = "" if not remainder else repr(BitsAndQubits(*remainder))
        return " @ ".join(s for s in [left, middle, right] if s)

    def __str__(self):
        return repr(self)


class Circuit(Diagram):
    @staticmethod
    def _upgrade(diagram):
        dom, cod = BitsAndQubits(*diagram.dom), BitsAndQubits(*diagram.cod)
        return Circuit(
            dom, cod, diagram.boxes, diagram.offsets, diagram.layers)

    def __repr__(self):
        return super().__repr__().replace('Diagram', 'Circuit')

    @staticmethod
    def id(dom):
        return Id(dom)

    @staticmethod
    def swap(left, right):
        return monoidal.swap(left, right,
                             ar_factory=Circuit, swap_factory=Swap)

    @staticmethod
    def permutation(perm, dom=None):
        return permutation(perm, dom)

    def eval(self, mixed=False):
        mixed = mixed or any(
            isinstance(box, (Discard, MixedState, Measure, Encode))
            for box in self.boxes)
        if mixed:
            ob, ar = {Ty('bit'): C(Dim(2)), Ty('qubit'): Q(Dim(2))}, {}
            return CQMapFunctor(ob, ar)(self)
        ob, ar = {Ty('bit'): 2, Ty('qubit'): 2}, Quiver(lambda g: g.array)
        return TensorFunctor(ob, ar)(self)

    def to_tk(self):
        """
        Returns
        -------
        tk_circuit : pytket.Circuit
            A :class:`pytket.Circuit`.

        Note
        ----
        * No measurements are performed.
        * SWAP gates are treated as logical swaps.
        * If the circuit contains scalars or a :class:`Bra`,
          then :code:`tk_circuit` will hold attributes
          :code:`post_selection` and :code:`scalar`.

        Examples
        --------
        >>> circuit0 = H @ Rx(0.5) >> CX
        >>> print(list(circuit0.to_tk()))
        [H q[0];, Rx(1*PI) q[1];, CX q[0], q[1];]

        >>> circuit1 = Ket(1, 0) >> CX >> Id(1) @ Ket(0) @ Id(1)
        >>> print(list(circuit1.to_tk()))
        [X q[0];, CX q[0], q[2];]
        >>> circuit2 = Circuit.from_tk(circuit1.to_tk())
        >>> print(circuit2)
        X @ Id(2) >> Id(1) @ SWAP >> CX @ Id(1) >> Id(1) @ SWAP
        >>> print(list(circuit2.to_tk()))
        [X q[0];, CX q[0], q[2];]
        >>> assert CRz(0.5) == Circuit.from_tk(CRz(0.5).to_tk())

        >>> circuit = Ket(0, 0)\\
        ...     >> sqrt(2) @ Id(2)\\
        ...     >> H @ Id(1)\\
        ...     >> Id(1) @ X\\
        ...     >> CX\\
        ...     >> Id(1) @ Bra(0)
        >>> tk_circ = circuit.to_tk()
        >>> print(list(tk_circ))
        [H q[0];, X q[1];, CX q[0], q[1];]
        >>> print(tk_circ.post_selection)
        {1: 0}
        >>> print(np.round(abs(tk_circ.scalar) ** 2))
        2.0
        """
        from discopy.tk_interface import to_tk
        return to_tk(self)

    @staticmethod
    def from_tk(tk_circuit):
        """
        Parameters
        ----------
        tk_circuit : pytket.Circuit
            A pytket.Circuit, potentially with :code:`scalar` and
            :code:`post_selection` attributes.

        Returns
        -------
        circuit : :class:`Circuit`
            Such that :code:`Circuit.from_tk(circuit.to_tk()) == circuit`.

        Note
        ----
        * SWAP gates are introduced when applying gates to non-adjacent qubits.

        Examples
        --------
        >>> c1 = Rz(0.5) @ Id(1) >> Id(1) @ Rx(0.25) >> CX
        >>> c2 = Circuit.from_tk(c1.to_tk())
        >>> assert c1.normal_form() == c2.normal_form()

        >>> import pytket as tk
        >>> tk_GHZ = tk.Circuit(3).H(1).CX(1, 2).CX(1, 0)
        >>> print(Circuit.from_tk(tk_GHZ))
        Id(1) @ H @ Id(1)\\
          >> Id(1) @ CX\\
          >> SWAP @ Id(1)\\
          >> CX @ Id(1)\\
          >> SWAP @ Id(1)
        >>> circuit = Ket(1, 0) >> CX >> Id(1) @ Ket(0) @ Id(1)
        >>> print(Circuit.from_tk(circuit.to_tk()))
        X @ Id(2) >> Id(1) @ SWAP >> CX @ Id(1) >> Id(1) @ SWAP

        >>> bell_state = Circuit.caps(PRO(1), PRO(1))
        >>> bell_effect = bell_state[::-1]
        >>> circuit = bell_state @ Id(1) >> Id(1) @ bell_effect >> Bra(0)
        >>> print(Circuit.from_tk(circuit.to_tk()))
        H @ Id(2)\\
          >> CX @ Id(1)\\
          >> Id(1) @ CX\\
          >> Id(1) @ H @ Id(1)\\
          >> Id(2) @ Bra(0)\\
          >> Id(1) @ Bra(0)\\
          >> Bra(0)\\
          >> scalar(2.000)
        """
        from discopy.tk_interface import from_tk
        return from_tk(tk_circuit)

    def get_counts(self, backend, **params):
        """
        Parameters
        ----------
        backend : pytket.Backend
         Backend on which to run the circuit.
        n_shots : int, optional
         Number of shots, default is :code:`2**10`.
        measure_all : bool, optional
         Whether to measure all qubits, default is :code:`True`.
        normalize : bool, optional
         Whether to normalize the counts, default is :code:`True`.
        post_select : bool, optional
         Whether to perform post-selection, default is :code:`True`.
        scale : bool, optional
         Whether to scale the output, default is :code:`True`.
        seed : int, optional
         Seed to feed the backend, default is :code:`None`.

        Returns
        -------
        tensor : :class:`discopy.tensor.Tensor`
         Of dimension :code:`n_qubits * (2, )` for :code:`n_qubits` the
         number of post-selected qubits.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> backend = Mock()
        >>> backend.get_counts.return_value = {(0, 0): 502, (1, 1): 522}
        >>> circuit = H @ Id(1) >> CX >> Id(1) @ Bra(0)
        >>> circuit.get_counts(backend, seed=42)  # doctest: +ELLIPSIS
        Tensor(dom=Dim(1), cod=Dim(2), array=[0.49..., 0...])
        """
        from discopy.tk_interface import get_counts
        return get_counts(self, backend, **params)


class Id(rigid.Id, Circuit):
    def __init__(self, dom):
        if isinstance(dom, int):
            dom = qubit ** dom
        rigid.Id.__init__(self, dom)
        Circuit.__init__(self, dom, dom, [], [])


class Box(rigid.Box, Circuit):
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        rigid.Box.__init__(self, name, dom, cod, data=data, _dagger=_dagger)
        Circuit.__init__(self, dom, cod, [self], [0])

    def __repr__(self):
        return self.name


class Swap(rigid.Swap, Box):
    pass


class Discard(Box):
    def __init__(self, dom=1):
        if isinstance(dom, int):
            dom = qubit ** dom
        super().__init__("Discard({})".format(dom), dom, qubit ** 0)

    def dagger(self):
        return MixedState(self.dom)


class MixedState(Box):
    def __init__(self, cod=1):
        if isinstance(cod, int):
            cod = qubit ** cod
        super().__init__("MixedState({})".format(cod), qubit ** 0, cod)

    def dagger(self):
        return Discard(self.cod)


class Measure(Box):
    def __init__(self, n_qubits=1):
        dom, cod = qubit ** n_qubits, bit ** n_qubits
        super().__init__("Measure({})".format(n_qubits), dom, cod)

    def dagger(self):
        return Encode(len(self.cod))


class Encode(Box):
    def __init__(self, n_bits=1):
        dom, cod = bit ** n_bits, qubit ** n_bits
        super().__init__("Encode({})".format(n_bits), dom, cod)

    def dagger(self):
        return Measure(len(self.dom))


class QGate(Box):
    def __init__(self, name, n_qubits, array=None, _dagger=False):
        dom = qubit ** n_qubits
        if array is not None:
            self._array = np.array(array).reshape(2 * n_qubits * (2, ) or 1)
        super().__init__(name, dom, dom, _dagger=_dagger)

    @property
    def array(self):
        return self._array

    def __repr__(self):
        return "QGate({}, {}, {})".format(
            repr(self.name), len(self.dom),
            np.array2string(self.array.flatten()))

    def dagger(self):
        return QGate(
            self.name, len(self.dom), self.array,
            _dagger=None if self._dagger is None else not self._dagger)


class CGate(Box):
    def __init__(self, name, n_bits_in, n_bits_out, array, _dagger=False):
        dom, cod = bit ** n_bits_in, bit ** n_bits_out
        if array is not None:
            self._array = np.array(array).reshape(
                (n_bits_in + n_bits_out) * (2, ) or 1)
        super().__init__(name, dom, cod, _dagger=_dagger)

    @property
    def array(self):
        return self._array

    def __repr__(self):
        return "CGate({}, {}, {}, {})".format(
            repr(self.name), len(self.dom), len(self.cod),
            np.array2string(self.array))

    def dagger(self):
        return CGate(
            self.name, len(self.dom), len(self.cod), self.array,
            _dagger=None if self._dagger is None else not self._dagger)

    @staticmethod
    def func(name, n_bits_in, n_bits_out, function):
        array = np.zeros((n_bits_in + n_bits_out) * [2])
        for i in range(2 ** n_bits_in):
            bitstring = tuple(map(int, format(i, '0{}b'.format(n_bits_in))))
            array[bitstring + tuple(function(*bitstring))] = 1
        return CGate(name, n_bits_in, n_bits_out, array)


class Ket(Box):
    def __init__(self, *bitstring):
        self.bitstring = bitstring
        name = 'Ket({})'.format(', '.join(map(str, bitstring)))
        dom, cod = qubit ** 0, qubit ** len(bitstring)
        super().__init__(name, dom, cod)

    @property
    def array(self):
        tensor = Tensor(Dim(1), Dim(1), [1])
        for bit in self.bitstring:
            tensor = tensor @ Tensor(Dim(2), Dim(1), [0, 1] if bit else [1, 0])
        return tensor.array

    def dagger(self):
        return Bra(*self.bitstring)


class Bra(Box):
    def __init__(self, *bitstring):
        self.bitstring = bitstring
        name = 'Bra({})'.format(', '.join(map(str, bitstring)))
        dom, cod = qubit ** len(bitstring), qubit ** 0
        super().__init__(name, dom, cod)

    @property
    def array(self):
        return Ket(*self.bitstring).array

    def dagger(self):
        return Ket(*self.bitstring)


class Rotation(QGate):
    def __init__(self, phase, n_qubits=1):
        self._phase = phase
        super().__init__('Rx', n_qubits, array=None)

    @property
    def phase(self):
        return self._phase

    @property
    def name(self):
        return '{}({})'.format(self._name, self.phase)

    def dagger(self):
        return type(self)(-self.phase)

    def __repr__(self):
        return self.name


class Rx(Rotation):
    """"""
    @property
    def array(self):
        half_theta = np.pi * self.phase
        global_phase = np.exp(1j * half_theta)
        sin, cos = np.sin(half_theta), np.cos(half_theta)
        return global_phase * np.array([[cos, -1j * sin], [-1j * sin, cos]])


class Rz(Rotation):
    @property
    def array(self):
        theta = 2 * np.pi * self.phase
        return np.array([[1, 0], [0, np.exp(1j * theta)]])


class CRz(Rotation):
    def __init__(self, phase):
        super().__init__(self, phase, n_qubits=2)

    @property
    def array(self):
        phase = np.exp(1j * 2 * np.pi * self.phase)
        return np.array([1, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, 1, 0,
                         0, 0, 0, phase])


class CQMapFunctor(rigid.Functor):
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=CQ, ar_factory=CQMap)

    def __repr__(self):
        return super().__repr__().replace("Functor", "CQMapFunctor")

    def __call__(self, diagram):
        if isinstance(diagram, (QGate, Bra, Ket)):
            dom, cod = self(diagram.dom).quantum, self(diagram.cod).quantum
            return CQMap.pure(Tensor(dom, cod, diagram.array))
        if isinstance(diagram, CGate):
            dom, cod = self(diagram.dom), self(diagram.cod)
            return CQMap(dom, cod, diagram.array)
        if isinstance(diagram, Swap):
            return CQMap.swap(self(diagram.dom[:1]), self(diagram.dom[1:]))
        if isinstance(diagram, Discard):
            return CQMap.discard(self(diagram.dom))
        if isinstance(diagram, Measure):
            return CQMap.measure(self(diagram.dom).quantum)
        if isinstance(diagram, Encode):
            return CQMap.encode(self(diagram.dom).classical)
        return super().__call__(diagram)


class CircuitFunctor(rigid.Functor):
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=BitsAndQubits, ar_factory=Circuit)

    def __repr__(self):
        return super().__repr__().replace("Functor", "CircuitFunctor")


bit, qubit = BitsAndQubits("bit"), BitsAndQubits("qubit")

SWAP = Swap(qubit, qubit)
CX = QGate('CX', 2, [1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1,
                     0, 0, 1, 0], _dagger=None)
CZ = QGate('CZ', 2, [1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, -1], _dagger=None)
H = QGate('H', 1, 1 / np.sqrt(2) * np.array([1, 1, 1, -1]), _dagger=None)
S = QGate('S', 1, [1, 0, 0, 1j])
T = QGate('T', 1, [1, 0, 0, np.exp(1j * np.pi / 4)])
X = QGate('X', 1, [0, 1, 1, 0], _dagger=None)
Y = QGate('Y', 1, [0, -1j, 1j, 0])
Z = QGate('Z', 1, [1, 0, 0, -1], _dagger=None)


def permutation(perm, dom=None):
    if dom is None:
        dom = qubit ** len(perm)
    return monoidal.permutation(perm, dom, ar_factory=Circuit)


def sqrt(real):
    return QGate('sqrt({})'.format(real), 0, np.sqrt(real), _dagger=None)


def scalar(complex):
    return Gate('scalar({:.3f})'.format(complex), 0, complex,
                _dagger=None if np.conjugate(complex) == complex else False)


def random_tiling(n_qubits, depth=3, gateset=[H, Rx, CX], seed=None):
    """ Returns a random Euler decomposition if n_qubits == 1,
    otherwise returns a random tiling with the given depth and gateset.

    >>> c = Circuit.random(1, seed=420)
    >>> print(c)  # doctest: +ELLIPSIS
    Rx(0.026... >> Rz(0.781... >> Rx(0.272...
    >>> print(Circuit.random(2, 2, gateset=[CX, H, T], seed=420))
    CX >> T @ Id(1) >> Id(1) @ T
    >>> print(Circuit.random(3, 2, gateset=[CX, H, T], seed=420))
    CX @ Id(1) >> Id(2) @ T >> H @ Id(2) >> Id(1) @ H @ Id(1) >> Id(2) @ H
    >>> print(Circuit.random(2, 1, gateset=[Rz, Rx], seed=420))
    Rz(0.6731171219152886) @ Id(1) >> Id(1) @ Rx(0.2726063832840899)
    """
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
            if gate is Rx or gate is Rz:
                gate = gate(random.random())
            line = line @ gate
            n_affected += len(gate.dom)
        result = result >> line
    return result


def IQPansatz(n, params):
    """
    Builds an IQP ansatz on n qubits, if n = 1 returns an Euler decomposition

    >>> print(IQPansatz(3, [[0.1, 0.2], [0.3, 0.4]]))
    H @ Id(2)\\
    >> Id(1) @ H @ Id(1)\\
    >> Id(2) @ H\\
    >> CRz(0.1) @ Id(1)\\
    >> Id(1) @ CRz(0.2)\\
    >> H @ Id(2)\\
    >> Id(1) @ H @ Id(1)\\
    >> Id(2) @ H\\
    >> CRz(0.3) @ Id(1)\\
    >> Id(1) @ CRz(0.4)
    >>> print(IQPansatz(1, [0.3, 0.8, 0.4]))
    Rx(0.3) >> Rz(0.8) >> Rx(0.4)
    """
    def Hlayer(n):
        layer = H
        for nn in range(1, n):
            layer = layer @ H
        return layer

    def CRzlayer(thetas):
        n = len(thetas) + 1
        layer = CRz(thetas[0]) @ Id(n - 2)
        for tt in range(1, len(thetas)):
            layer = layer >> Id(tt) @ CRz(thetas[tt]) @ Id(n - 2 - tt)
        return layer

    def IQPlayer(thetas):
        n = len(thetas) + 1
        return Hlayer(n) >> CRzlayer(thetas)
    if n == 1:
        assert np.shape(params) == (3,)
        return Rx(params[0]) >> Rz(params[1]) >> Rx(params[2])
    depth = np.shape(params)[0]
    assert n == np.shape(params)[1] + 1
    ansatz = IQPlayer(params[0])
    for i in range(1, depth):
        ansatz = ansatz >> IQPlayer(params[i])
    return ansatz
