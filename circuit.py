# -*- coding: utf-8 -*-

"""
Implements quantum circuits as diagrams and circuit-valued monoidal functors.

>>> n = Ty('n')
>>> Alice = Box('Alice', Ty(), n)
>>> loves = Box('loves', n, n)
>>> Bob = Box('Bob', n, Ty())
>>> ob, ar = {n: 1}, {Alice: Ket(0), loves: X, Bob: Bra(1)}
>>> F = CircuitFunctor(ob, ar)
>>> F(Alice >> loves >> Bob)
Circuit(0, 0, [Ket(0), Gate('X', 1, [0, 1, 1, 0]), Bra(1)], [0, 0, 0])
>>> assert F(Alice >> loves >> Bob).eval()
"""

from random import choice, random as _random, seed as _seed
import pytket as tk
from discopy.cat import Quiver
from discopy.rigidcat import Ty, Box, Diagram, RigidFunctor
from discopy.matrix import np, Dim, Matrix, MatrixFunctor


class PRO(Ty):
    """ Implements the objects of a PRO, i.e. a non-symmetric PROP.
    Wraps a natural number n into a unary type Ty(1, ..., 1) of length n.

    >>> PRO(1) @ PRO(1)
    PRO(2)
    >>> assert PRO(3) == Ty(1, 1, 1)
    """
    @property
    def l(self):
        """
        >>> assert PRO(2).l == PRO(2)
        """
        return self

    @property
    def r(self):
        return self

    def tensor(self, other):
        return PRO(len(self) + len(other))

    def __init__(self, n=0):
        n = n if isinstance(n, int) else len(n)
        super().__init__(*(n * [1]))

    def __repr__(self):
        return "PRO({})".format(len(self))

    def __str__(self):
        return repr(self)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return PRO(len(super().__getitem__(key)))
        return super().__getitem__(key)


class Circuit(Diagram):
    """
    Implements quantum circuits as diagrams.
    """
    def __init__(self, dom, cod, gates, offsets, _fast=False):
        """
        >>> c = Circuit(2, 2, [CX, CX], [0, 0])
        """
        self._gates = gates
        super().__init__(PRO(dom), PRO(cod), gates, offsets, _fast=_fast)

    @property
    def gates(self):
        """
        >>> Circuit(1, 1, [X, X], [0, 0]).gates
        [Gate('X', 1, [0, 1, 1, 0]), Gate('X', 1, [0, 1, 1, 0])]
        """
        return self._gates

    def __repr__(self):
        """
        >>> Circuit(2, 2, [CX, CX], [0, 0])  # doctest: +ELLIPSIS
        Circuit(2, 2, [Gate('CX', 2, [...]), Gate('CX', 2, [...])], [0, 0])
        """
        return "Circuit({}, {}, {}, {})".format(
            len(self.dom), len(self.cod), self.gates, self.offsets)

    def then(self, other):
        """
        >>> print(SWAP >> CX)
        SWAP >> CX
        """
        result = super().then(other)
        return Circuit(len(result.dom), len(result.cod),
                       result.boxes, result.offsets, _fast=True)

    def tensor(self, other):
        """
        >>> print(CX @ H)
        CX @ Id(1) >> Id(2) @ H
        """
        result = super().tensor(other)
        return Circuit(len(result.dom), len(result.cod),
                       result.boxes, result.offsets, _fast=True)

    def dagger(self):
        """
        >>> print((CX >> SWAP).dagger())
        SWAP.dagger() >> CX.dagger()
        """
        result = super().dagger()
        return Circuit(len(result.dom), len(result.cod),
                       result.boxes, result.offsets, _fast=True)

    def normal_form(self, left=False):
        """
        >>> circuit = Id(1) @ X >> X @ Id(1)
        >>> print(circuit.normal_form())
        X @ Id(1) >> Id(1) @ X
        >>> print(circuit.normal_form(left=True))
        Id(1) @ X >> X @ Id(1)
        """
        result = super().normal_form(left=left)
        return Circuit(len(result.dom), len(result.cod),
                       result.boxes, result.offsets, _fast=True)

    @staticmethod
    def id(x):
        """
        >>> Circuit.id(2)
        Id(2)
        """
        return Id(x)

    @staticmethod
    def cups(x, y):
        """
        >>> Circuit.cups(PRO(1), PRO(1)).eval()
        Matrix(dom=Dim(2, 2), cod=Dim(1), array=[1.0, 0.0, 0.0, 1.0])
        >>> arr = Circuit.cups(PRO(2), PRO(2)).eval().array
        >>> for i in range(4): print(list(arr[i % 2, i // 2].flatten()))
        [1.0, 0.0, 0.0, 0.0]
        [0.0, 1.0, 0.0, 0.0]
        [0.0, 0.0, 1.0, 0.0]
        [0.0, 0.0, 0.0, 1.0]
        """
        if not isinstance(x, PRO) or not isinstance(y, PRO):
            raise ValueError("Expected PRO, got {} of type {} instead."
                             .format((repr(x), repr(y)), (type(x), type(y))))
        result = Id(x @ y)
        cup = CX >> Gate('H @ sqrt(2)', 1, [1, 1, 1, -1]) @ Id(1) >> Bra(0, 0)
        for i in range(1, len(x) + 1):
            result = result >> Id(len(x) - i) @ cup @ Id(len(x) - i)
        return result

    @staticmethod
    def caps(x, y):
        """
        >>> Circuit.caps(PRO(1), PRO(1)).eval()
        Matrix(dom=Dim(1), cod=Dim(2, 2), array=[1, 0, 0, 1])
        >>> arr = Circuit.caps(PRO(2), PRO(2)).eval().array
        >>> for i in range(4): print(list(arr[i % 2, i // 2].flatten()))
        [1, 0, 0, 0]
        [0, 1, 0, 0]
        [0, 0, 1, 0]
        [0, 0, 0, 1]
        """
        return Circuit.cups(x, y).dagger()

    def eval(self):
        """
        Evaluates the circuit as a discopy Matrix.
        """
        return MatrixFunctor({Ty(1): 2}, Quiver(lambda g: g.array))(self)

    def measure(self):
        """
        Applies the Born rule and outputs a stochastic matrix.
        The input maybe any circuit c, the output will be a numpy array
        with shape len(c.dom @ c.cod) * (2, )

        >>> c = Circuit(2, 2, [sqrt(2), H, Rx(0.5), CX], [0, 0, 1, 0])
        >>> m = c.measure()
        >>> list(np.round(m[0, 0].flatten()))
        [0.0, 1.0, 1.0, 0.0]
        >>> assert (Ket(1, 0) >> c >> Bra(0, 1)).measure() == m[1, 0, 0, 1]
        """
        def bitstring(i, length):
            return map(int, '{{:0{}b}}'.format(length).format(i))
        process = self.eval()
        states, effects = [], []
        states = [Ket(*bitstring(i, len(self.dom))).eval()
                  for i in range(2 ** len(self.dom))]
        effects = [Bra(*bitstring(j, len(self.cod))).eval()
                   for j in range(2 ** len(self.cod))]
        array = np.zeros(len(self.dom + self.cod) * (2, ))
        for state in states if self.dom else [Matrix.id(1)]:
            for effect in effects if self.cod else [Matrix.id(1)]:
                scalar = np.absolute((state >> process >> effect).array ** 2)
                array += scalar * (state.dagger() >> effect.dagger()).array
        return array

    def to_tk(self):
        """ Returns a pytket circuit.

        >>> circuit = Circuit(3, 3, [SWAP, Rx(0.25), CX], [0, 1, 1]).to_tk()
        >>> for g in circuit: print((g.op.get_type(), g.op.get_params()))
        (OpType.SWAP, [])
        (OpType.Rx, [0.25])
        (OpType.CX, [])
        """
        tk_circuit = tk.Circuit(len(self.dom))
        for gate, off in zip(self.gates, self.offsets):
            if isinstance(gate, Rx):
                tk_circuit.Rx(
                    gate.phase, *(off + i for i in range(len(gate.dom))))
            elif isinstance(gate, Rz):
                tk_circuit.Rz(
                    gate.phase, *(off + i for i in range(len(gate.dom))))
            else:
                tk_circuit.__getattribute__(gate.name)(
                    *(off + i for i in range(len(gate.dom))))
        return tk_circuit

    @staticmethod
    def from_tk(tk_circuit):
        """ Takes a pytket circuit and returns a planar circuit,
        SWAP gates are introduced when applying gates to non-adjacent qubits.

        >>> c1 = Circuit(3, 3, [SWAP, Rx(0.25), CX], [0, 1, 1])
        >>> c2 = Circuit.from_tk(c1.to_tk())
        >>> assert c1.normal_form() == c2.normal_form()
        """
        def gates_from_tk(tk_gate):
            name = tk_gate.op.get_type().name
            if name == 'Rx':
                return Rx(tk_gate.op.get_params()[0])
            if name == 'Rz':
                return Rz(tk_gate.op.get_params()[0])
            for gate in [SWAP, CX, H, S, T, X, Y, Z]:
                if name == gate.name:
                    return gate
            raise NotImplementedError
        gates, offsets = [], []
        for tk_gate in tk_circuit.get_commands():
            i_0 = tk_gate.qubits[0].index[0]
            for i, qubit in enumerate(tk_gate.qubits[1:]):
                if qubit.index[0] == i_0 + i + 1:
                    break  # gate applies to adjacent qubit already
                if qubit.index[0] < i_0 + i + 1:
                    for j in range(qubit.index, i_0 + i):
                        gates.append(SWAP)
                        offsets.append(j)
                    if qubit.index <= i_0:
                        i_0 -= 1
                else:
                    for j in range(qubit.index - i_0 + i - 1):
                        gates.append(SWAP)
                        offsets.append(qubit.index - j - 1)
            gates.append(gates_from_tk(tk_gate))
            offsets.append(i_0)
        return Circuit(tk_circuit.n_qubits, tk_circuit.n_qubits,
                       gates, offsets)

    @staticmethod
    def random(n_qubits, depth=3, gateset=None, seed=None):
        """ Returns a random Euler decomposition if n_qubits == 1,
        otherwise returns a random tiling with the given depth and gateset.

        >>> c = Circuit.random(1, seed=420)
        >>> print(c)  # doctest: +ELLIPSIS
        Rx(0.026...) >> Rz(0.781...) >> Rx(0.272...)
        >>> array = (c >> c.dagger()).eval().array
        >>> assert np.all(np.round(array) == np.identity(2))
        >>> print(Circuit.random(2, 2, gateset=[CX, H, T], seed=420))
        CX >> T @ Id(1) >> Id(1) @ T
        >>> print(Circuit.random(3, 2, gateset=[CX, H, T], seed=420))
        CX @ Id(1) >> Id(2) @ T >> H @ Id(2) >> Id(1) @ H @ Id(1) >> Id(2) @ H
        >>> print(Circuit.random(2, 1, gateset=[Rz, Rx], seed=420))
        Rz(0.6731171219152886) @ Id(1) >> Id(1) @ Rx(0.2726063832840899)
        """
        if seed is not None:
            _seed(seed)
        if n_qubits == 1:
            return Rx(_random()) >> Rz(_random()) >> Rx(_random())
        result = Id(n_qubits)
        for _ in range(depth):
            line, n_affected = Id(0), 0
            while n_affected < n_qubits:
                gate = choice(gateset if n_qubits - n_affected > 1
                              else [g for g in gateset
                                    if g is Rx or g is Rz or len(g.dom) == 1])
                if gate is Rx or gate is Rz:
                    gate = gate(_random())
                line = line @ gate
                n_affected += len(gate.dom)
            result = result >> line
        return result


class Id(Circuit):
    """ Implements identity circuit on n qubits.

    >>> c = CX @ H >> T @ SWAP
    >>> assert Id(3) >> c == c == c >> Id(3)
    """
    def __init__(self, n_qubits):
        """
        >>> assert Circuit.id(42) == Id(42) == Circuit(42, 42, [], [])
        """
        if isinstance(n_qubits, PRO):
            n_qubits = len(n_qubits)
        super().__init__(n_qubits, n_qubits, [], [], _fast=True)

    def __repr__(self):
        """
        >>> Id(42)
        Id(42)
        """
        return "Id({})".format(len(self.dom))

    def __str__(self):
        """
        >>> print(Id(42))
        Id(42)
        """
        return repr(self)


class Gate(Box, Circuit):
    """ Implements quantum gates as boxes in a circuit diagram.

    >>> CX
    Gate('CX', 2, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
    """
    def __init__(self, name, n_qubits, array=None, data=None, _dagger=False):
        """
        >>> g = CX
        >>> assert g.dom == g.cod == PRO(2)
        """
        if array is not None:
            self._array = np.array(array).reshape(2 * n_qubits * (2, ) or 1)
        Box.__init__(self, name, PRO(n_qubits), PRO(n_qubits),
                     data=data, _dagger=_dagger)
        Circuit.__init__(self, n_qubits, n_qubits, [self], [0], _fast=True)

    @property
    def array(self):
        """
        >>> list(X.array.flatten())
        [0, 1, 1, 0]
        """
        return self._array

    def __repr__(self):
        """
        >>> CX
        Gate('CX', 2, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        """
        if self._dagger:
            return repr(self.dagger()) + '.dagger()'
        return "Gate({}, {}, {}{})".format(
            repr(self.name), len(self.dom), list(self.array.flatten()),
            ', data=' + repr(self.data) if self.data else '')

    def dagger(self):
        """
        >>> print(CX.dagger())
        CX.dagger()
        >>> print(Rx(0.25).dagger())
        Rx(-0.25)
        >>> assert Rx(0.25).eval().dagger() == Rx(0.25).dagger().eval()
        """
        return Gate(self.name, len(self.dom), self.array,
                    data=self.data, _dagger=not self._dagger)


class Ket(Box, Circuit):
    """ Implements ket for a given bitstring.

    >>> Ket(1, 1, 0).eval()
    Matrix(dom=Dim(1), cod=Dim(2, 2, 2), array=[0, 0, 0, 0, 0, 0, 1, 0])
    """
    def __init__(self, *bitstring):
        """
        >>> g = Ket(1, 1, 0)
        """
        self.bitstring = bitstring
        Box.__init__(self, 'Ket({})'.format(', '.join(map(str, bitstring))),
                     PRO(0), PRO(len(bitstring)))
        Circuit.__init__(self, 0, len(bitstring), [self], [0], _fast=True)

    def __repr__(self):
        """
        >>> Ket(1, 1, 0)
        Ket(1, 1, 0)
        """
        return self.name

    def dagger(self):
        """
        >>> Ket(0, 1).dagger()
        Bra(0, 1)
        """
        return Bra(*self.bitstring)

    @property
    def array(self):
        """
        >>> Ket(0).eval()
        Matrix(dom=Dim(1), cod=Dim(2), array=[1, 0])
        >>> Ket(0, 1).eval()
        Matrix(dom=Dim(1), cod=Dim(2, 2), array=[0, 1, 0, 0])
        """
        matrix = Matrix(Dim(1), Dim(1), [1])
        for bit in self.bitstring:
            matrix = matrix @ Matrix(Dim(2), Dim(1), [0, 1] if bit else [1, 0])
        return matrix.array


class Bra(Box, Circuit):
    """ Implements bra for a given bitstring.

    >>> Bra(1, 1, 0).eval()
    Matrix(dom=Dim(2, 2, 2), cod=Dim(1), array=[0, 0, 0, 0, 0, 0, 1, 0])
    >>> assert all((Bra(x, y, z) << Ket(x, y, z)).eval() == 1
    ...            for x in [0, 1] for y in [0, 1] for z in [0, 1])
    """
    def __init__(self, *bitstring):
        """
        >>> g = Bra(1, 1, 0)
        """
        self.bitstring = bitstring
        Box.__init__(self, 'Bra({})'.format(', '.join(map(str, bitstring))),
                     PRO(len(bitstring)), PRO(0))
        Circuit.__init__(self, len(bitstring), 0, [self], [0], _fast=True)

    def __repr__(self):
        """
        >>> Bra(1, 1, 0)
        Bra(1, 1, 0)
        """
        return self.name

    def dagger(self):
        """
        >>> Bra(0, 1).dagger()
        Ket(0, 1)
        """
        return Ket(*self.bitstring)

    @property
    def array(self):
        """
        >>> Bra(0).eval()
        Matrix(dom=Dim(2), cod=Dim(1), array=[1, 0])
        >>> Bra(0, 1).eval()
        Matrix(dom=Dim(2, 2), cod=Dim(1), array=[0, 1, 0, 0])
        """
        return Ket(*self.bitstring).array


class Rz(Gate):
    """
    >>> assert np.all(Rz(0).array == np.identity(2))
    >>> assert np.allclose(Rz(0.5).array, Z.array)
    >>> assert np.allclose(Rz(0.25).array, S.array)
    >>> assert np.allclose(Rz(0.125).array, T.array)
    """
    def __init__(self, phase):
        """
        >>> Rz(0.25)
        Rz(0.25)
        """
        self._phase = phase
        super().__init__('Rz', 1)

    @property
    def phase(self):
        """
        >>> Rz(0.25).phase
        0.25
        """
        return self._phase

    @property
    def name(self):
        """
        >>> assert str(Rz(0.125)) == repr(Rz(0.125)) == Rz(0.125).name
        """
        return 'Rz({})'.format(self.phase)

    def __repr__(self):
        """
        >>> assert str(Rz(0.125)) == repr(Rz(0.125))
        """
        return self.name

    def dagger(self):
        """
        >>> assert Rz(0.125).dagger().eval() == Rz(0.125).eval().dagger()
        """
        return Rz(-self.phase)

    @property
    def array(self):
        """
        >>> assert np.allclose(Rz(-1).array, np.identity(2))
        >>> assert np.allclose(Rz(0).array, np.identity(2))
        >>> assert np.allclose(Rz(1).array, np.identity(2))
        """
        theta = 2 * np.pi * self.phase
        return np.array([[1, 0], [0, np.exp(1j * theta)]])


class Rx(Gate):
    """
    >>> assert np.all(Rx(0).array == np.identity(2))
    >>> assert np.all(np.round(Rx(0.5).array) == X.array)
    """
    def __init__(self, phase):
        """
        >>> Rx(0.25)
        Rx(0.25)
        """
        self._phase = phase
        super().__init__('Rx', 1)

    @property
    def phase(self):
        """
        >>> Rx(0.25).phase
        0.25
        """
        return self._phase

    @property
    def name(self):
        """
        >>> assert str(Rx(0.125)) == Rx(0.125).name
        """
        return 'Rx({})'.format(self.phase)

    def __repr__(self):
        """
        >>> assert str(Rx(0.125)) == repr(Rx(0.125))
        """
        return self.name

    def dagger(self):
        """
        >>> assert Rx(0.125).dagger().eval() == Rx(0.125).eval().dagger()
        """
        return Rx(-self.phase)

    @property
    def array(self):
        """
        >>> assert np.allclose(Rx(0).array, np.identity(2))
        >>> assert np.allclose(np.round(Rx(0.5).array), X.array)
        >>> assert np.allclose(np.round(Rx(-1).array), np.identity(2))
        >>> assert np.allclose(np.round(Rx(1).array), np.identity(2))
        >>> assert np.allclose(np.round(Rx(2).array), np.identity(2))
        """
        half_theta = np.pi * self.phase
        global_phase = np.exp(1j * half_theta)
        sin, cos = np.sin(half_theta), np.cos(half_theta)
        return global_phase * np.array([[cos, -1j * sin], [-1j * sin, cos]])


class CircuitFunctor(RigidFunctor):
    """ Implements funtors from monoidal categories to circuits

    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> f, g, h = Box('f', x, y + z), Box('g', z, y), Box('h', y + z, x)
    >>> d = (f @ Diagram.id(z)
    ...       >> Diagram.id(y) @ g @ Diagram.id(z)
    ...       >> Diagram.id(y) @ h)
    >>> ob = {x: 2, y: 1, z: 1}
    >>> ar = {f: SWAP, g: Rx(0.25), h: CX}
    >>> F = CircuitFunctor(ob, ar)
    >>> print(F(d))
    SWAP @ Id(1) >> Id(1) @ Rx(0.25) @ Id(1) >> Id(1) @ CX
    """
    def __init__(self, ob, ar):
        """
        >>> F = CircuitFunctor({}, {})
        """
        super().__init__({x: PRO(y) for x, y in ob.items()}, ar,
                         ob_cls=PRO, ar_cls=Circuit)

    def __repr__(self):
        """
        >>> CircuitFunctor({}, {})
        CircuitFunctor(ob={}, ar={})
        """
        return "CircuitFunctor(ob={}, ar={})".format(
            repr({x: len(y) for x, y in self.ob.items()}), repr(self.ar))

    def __call__(self, diagram):
        """
        >>> x = Ty('x')
        >>> F = CircuitFunctor({x: 1}, {})
        >>> assert isinstance(F(Diagram.id(x)), Circuit)
        """
        result = super().__call__(diagram)
        if isinstance(diagram, Ty):
            return PRO(len(result))
        if isinstance(diagram, Diagram):
            return Circuit(
                len(result.dom), len(result.cod),
                result.boxes, result.offsets, _fast=True)
        return result


def sqrt(x):
    """
    >>> sqrt(2)  # doctest: +ELLIPSIS
    Gate('sqrt(2)', 0, [1.41...])
    """
    return Gate('sqrt({})'.format(x), 0, np.sqrt(x))


SWAP = Gate('SWAP', 2, [1, 0, 0, 0,
                        0, 0, 1, 0,
                        0, 1, 0, 0,
                        0, 0, 0, 1])
CX = Gate('CX', 2, [1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1,
                    0, 0, 1, 0])
H = Gate('H', 1, 1 / np.sqrt(2) * np.array([1, 1, 1, -1]))
S = Gate('S', 1, [1, 0, 0, 1j])
T = Gate('T', 1, [1, 0, 0, np.exp(1j * np.pi / 4)])
X = Gate('X', 1, [0, 1, 1, 0])
Y = Gate('Y', 1, [0, -1j, 1j, 0])
Z = Gate('Z', 1, [1, 0, 0, -1])
