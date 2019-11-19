import numpy as np
from discopy.cat import fold, Quiver
from discopy.moncat import Ob, Ty, Box, Diagram, MonoidalFunctor
from discopy.matrix import Dim, Matrix, MatrixFunctor


class PRO(Ty):
    """ Implements the objects of a PRO, i.e. a non-symmetric PROP.
    Wraps a natural number n into a unary type Ty(1, ..., 1) of length n.

    >>> PRO(1) @ PRO(1)
    PRO(2)
    >>> assert PRO(3) == Ty(1, 1, 1)
    """
    def __init__(self, n):
        """
        >>> list(PRO(0))
        []
        >>> list(PRO(1))
        [Ob(1)]
        >>> assert all(len(PRO(n)) == n for n in range(5))
        """
        n = n if isinstance(n, int) else len(n)
        super().__init__(*(n * [Ob(1)]))

    def __repr__(self):
        """
        >>> PRO(0), PRO(1)
        (PRO(0), PRO(1))
        """
        return "PRO({})".format(len(self))

    def __str__(self):
        """
        >>> print(PRO(2 * 3 * 7))
        PRO(42)
        """
        return repr(self)

    def __add__(self, other):
        """
        >>> sum((PRO(n) for n in range(5)), PRO(0))
        PRO(10)
        """
        return PRO(len(self) + len(other))

    def __getitem__(self, key):
        """
        >>> PRO(42)[2:4]
        PRO(2)
        >>> assert all(PRO(42)[i] == Ob(1) for i in range(42))
        """
        if isinstance(key, slice):
            return PRO(len(super().__getitem__(key)))
        return super().__getitem__(key)

class Circuit(Diagram):
    """ Implements quantum circuits as diagrams.

    >>> circuit = CX >> CX >> CX >> CX >> CX >> CX
    >>> assert np.all(circuit.eval() == Id(2).eval())
    """
    def __init__(self, dom, cod, gates, offsets):
        """
        >>> c = Circuit(2, 2, [CX, CX], [0, 0])
        """
        self._gates = gates
        super().__init__(PRO(dom), PRO(cod), gates, offsets)

    @property
    def gates(self):
        """
        >>> Circuit(2, 2, [CX, CX], [0, 0]).gates
        [Gate('CX', 2), Gate('CX', 2)]
        """
        return self._gates

    def __repr__(self):
        """
        >>> Circuit(2, 2, [CX, CX], [0, 0])
        Circuit(2, 2, [Gate('CX', 2), Gate('CX', 2)], [0, 0])
        """
        return "Circuit({}, {}, {}, {})".format(
            len(self.dom), len(self.cod), self.gates, self.offsets)

    def then(self, other):
        """
        >>> CX >> CX
        Circuit(2, 2, [Gate('CX', 2), Gate('CX', 2)], [0, 0])
        """
        r = super().then(other)
        return Circuit(len(r.dom), len(r.cod), r.boxes, r.offsets)

    def tensor(self, other):
        """
        >>> CX @ H
        Circuit(3, 3, [Gate('CX', 2), Gate('H', 1)], [0, 2])
        """
        r = super().tensor(other)
        return Circuit(len(r.dom), len(r.cod), r.boxes, r.offsets)

    def dagger(self):
        """
        >>> (CX >> CX).dagger()
        Circuit(2, 2, [Gate('CX', 2).dagger(), Gate('CX', 2).dagger()], [0, 0])
        """
        r = super().dagger()
        return Circuit(len(r.dom), len(r.cod), r.boxes, r.offsets)

    @staticmethod
    def id(n):
        """
        >>> Circuit.id(2)
        Id(2)
        """
        return Id(n)

    def eval(self):
        """ Evaluates the circuit as a numpy array.

        >>> for U in [SWAP, X, Y, Z, S >> S, CX >> CX >> CX]:
        ...     assert np.all((U >> U.dagger()).eval() == Id(len(U.dom)).eval())
        >>> for U in [H, T >> T >> T >> T]:
        ...     assert np.allclose((U >> U.dagger()).eval(), Id(len(U.dom)).eval())
        """
        return MatrixFunctor({PRO(1): 2}, Quiver(lambda x: x.to_array()))(self)

    def to_tk(self):
        """ Returns a pytket circuit.

        >>> c1 = Circuit(3, 3, [SWAP, Rx(0.25), CX], [0, 1, 1])
        >>> c2 = Circuit.from_tk(c1.to_tk())
        >>> assert c1 == c2  # This works as long as there are no interchangers!
        """
        import pytket as tk
        c = tk.Circuit(len(self.dom))
        for g, n in zip(self.gates, self.offsets):
            if g.data:
                assert g.name in ['Rx', 'Rz']
                c.__getattribute__(g.name)(
                    *(n + i for i in range(len(g.dom))), g.data['phase'])
            else:
                c.__getattribute__(g.name)(*(n + i for i in range(len(g.dom))))
        return c

    @staticmethod
    def from_tk(c):
        """ Takes a pytket circuit and returns a planar circuit,
        SWAP gates are introduced when applying gates to non-adjacent qubits.

        >>> c = Circuit(3, 3, [SWAP, Rx(0.25), CX], [0, 1, 1]).to_tk()
        >>> list(c)
        [SWAP q[0], q[1];, Rx(0.25PI) q[1];, CX q[1], q[2];]
        """
        gates_to_pytket = lambda g: Gate(g.op.get_type().name, len(g.qubits),
            data={'phase': g.op.get_params()[0]} if g.op.get_params() else {})
        gates, offsets = [], []
        for g in c.get_commands():
            i0 = g.qubits[0].index
            for i, q in enumerate(g.qubits[1:]):
                if q.index == i0 + i + 1:
                    break  # gate applies to adjacent qubit already
                elif q.index < i0 + i + 1:
                    for j in range(q.index, i0 + i):
                        gates.append(SWAP)
                        offsets.append(j)
                    if q.index <= i0:
                        i0 -= 1  # we just swapped q to the right of q0
                elif q.index > i0 + i + 1:
                    for j in range(q.index - i0 + i - 1):
                        gates.append(SWAP)
                        offsets.append(q.index - j - 1)
            gates.append(gates_to_pytket(g))
            offsets.append(i0)
        return Circuit(c.n_qubits, c.n_qubits, gates, offsets)

    def Euler(a, b, c):
        """ Returns a 1-qubit Euler decomposition with angles 2 * pi * a, b, c.

        >>> print(Circuit.Euler(1, 0, 1))
        Rx(1) >> Rz(0) >> Rx(1)
        >>> assert np.allclose(Circuit.Euler(1, 0, 1).eval(), Id(1))
        """
        return Circuit(1, 1, [Rx(a), Rz(b), Rx(c)], [0, 0, 0])

    @staticmethod
    def random(n_qubits, depth=0, gateset=[], seed=None):
        """ Returns a random Euler decomposition if n_qubits == 1,
        otherwise returns a random tiling with the given depth and gateset.

        >>> c = Circuit.random(1, seed=420)
        >>> [g.data['phase'] for g in c.gates]
        [0.026343380459525556, 0.7813690555430765, 0.2726063832840899]
        >>> c.eval().array[0]
        array([ 0.64067536+0.06124486j, -0.05310971-0.76352047j])
        >>> c.eval().array[1]
        array([-0.73833785-0.20159704j,  0.06540048-0.6402645j ])
        >>> print(Circuit.random(2, 2, [CX, H, T], seed=420))
        CX >> T @ Id(1) >> Id(1) @ T
        >>> print(Circuit.random(3, 2, [CX, H, T], seed=420))
        CX @ Id(1) >> Id(2) @ T >> H @ Id(2) >> Id(1) @ H @ Id(1) >> Id(2) @ H
        """
        import random
        if seed:
            random.seed(seed)
        if n_qubits == 1:
            return Circuit.Euler(
                random.random(), random.random(), random.random())
        U = Id(n_qubits)
        for d in range(depth):
            L, affected = Id(0), 0
            while affected <= n_qubits - 2:
                gate = random.choice(gateset)
                L = L @ gate
                affected += gate.n_qubits
            if n_qubits - affected == 1:
                L = L @ random.choice([g for g in gateset if g.n_qubits == 1])
            U = U >> L
        return U

class Id(Circuit):
    """ Implements identity circuit on n qubits.

    >>> c = CX @ H >> T @ SWAP
    >>> assert Id(3) >> c == c == c >> Id(3)
    """
    def __init__(self, n):
        """
        >>> assert Circuit.id(42) == Id(42) == Circuit(42, 42, [], [])
        """
        if isinstance(n, PRO):
            n = len(n)
        Diagram.__init__(self, PRO(n), PRO(n), [], [])

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
    Gate('CX', 2)
    >>> Rx(0.25)
    Gate('Rx', 1, data={'phase': 0.25})
    """
    def __init__(self, name, n_qubits, dagger=False, data={}):
        """
        >>> g = Gate('Rx', 1, data={'phase': 0.25})
        >>> assert g == Rx(0.25)
        >>> assert g.dom == g.cod == PRO(1)
        """
        self.n_qubits = n_qubits
        Box.__init__(self, name, PRO(n_qubits), PRO(n_qubits), dagger, data)

    def __repr__(self):
        """
        >>> Gate('CX', 2)
        Gate('CX', 2)
        >>> Gate('Rx', 1, data={'phase': 0.25})
        Gate('Rx', 1, data={'phase': 0.25})
        """
        return "Gate({}, {}{}){}".format(repr(self.name), len(self.dom),
            ', data=' + repr(self.data) if self.data else '',
            '.dagger()' if self._dagger else '')

    def __str__(self):
        """
        >>> print(Gate('CX', 2))
        CX
        >>> print(Gate('Rx', 1, data={'phase': 0.25}))
        Rx(0.25)
        """
        if self.name in ['Rx', 'Rz']:
            return "{}({})".format(self.name, self.data['phase'])
        return self.name

    def dagger(self):
        """
        >>> Gate('CX', 2).dagger()
        Gate('CX', 2).dagger()
        >>> Gate('Rx', 1, data={'phase': 0.25}).dagger()
        Gate('Rx', 1, data={'phase': 0.25}).dagger()
        """
        return Gate(self.name, self.n_qubits,
                    dagger=not self._dagger, data=self.data)

    def to_array(self):
        """
        >>> assert np.all((Ket(0, 0) >> SWAP).eval() == Ket(0, 0).eval())
        >>> assert np.all((Ket(0, 1) >> SWAP).eval() == Ket(1, 0).eval())
        >>> assert np.all((Ket(1, 0) >> SWAP).eval() == Ket(0, 1).eval())
        >>> assert np.all((Ket(1, 1) >> SWAP).eval() == Ket(1, 1).eval())

        >>> assert np.all((Ket(0, 0) >> CX).eval() == Ket(0, 0).eval())
        >>> assert np.all((Ket(0, 1) >> CX).eval() == Ket(0, 1).eval())
        >>> assert np.all((Ket(1, 0) >> CX).eval() == Ket(1, 1).eval())
        >>> assert np.all((Ket(1, 1) >> CX).eval() == Ket(1, 0).eval())
        """
        if self.name in ['Rx', 'Rz']:
            theta = 2 * np.pi * float(self.data['phase'])
            if self.name == 'Rz':
                return [1, 0, 0, np.exp(1j * theta)]
            elif self.name == 'Rx':
                return [np.cos(theta / 2), -1j * np.sin(theta / 2),
                        -1j * np.sin(theta / 2), np.cos(theta / 2)]
        gate_to_array = {
            'H': 1 / np.sqrt(2) * np.array([1, 1, 1, -1]),
            'S': [1, 0, 0, 1j],
            'T': [1, 0, 0, np.exp(1j * np.pi / 4)],
            'X': [0, 1, 1, 0],
            'Y': [0, -1j, 1j, 0],
            'Z': [1, 0, 0, -1],
            'CX': [1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 0, 1,
                   0, 0, 1, 0],
            'SWAP': [1, 0, 0, 0,
                     0, 0, 1, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1]
            }
        if self.name not in gate_to_array.keys():
            raise NotImplementedError
        return gate_to_array[self.name]

class Ket(Gate):
    """ Implements ket of a given bitstring.

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

    def __repr__(self):
        """
        >>> Ket(1, 1, 0)
        Ket(1, 1, 0)
        """
        return self.name

    def to_array(self):
        """
        >>> Ket(0).eval()
        Matrix(dom=Dim(1), cod=Dim(2), array=[1, 0])
        >>> Ket(0, 1).eval()
        Matrix(dom=Dim(1), cod=Dim(2, 2), array=[0, 1, 0, 0])
        """
        m = Matrix(Dim(1), Dim(1), [1])
        for b in self.bitstring:
            m = m @ Matrix(Dim(1), Dim(2), [0, 1] if b else [1, 0])
        return m.array

class Bra(Gate):
    """
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

    def __repr__(self):
        """
        >>> Bra(1, 1, 0)
        Bra(1, 1, 0)
        """
        return self.name

    def to_array(self):
        """
        >>> Bra(0).eval()
        Matrix(dom=Dim(2), cod=Dim(1), array=[1, 0])
        >>> Bra(0, 1).eval()
        Matrix(dom=Dim(2, 2), cod=Dim(1), array=[0, 1, 0, 0])
        """
        m = Matrix(Dim(1), Dim(1), [1])
        for b in self.bitstring:
            m = m @ Matrix(Dim(2), Dim(1), [0, 1] if b else [1, 0])
        return m.array

class CircuitFunctor(MonoidalFunctor):
    """ Implements funtors from monoidal categories to circuits

    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> f, g, h = Box('f', x, y + z), Box('g', z, y), Box('h', y + z, x)
    >>> d = (f @ Diagram.id(z)
    ...       >> Diagram.id(y) @ g @ Diagram.id(z)
    ...       >> Diagram.id(y) @ h)
    >>> ob = {x: PRO(2), y: PRO(1), z: PRO(1)}
    >>> ar = {f: SWAP, g: Rx(0.25), h: CX}
    >>> F = CircuitFunctor(ob, ar)
    >>> c1 = SWAP @ Id(1) >> Id(1) @ Rx(0.25) @ Id(1) >> Id(1) @ CX
    >>> assert F(d) == c1
    """
    def __call__(self, d):
        r = super().__call__(d)
        if isinstance(d, Diagram):
            return Circuit(len(r.dom), len(r.cod), r.boxes, r.offsets)
        return r

SWAP, CX = Gate('SWAP', 2), Gate('CX', 2)
H, S, T = Gate('H', 1), Gate('S', 1), Gate('T', 1)
X, Y, Z = Gate('X', 1), Gate('Y', 1), Gate('Z', 1)
Rx = lambda phase: Gate('Rx', 1, data={'phase': phase})
Rz = lambda phase: Gate('Rz', 1, data={'phase': phase})
