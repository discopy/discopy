import numpy as np
from discopy.cat import fold, Quiver
from discopy.moncat import Ty, Box, Diagram, MonoidalFunctor
from discopy.matrix import MatrixFunctor


#  Turns natural numbers into types encoded in unary.
PRO = lambda n: sum(n * [Ty(1)], Ty())


class Circuit(Diagram):
    """ Implements quantum circuits as diagrams.
    >>> circuit = CX >> SWAP >> CX >> SWAP >> CX
    >>> assert np.all(circuit.eval() == SWAP.eval())
    >>> for U in [SWAP, X, Y, Z, S >> S, CX >> CX >> CX]:
    ...     assert np.all((U >> U.dagger()).eval() == Circuit.id(U.n_qubits).eval())
    >>> for U in [H, T >> T >> T >> T]:
    ...     assert np.allclose((U >> U.dagger()).eval(), Circuit.id(U.n_qubits).eval())
    """
    def __init__(self, n_qubits, gates, offsets):
        self.n_qubits = n_qubits
        super().__init__(PRO(n_qubits), PRO(n_qubits), gates, offsets)

    def __repr__(self):
        return "Circuit({}, {}, {})".format(
            len(self.dom), self.boxes, self.offsets)

    def then(self, other):
        result = super().then(other)
        return Circuit(len(result.dom), result.boxes, result.offsets)

    def tensor(self, other):
        result = super().tensor(other)
        return Circuit(len(result.dom), result.boxes, result.offsets)

    def dagger(self):
        return Circuit(len(self.dom), [g.dagger() for g in self.boxes[::-1]],
                       self.offsets[::-1])

    @staticmethod
    def id(n_qubits):
        return Id(n_qubits)

    def eval(self):
        return EVAL(self)

    def to_tk(self):
        """ Interface with pytket

        >>> import pytket as tk
        >>> c1_tk = tk.Circuit(3).SWAP(0, 1).Rx(1, 0.25).CX(1, 2)
        >>> c1 = Circuit.from_tk(c1_tk)
        >>> assert c1 == Circuit(3, [SWAP, Rx(0.25), CX], [0, 1, 1])
        >>> c2_tk = c1.to_tk()
        >>> c2 = Circuit.from_tk(c2_tk)
        >>> assert not c1_tk == c2_tk  # Equality of circuits in tket doesn't work!
        >>> assert c1 == c2  # This works as long as there are no interchangers!
        """
        import pytket as tk
        c = tk.Circuit(len(self.dom))
        for g, n in zip(self.boxes, self.offsets):
            if g.data:
                c.__getattribute__(g.name)(*(n + i for i in range(len(g.dom))),
                                           g.data['phase'])
            else:
                c.__getattribute__(g.name)(*(n + i for i in range(len(g.dom))))
        return c

    @staticmethod
    def from_tk(c):
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
        return Circuit(c.n_qubits, gates, offsets)

    def Euler(a, b, c):
        """ Returns a 1-qubit Euler decomposition with angles 2 * pi * a, b, c.
        """
        return Circuit(1, [Rx(a), Rz(b), Rx(c)], [0, 0, 0])

    @staticmethod
    def random(n_qubits, depth=0, gateset=[]):
        """ Returns a random Euler decomposition if n_qubits == 1,
        otherwise returns a random tiling with the given depth and gateset.
        """
        from random import random, choice
        if n_qubits == 1:
            return Circuit.Euler(random(), random(), random())
        U = Id(n_qubits)
        for d in range(depth):
            L, affected = Id(0), 0
            while affected <= n_qubits - 2:
                gate = choice(gateset)
                L = L @ gate
                affected += gate.n_qubits
            if n_qubits - affected == 1:
                L = L @ choice([g for g in gateset if g.n_qubits == 1])
            U = U >> L
        return U

class Id(Circuit):
    """ Implements identity circuits.
    """
    def __init__(self, n_qubits):
        if isinstance(n_qubits, Ty):
            super().__init__(len(n_qubits), [], [])
        assert isinstance(n_qubits, int)
        super().__init__(n_qubits, [], [])

class Gate(Box, Circuit):
    """ Implements quantum gates.
    """
    def __init__(self, name, n_qubits, dagger=False, data={}):
        self.n_qubits = n_qubits
        Box.__init__(self, name, PRO(n_qubits), PRO(n_qubits), dagger, data)

    def __repr__(self):
        return "Gate({}, {}{}){}".format(repr(self.name), len(self.dom),
            ', ' + repr(self.data) if self.data else '',
            '.dagger()' if self._dagger else '')

    def dagger(self):
        return Gate(self.name, self.n_qubits,
                    dagger=not self._dagger, data=self.data)

def gate_to_numpy(g):
    if g.name == 'ket_0' or g.name == 'bra_0':
        return [0, 1]
    elif g.name == 'ket_1' or g.name == 'bra_1':
        return [1, 0]
    elif g.name == 'H':
        return 1 / np.sqrt(2) * np.array([1, 1, 1, -1])
    elif g.name == 'S':
        return [1, 0, 0, 1j]
    elif g.name == 'T':
        return [1, 0, 0, np.exp(1j * np.pi / 4)]
    elif g.name == 'X':
        return [0, 1, 1, 0]
    elif g.name == 'Y':
        return [0, -1j, 1j, 0]
    elif g.name == 'Z':
        return [1, 0, 0, -1]
    elif g.name in ['Rx', 'Rz']:
        theta = 2 * np.pi * float(g.data['phase'])
        if g.name == 'Rz':
            return [1, 0, 0, np.exp(1j * theta)]
        elif g.name == 'Rx':
            return [np.cos(theta / 2), -1j * np.sin(theta / 2),
                    -1j * np.sin(theta / 2), np.cos(theta / 2)]
    elif g.name == 'CX':
        return [1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 0, 1,
                0, 0, 1, 0]
    elif g.name == 'SWAP':
        return [1, 0, 0, 0,
                0, 0, 1, 0,
                0, 1, 0, 0,
                0, 0, 0, 1]
    raise NotImplementedError

EVAL = MatrixFunctor({PRO(1): 2}, Quiver(gate_to_numpy))

SWAP, CX = Gate('SWAP', 2), Gate('CX', 2)
H, S, T = Gate('H', 1), Gate('S', 1), Gate('T', 1)
X, Y, Z = Gate('X', 1), Gate('Y', 1), Gate('Z', 1)
Rx = lambda phase: Gate('Rx', 1, data={'phase': phase})
Rz = lambda phase: Gate('Rz', 1, data={'phase': phase})

#  Gates are unitaries, bras and kets are not. They are only boxes for now.
Ket = lambda b: Box('ket' + str(b), PRO(0), PRO(1))
Bra = lambda b: Box('bra' + str(b), PRO(1), PRO(0))
Kets = lambda b, n: fold(lambda x, y: x @ y, n * [Ket(b)])
Bras = lambda b, n: fold(lambda x, y: x @ y, n * [Bra(b)])
