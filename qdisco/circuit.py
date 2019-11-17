import math
import numpy as np
from random import random, randint
from functools import reduce as fold
import pytket as tk
from discopy.cat import Quiver
from discopy.moncat import Ty, Box, Diagram, MonoidalFunctor
from discopy.matrix import MatrixFunctor


#  Turns natural numbers into types encoded in unary.
PRO = lambda n: sum(n * [Ty(1)], Ty())


class Circuit(Diagram):
    """ Implements quantum circuits as diagrams

    >>> SWAP, CX = Gate('SWAP', 2), Gate('CX', 2)
    >>> H, S, T = Gate('H', 1), Gate('S', 1), Gate('T', 1)
    >>> X, Y, Z = Gate('X', 1), Gate('Y', 1), Gate('Z', 1)
    >>> assert isinstance(H >> S >> T, Circuit)
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

    """ Interface with pytket

    >>> c1_tk = tk.Circuit(3).SWAP(0, 1).Rx(1, 0.25).CX(1, 2)
    >>> c1 = Circuit.from_tk(c1_tk)
    >>> assert c1 == Circuit(3, [SWAP, Rx(0.25), CX], [0, 1, 1])
    >>> c2_tk = c1.to_tk()
    >>> c2 = Circuit.from_tk(c2_tk)
    >>> assert not c1_tk == c2_tk  # Equality of circuits in tket doesn't work!
    >>> assert c1 == c2  # This works as long as there are no interchangers!
    """
    def to_tk(self):
        c = tk.Circuit(len(self.dom))
        for g, n in zip(self.boxes, self.offsets):
            c.__getattribute__(g.name)(
                *[n + i for i in range(len(g.dom))], *(g.data or []))
        return c

    @staticmethod
    def from_tk(c):
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
            gates.append(GATES_TO_PYTKET[g])
            offsets.append(i0)
        return Circuit(c.n_qubits, gates, offsets)

    """ 1-qubit Euler ansatz

    """
    def Euler(x0, z, x1):
        return Circuit(1, [Rx(x0), Rz(z), Rx(x1)], [0, 0, 0])

    """ Random Tiling ansatz

    """
    @staticmethod
    def random(n_qubits, depth, gateset=[]):
        if n_qubits == 1:
            return Circuit.Euler(random(), random(), random())
        else:
        	U = Id(n_qubits)

        	g1 = [] #the single-qubit gates
        	for ii in range(len(gateset)):
        		if gateset[ii].n_qubits == 1:
        			g1.append(gateset[ii])

        	for d in range(depth):

        		#make a random layer of gates
        		l = [] # initialise empty layer
        		nqa = 0 # number of qubits affected

        		while nqa <= n_qubits-2:
        			#randomly choose a gate to apply
        			#and append to layer
        			gate = gateset[randint(0, len(gateset)-1)]
        			l.append(gate)

        			#count how many qubits affected
        			nqa += gate.n_qubits

        		if n_qubits-nqa == 1:
        			#which single-qubit gate to apply
        			#on the nth qubit in case it has been left unaffected
        			l.append(g1[randint(0, len(g1)-1)])
        			nqa += 1

        		#circuit of this layer is
        		#tensor product of all the gates in layer
        		Ul = l[0]
        		for ll in range(1, len(l)):
        			Ul = Ul @ l[ll]

        		U = U >> Ul

        	return U

class Id(Circuit):
    """ Implements identity circuits

    >>> Id(3)
    Circuit(3, [], [])
    >>> assert np.all(Id(2).eval() == (CX >> CX).eval())
    """
    def __init__(self, n_qubits):
        if isinstance(n_qubits, Ty):
            super().__init__(len(n_qubits), [], [])
        assert isinstance(n_qubits, int)
        super().__init__(n_qubits, [], [])

class Gate(Box, Circuit):
    """ Gates are generating Circuits

    >>> SWAP, CX = Gate('SWAP', 2), Gate('CX', 2)
    >>> H, S, T = Gate('H', 1), Gate('S', 1), Gate('T', 1)
    >>> X, Y, Z = Gate('X', 1), Gate('Y', 1), Gate('Z', 1)
    >>> Rx = lambda phase: Gate('Rx', 1, data=[phase])
    >>> Rz = lambda phase: Gate('Rz', 1, data=[phase])
    """
    def __init__(self, name, n_qubits, dagger=False, data=[]):
        self.n_qubits = n_qubits
        if isinstance(n_qubits, Ty):
            Box.__init__(self, name, n_qubits, n_qubits, dagger, data)
        else:
            Box.__init__(self, name, PRO(n_qubits), PRO(n_qubits), dagger, data)

    def __repr__(self):
        return "Gate({}, {}{}){}".format(repr(self.name), len(self.dom),
            ', ' + repr(self.data) if self.data else '',
            '.dagger()' if self._dagger else '')

    def dagger(self):
        return Gate(self.name, self.n_qubits,
                    dagger=not self._dagger, data=self.data)

class CircuitFunctor(MonoidalFunctor):
    """ Implements funtors from monoidal categories to circuits

    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> f, g, h = Box('f', x, y + z), Box('g', z, y), Box('h', y + z, x)
    >>> d = (f @ Diagram.id(z)
    ...       >> Diagram.id(y) @ g @ Diagram.id(z)
    ...       >> Diagram.id(y) @ h)
    >>> F = CircuitFunctor({x: PRO(2), y: PRO(1), z: PRO(1)}, {f: SWAP, g: Rx(0.25), h: CX})
    >>> c1_tk = tk.Circuit(3).SWAP(0, 1).Rx(1, 0.25).CX(1, 2)
    >>> c1 = Circuit.from_tk(c1_tk)
    >>> assert F(d) == c1
    """
    def __call__(self, d):
        r = super().__call__(d)
        if isinstance(d, Diagram):
            return Circuit(len(r.dom), r.boxes, r.offsets)
        return r

# Quantum Gates:
SWAP, CX = Gate('SWAP', 2), Gate('CX', 2)
H, S, T = Gate('H', 1), Gate('S', 1), Gate('T', 1)
X, Y, Z = Gate('X', 1), Gate('Y', 1), Gate('Z', 1)
Rx = lambda phase: Gate('Rx', 1, data=[phase])
Rz = lambda phase: Gate('Rz', 1, data=[phase])

#  Gates are unitaries, bras and kets are not. They are only boxes for now.
Ket = lambda b: Box('ket' + str(b), PRO(0), PRO(1))
Bra = lambda b: Box('bra' + str(b), PRO(1), PRO(0))
Kets = lambda b, n: fold(lambda x, y: x @ y, n * [Ket(b)])
Bras = lambda b, n: fold(lambda x, y: x @ y, n * [Bra(b)])

def gates_to_numpy(g):
    if g.name == 'ket0' or g.name == 'bra0':
        return [0, 1]
    elif g.name == 'ket1' or g.name == 'bra1':
        return [1, 0]
    elif g.name == 'S':
        return [1, 0,
                0, 1j]
    elif g.name == 'T':
        return [1, 0, 0, np.exp(1j * np.pi / 4)]
    elif g.name == 'H':
        return 1 / np.sqrt(2) * np.array([1, 1, 1, -1])
    elif g.name == 'X':
        return [0, 1,
                1, 0]
    elif g.name == 'Y':
        return [0, -1j, 1j, 0]
    elif g.name == 'Z':
        return [1, 0, 0, -1]
    elif g.name in ['Rx', 'Rz']:
        theta = 2 * np.pi * float(g.data[0])
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

EVAL = MatrixFunctor({PRO(1): 2}, Quiver(gates_to_numpy))

GATES_TO_PYTKET = Quiver(lambda g: Gate(
    g.op.get_type().name, len(g.qubits), data=g.op.get_params()))


# Permutations

def Permutation(n_qubits, perm):
    assert set(range(n_qubits)) == set(perm)
    gates = []
    offsets = []
    for i in range(n_qubits):
        if i >= perm[i]:
            pass
        else:
            num_swaps = perm[i] - i
            gates += [Gate('SWAP', 2) for x in range(num_swaps)]
            offsets += range(i, perm[i])[::-1]
    return Circuit(n_qubits, gates, offsets)

# The Generalized CX gate returns cups/caps if pre/post-composed with bras/kets

def GCX(n):
    perm = []
    for i in range(n):
        perm += [i, 2*n - 1 - i]
    SWAPS = Permutation(2*n, perm)
    CNOTS = Circuit(0, [], [])
    for i in range(n):
        CNOTS = CNOTS @ CX
    SWAPS_inv = Circuit(SWAPS.n_qubits, SWAPS.boxes[::-1], SWAPS.offsets[::-1])
    return SWAPS >> CNOTS >> SWAPS_inv
