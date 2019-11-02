import math
import numpy as np
from random import random
from functools import reduce as fold
import pyzx as zx
import pytket as tk
from pytket.pyzx import pyzx_to_tk, tk_to_pyzx
from gates import GATES_TO_NUMPY
from moncat import Type, Diagram, Box, MonoidalFunctor, NumpyFunctor


PYTKET_GATES = tk.OpType.__entries.keys()

#  Turns natural numbers into types encoded in unary.
PRO = lambda n: sum(n * [Type(1)]) or Type()

EVAL = NumpyFunctor({PRO(1): 2}, GATES_TO_NUMPY)

class Circuit(Diagram):
    def __init__(self, n_qubits, gates, offsets):
        self.n_qubits = n_qubits
        super().__init__(PRO(n_qubits), PRO(n_qubits), gates, offsets)

    def __repr__(self):
        return "Circuit({}, {}, {})".format(
            len(self.dom), self.boxes, self.offsets)

    def then(self, other):
        assert isinstance(other, Circuit)
        r = super().then(other)
        return Circuit(len(r.dom), r.boxes, r.offsets)

    def tensor(self, other):
        assert isinstance(other, Circuit)
        r = super().tensor(other)
        return Circuit(len(r.dom), r.boxes, r.offsets)

    @staticmethod
    def id(n_qubits):
        if isinstance(n_qubits, Type):
            return Circuit(len(n_qubits), [], [])
        assert isinstance(n_qubits, int)
        return Circuit(n_qubits, [], [])

    def eval(self):
        return EVAL(self)

    def to_zx(self):
        return tk_to_pyzx(self.to_tk()).to_graph()

    def to_tk(self):
        c = tk.Circuit(len(self.dom))
        for g, n in zip(self.boxes, self.offsets):
            assert g.dom == g.cod and g.name in PYTKET_GATES
            c.__getattribute__(g.name)(
                *[n + i for i in range(len(g.dom))], *g.params)
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
                        gates.append(Gate('SWAP', 2))
                        offsets.append(j)
                    if q.index <= i0:
                        i0 -= 1  # we just swapped q to the right of q0
                elif q.index > i0 + i + 1:
                    for j in range(q.index - i0 + i - 1):
                        gates.append(Gate('SWAP', 2))
                        offsets.append(q.index - j - 1)
            gates.append(Gate(g.op.get_type().name,
                len(g.qubits), *g.op.get_params()))
            offsets.append(i0)
        return Circuit(c.n_qubits, gates, offsets)

    @staticmethod
    def random(n_qubits, depth):
        if n_qubits == 1:
            f, g, h = (Gate(t, 1, 2 * p) for t, p in zip(
                ['Rx', 'Rz', 'Rx'], [random(), random(), random()]))
            return Circuit(1, [f, g, h], [0, 0, 0])
        g = zx.generate.cliffordT(n_qubits, depth)
        c = zx.extract.streaming_extract(g)
        return Circuit.from_tk(pyzx_to_tk(c))

class Gate(Box, Circuit):
    def __init__(self, name, n_qubits, *params):
        self.n_qubits, self.params = n_qubits, params
        Box.__init__(self, name, PRO(n_qubits), PRO(n_qubits))

    def __repr__(self):
        return "Gate('{}', {}{})".format(self.name, len(self.dom),
            '' if not self.params else ', ' + ', '.join(map(str, self.params)))

class CircuitFunctor(MonoidalFunctor):
    def __call__(self, d):
        r = super().__call__(d)
        if isinstance(d, Diagram):
            return Circuit(len(r.dom), r.boxes, r.offsets)
        return r

#  Gates are unitaries, bras and kets are not. They are only boxes.
Ket = lambda b: Box('ket' + str(b), PRO(0), PRO(1))
Bra = lambda b: Box('bra' + str(b), PRO(1), PRO(0))
Kets = lambda b, n: fold(lambda x, y: x @ y, n * [Ket(b)])
Bras = lambda b, n: fold(lambda x, y: x @ y, n * [Bra(b)])
Id = lambda n: Circuit.id(n)
SWAP, CX = Gate('SWAP', 2), Gate('CX', 2)
H, S, T = Gate('H', 1), Gate('S', 1), Gate('T', 1)
X, Y, Z = Gate('X', 1), Gate('Y', 1), Gate('Z', 1)
Rx = lambda phase: Gate('Rx', 1, phase)
Rz = lambda phase: Gate('Rz', 1, phase)

for U in [SWAP, X, Y, Z, S >> S, CX >> CX >> CX]:
    assert np.all((U >> U).eval() == Circuit.id(U.n_qubits).eval())
for U in [H, T >> T >> T >> T]:
    np.allclose((U >> U).eval(), Circuit.id(U.n_qubits).eval())

c1_tk = tk.Circuit(3).SWAP(0, 1).Rx(1, 0.25).CX(1, 2)
c1 = Circuit.from_tk(c1_tk)
assert c1 == Circuit(3, [SWAP, Rx(0.25), CX], [0, 1, 1])
c2_tk = c1.to_tk()
c2 = Circuit.from_tk(c2_tk)
assert not c1_tk == c2_tk  # Equality of circuits in tket doesn't work!
assert c1 == c2  # This works as long as there are no interchangers!

x, y, z = Type('x'), Type('y'), Type('z')
f, g, h = Box('f', x, y + z), Box('g', z, y), Box('h', y + z, x)
d = f @ Diagram.id(z) >> Diagram.id(y) @ g @ Diagram.id(z) >> Diagram.id(y) @ h
F = CircuitFunctor({x: PRO(2), y: PRO(1), z: PRO(1)},
                   {f: SWAP, g: Rx(0.25), h: CX})
assert F(d) == c1
