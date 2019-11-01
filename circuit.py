import math
import numpy as np
import pyzx as zx
from pyzx import tensorfy
import pytket as tk
from pytket.pyzx import pyzx_to_tk, tk_to_pyzx
from random import random
from moncat import Type, Diagram, Box, MonoidalFunctor, NumpyFunctor


############ LIST OF GATES:
# Pauli X
X=np.array( [ [0,1] , [1,0] ] )

# Pauli Y
Y= np.array( [ [0,-1j] , [1j,0] ] )

# Pauli Z
Z= np.array( [ [1,0] , [0,-1] ] )

# Phases:
def Rz(theta):
	return	math.cos(theta/2)*np.identity(2)-1j*math.sin(theta/2)*Z

def Rx(theta):
	return	math.cos(theta/2)*np.identity(2)-1j*math.sin(theta/2)*X

# Control Gates
def ctrl(A):
	A=np.array(A)
	B=np.zeros((4,4))+1j*np.zeros((4,4))
	B[0,0]=1;B[1,1]=1;
	B[2,2]=A[0,0];B[2,3]=A[0,1];
	B[3,2]=A[1,0];B[3,3]=A[1,1];
	return B

#CNOT
cX= ctrl(X)

###########

GATES = { 'CX' : cX , 'Z' : Z, 'X' : X, 'Y' : Y}
PYTKET_GATES = tk.OpType.__entries.keys()

#  Turns natural numbers into types encoded in unary.
PRO = lambda n: sum(n * [Type(1)]) or Type()

class Circuit(Diagram):
    def __init__(self, n_qubits, gates, offsets):
        self.n_qubits = n_qubits
        super().__init__(PRO(n_qubits), PRO(n_qubits), gates, offsets)

    def __repr__(self):
        return "Circuit({}, {}, {})".format(
            len(self.dom), self.boxes, self.offsets)

    @staticmethod
    def id(n_qubits):
        return Circuit(n_qubits, [], [])

    def eval(self):
        class gates_to_numpy(dict):
            def __getitem__(self, g):
                if g.params:
                    if g.name == 'Rz':
                        return Rz(g.params[0])
                    if g.name == 'Rx':
                        return Rx(g.params[0])
                return GATES[g.name]
        return NumpyFunctor({PRO(1): 2}, gates_to_numpy())(self)

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
            f, g, h = (Gate(t, 1, [phase]) for t, phase in zip(
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


c1_tk = tk.Circuit(3).CX(0, 1).Rx(1, 0.25).CX(1, 2)
c1 = Circuit.from_tk(c1_tk)
assert c1 == Circuit(3,
    [Gate('CX', 2), Gate('Rx', 1, 0.25), Gate('CX', 2)], [0, 1, 1])
c2_tk = c1.to_tk()
c2 = Circuit.from_tk(c2_tk)
assert not c1_tk == c2_tk  # Equality of circuits in tket doesn't work!
assert c1 == c2  # This works as long as there are no interchangers!

# Test circuit functor
x, y, z = Type('x'), Type('y'), Type('z')
f, g, h = Box('f', x, y + z), Box('g', z, y), Box('h', y + z, x)
d = f @ Diagram.id(z) >> Diagram.id(y) @ g @ Diagram.id(z) >> Diagram.id(y) @ h
F = CircuitFunctor({x: PRO(2), y: PRO(1), z: PRO(1)},
    {f: Gate('CX', 2), g: Gate('Rx', 1, 0.25), h: Gate('CX', 2)})
assert F(d) == c1

# Test circuit evaluation
assert c1.eval().shape == tensorfy(tk_to_pyzx(c1.to_tk()).to_graph()).shape
assert c1.eval().all() == tensorfy(tk_to_pyzx(c1.to_tk()).to_graph()).all()
