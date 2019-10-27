import pyzx as zx
import pytket as tk
from pytket.pyzx import pyzx_to_tk
from random import random
from diagram import Diagram, Box, Wire, MonoidalFunctor


PRO = lambda n: n * [1]
GATES = tk.OpType.__entries.keys()


class Circuit(Diagram):
    def __init__(self, n_qubits, gates, offsets):
        super.n_qubits = n_qubits
        super().__init__(PRO(n_qubits), PRO(n_qubits), gates, offsets)

    def __repr__(self):
        return "Circuit({}, {}, {})".format(
            len(self.dom), self.nodes, self.offsets)

    def to_tk(self):
        c = tk.Circuit(len(self.dom))
        for f, n in zip(self.nodes, self.offsets):
            assert f.dom == f.cod and f.name in GATES
            c.__getattribute__(f.name)(
                *[n + i for i in range(len(f.dom))], *f.data)
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
                len(g.qubits), g.op.get_params()))
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
    def __init__(self, name, n_qubits, data=[]):
        self.n_qubits, self.data = n_qubits, data
        super().__init__(name, PRO(n_qubits), PRO(n_qubits))

    def __repr__(self):
        return "Gate('{}', {}{})".format(self.name, len(self.dom),
            '' if not self.data else ", " + repr(self.data))

class IdCircuit(Wire, Circuit):
    def __init__(self, n_qubits):
        n_qubits = n_qubits if isinstance(n_qubits, int) else len(n_qubits)
        super().__init__(PRO(n_qubits))

class PytketFunctor(MonoidalFunctor):
    def __call__(self, d):
        if not isinstance(d, Diagram):  # d must be an object
            xs = d if isinstance(d, list) else [d]
            return PRO(sum(self.ob[x] for x in xs))
        r = super().__call__(d)
        return Circuit(len(r.dom), r.nodes, r.offsets)


c1 = tk.Circuit(3).CX(0, 2).H(1).SWAP(0, 1).Rx(0, 0.25)
d1 = Circuit.from_tk(c1)
c2 = d1.to_tk()
d2 = Circuit.from_tk(c2)
assert not c1 == c2  # Equality in tket doesn't work!
assert d1 == d2  # This works as long as there are no interchangers!


x, y, z, w = 'x', 'y', 'z', 'w'
f, g, h = Box('f', [x], [y, z]), Box('g', [z, z], [w]), Box('h', [y, w], [x, z])
d = f.tensor(Wire(z)).then(Wire(y).tensor(g)).then(h)
F = PytketFunctor({x: 2, y: 1, z: 1, w: 2},
    {f: Gate('CX', 2), g: Gate('CZ', 2), h: Gate('CCX', 3)})
assert F(d) == Circuit(3,
    [Gate('CX', 2), Gate('CZ', 2), Gate('CCX', 3)], [0, 1, 0])
