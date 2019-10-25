import pyzx as zx
import pytket as tk
from pytket.pyzx import pyzx_to_tk
from random import random
from diagram import Diagram, Box, Wire, MonoidalFunctor


PRO = lambda n: n * [1]
GATES = tk.OpType.__entries.keys()


class Circuit(Diagram):
    def __init__(self, n_qubits, gates, offsets):
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
        nodes, offsets = [], []
        for g in c.get_commands():
            assert g.op.get_type().name in GATES
            nodes.append(Gate(
                g.op.get_type().name, len(g.qubits), g.op.get_params()))
            offsets.append(g.qubits[0].index)
            if len(g.qubits) > 1:
                for i, q in enumerate(g.qubits[1:]):
                    # Checking that gates apply to adjacent qubits.
                    assert q.index == g.qubits[0].index + i + 1
        return Circuit(c.n_qubits, nodes, offsets)

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


c1 = tk.Circuit(3).CX(1, 2).H(1).SWAP(0, 1).Rx(0, 0.25)
c2 = Circuit.from_tk(c1).to_tk()
assert not c1 == c2  # Equality in tket doesn't work!

d1 = Circuit.from_tk(c1)
d2 = Circuit.from_tk(d1.to_tk())
assert d1 == d2  # This works as long as there are no interchangers!


x, y, z, w = 'x', 'y', 'z', 'w'
f, g, h = Box('f', [x], [y, z]), Box('g', [z, z], [w]), Box('h', [y, w], [x, z])
d = f.tensor(Wire(z)).then(Wire(y).tensor(g)).then(h)
F = PytketFunctor({x: 2, y: 1, z: 1, w: 2},
    {f: Gate('CX', 2), g: Gate('CZ', 2), h: Gate('CCX', 3)})
assert F(d) == Circuit(3,
    [Gate('CX', 2), Gate('CZ', 2), Gate('CCX', 3)], [0, 1, 0])
