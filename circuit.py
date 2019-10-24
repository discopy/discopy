import pytket
from diagram import PRODiagram, PRONode


GATES = {
    'CX': pytket.Circuit.CX,
    'SWAP': pytket.Circuit.SWAP,
    'H': pytket.Circuit.H,
    'T': pytket.Circuit.T,
    'S': pytket.Circuit.S,
    'Rx': pytket.Circuit.Rx,
    'Ry': pytket.Circuit.Ry,
    'Rz': pytket.Circuit.Rz
}

class Gate(PRONode):
    def __init__(self, name, n_qubits, params):
        self.params, self.n_qubits = params, n_qubits
        super().__init__(name, n_qubits, n_qubits)

    @staticmethod
    def from_pytket(g):
        return Gate(g.op.get_type().name, len(g.qubits), g.op.get_params())

class Circuit(PRODiagram):
    def __init__(self, n_qubits, nodes, offsets):
        super().__init__(n_qubits, n_qubits, nodes, offsets)

    def to_pytket(self):
        assert self.dom == self.cod
        c = pytket.Circuit(self.dom)
        for f, n in zip(self.nodes, self.offsets):
            assert f.dom == f.cod and f.name in GATES.keys()
            GATES[f.name](c, *[n + i for i in range(f.dom)], *f.params)
        return c

    @staticmethod
    def from_pytket(c):
        nodes, offsets = [], []
        for g in c.get_commands():
            assert g.op.get_type().name in GATES.keys()
            nodes.append(Gate.from_pytket(g))
            offsets.append(g.qubits[0].index)
            if len(g.qubits) > 1:
                for i, q in enumerate(g.qubits[1:]):
                    # Checking that gates apply to adjacent qubits.
                    assert q.index == g.qubits[0].index + i + 1
        return Circuit(c.n_qubits, nodes, offsets)


c1 = pytket.Circuit(3).CX(1, 2).H(1).SWAP(0, 1).Rx(0, 0.25)
c2 = Circuit.from_pytket(c1).to_pytket()
assert not c1 == c2  # Equality in pytket doesn't work!

d1 = Circuit.from_pytket(c1)
d2 = Circuit.from_pytket(d1.to_pytket())
assert d1 == d2  # This works as long as there are no interchangers!
