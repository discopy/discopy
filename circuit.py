import pytket
from diagram import Diagram, Node


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


class PRODiagram(Diagram):
    def __init__(self, dom, cod, nodes, offsets):
        self.dom, self.cod = dom, cod
        self.nodes, self.offsets = nodes, offsets
        assert len(nodes) == len(offsets)
        u = dom
        for f, n in zip(nodes, offsets):
            assert n + f.dom <= u
            u = u - f.dom + f.cod
        assert u == cod

    def tensor(self, other):
        assert isinstance(other, PRODiagram)
        dom, cod = self.dom + other.dom, self.cod + other.cod
        nodes = self.nodes + other.nodes
        offsets = self.offsets + [n + self.cod for n in other.offsets]
        return PRODiagram(dom, cod, nodes, offsets)

    def compose(self, other):
        assert isinstance(other, PRODiagram) and self.cod == other.dom
        dom, cod, nodes = self.dom, other.cod, self.nodes + other.nodes
        offsets = self.offsets + other.offsets
        return PRODiagram(dom, cod, nodes, offsets)

    def __repr__(self):
        return "PRODiagram({}, {}, {}, {})".format(
            self.dom, self.cod, self.nodes, self.offsets)

class PROWire(Wire, PRODiagram):
    pass

class PRONode(Node, PRODiagram):
    def __repr__(self):
        return "PRONode('{}', {}, {})".format(
            self.name, self.dom, self.cod)


f, g, h = PRONode('f', 1, 2), PRONode('g', 2, 1), PRONode('h', 2, 1)
d1 = PRODiagram(2, 1, [f, g, h], [1, 0, 0])
d2 = PROWire(1).tensor(f).compose(g.tensor(PROWire(1))).compose(h)
assert d1 == d2

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
