import numpy as np


class Arrow:
    def __init__(self, dom, cod, nodes):
        self.dom, self.cod, self.nodes = dom, cod, nodes
        u = dom
        for f in nodes:
            assert u == f.dom
            u = f.cod
        assert u == cod

    def __eq__(self, other):
        assert isinstance(other, Arrow)
        return self.dom == other.dom and self.cod == other.cod\
                                     and self.nodes == other.nodes

    def __repr__(self):
        return "Arrow('{}', '{}', {})".format(self.dom, self.cod, self.nodes)

    def compose(self, other):
        assert isinstance(other, Arrow) and self.cod == other.dom
        return Arrow(self.dom, other.cod, self.nodes + other.nodes)

class Identity(Arrow):
    def __init__(self, x):
        self.dom, self.cod, self.nodes = x, x, []

class Generator(Arrow):
    def __init__(self, name, dom, cod):
        self.dom, self.cod = dom, cod
        self.nodes, self.name = [self], name

    def __eq__(self, other):
        return self.dom == other.dom and self.cod == other.cod\
                                     and self.name == other.name

    def __repr__(self):
        return "Generator('{}', '{}', '{}')".format(
            self.name, self.dom, self.cod)

x, y, z = 'x', 'y', 'z'
f, g = Generator('f', x, y), Generator('g', y, z)

assert f.compose(g) == f.compose(Identity(y)).compose(g) == Arrow(x, z, [f, g])


class Diagram(Arrow):
    def __init__(self, dom, cod, nodes, offsets):
        self.dom, self.cod = dom, cod
        self.nodes, self.offsets = nodes, offsets
        assert len(nodes) == len(offsets)
        u = dom
        for f, n in zip(nodes, offsets):
            assert u[n : n + len(f.dom)] == f.dom
            u = u[: n] + f.cod + u[n + len(f.dom) :]
        assert u == cod

    def __eq__(self, other):
        assert isinstance(other, Diagram)
        return super().__eq__(other) and self.offsets == other.offsets

    def __repr__(self):
        return "Diagram({}, {}, {}, {})".format(
            self.dom, self.cod, self.nodes, self.offsets)

    def tensor(self, other):
        assert isinstance(other, Diagram)
        dom, cod = self.dom + other.dom, self.cod + other.cod
        nodes = self.nodes + other.nodes
        offsets = self.offsets + [n + len(self.cod) for n in other.offsets]
        return Diagram(dom, cod, nodes, offsets)

    def compose(self, other):
        assert isinstance(other, Diagram) and self.cod == other.dom
        dom, cod, nodes = self.dom, other.cod, self.nodes + other.nodes
        offsets = self.offsets + other.offsets
        return Diagram(dom, cod, nodes, offsets)

class Wire(Identity, Diagram):
    def __init__(self, x):
        self.dom, self.cod, self.nodes, self.offsets = x, x, [], []

class Node(Generator, Diagram):
    def __init__(self, name, dom, cod):
        self.dom, self.cod, self.nodes, self.offsets = dom, cod, [self], [0]
        self.name = name

    def __repr__(self):
        return "Node('{}', {}, {})".format(self.name, self.dom, self.cod)

f, g, h = Node('f', [x], [y, z]), Node('g', [x, y], [z]), Node('h', [z, z], [x])
d1 = Diagram([x, x], [x], [f, g, h], [1, 0, 0])
d2 = Wire([x]).tensor(f).compose(g.tensor(Wire([z]))).compose(h)
assert d1 == d2


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
