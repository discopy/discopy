"""
Diagrams implement categories, PlanarDiagrams strict monoidal categories.
"""

class Diagram:
    def __init__(self, dom, cod, nodes):
        self.dom, self.cod, self.nodes = dom, cod, nodes
        u = dom
        for f in nodes:
            assert u == f.dom
            u = f.cod
        assert u == cod

    def __eq__(self, other):
        assert isinstance(other, Diagram)
        return self.dom == other.dom and self.cod == other.cod\
                                     and self.nodes == other.nodes

    def __repr__(self):
        return "Diagram('{}', '{}', {})".format(self.dom, self.cod, self.nodes)

    def compose(self, other):
        assert isinstance(other, Diagram) and self.cod == other.dom
        return Diagram(self.dom, other.cod, self.nodes + other.nodes)


class Identity(Diagram):
    def __init__(self, x):
        self.dom, self.cod, self.nodes = x, x, []


class Node(Diagram):
    def __init__(self, dom, cod, name):
        self.dom, self.cod = dom, cod
        self.nodes, self.name = [self], name

    def __eq__(self, other):
        return self.dom == other.dom and self.cod == other.cod\
                                     and self.name == other.name

    def __repr__(self):
        return "Node('{}', '{}', '{}')".format(self.dom, self.cod, self.name)


class PlanarDiagram(Diagram):
    def __init__(self, dom, cod, nodes, offsets):
        self.dom, self.cod = dom, cod
        self.nodes, self.offsets = nodes, offsets
        assert len(nodes) == len(offsets)
        u = dom
        for f, n in zip(nodes, offsets):
            assert(u[n : n + len(f.dom)] == f.dom)
            u = u[: n] + f.cod + u[n + len(f.dom) :]
        assert u == cod

    def __eq__(self, other):
        assert isinstance(other, PlanarDiagram)
        return super().__eq__(other) and self.offsets == other.offsets

    def __repr__(self):
        return "PlanarDiagram({}, {}, {}, {})".format(
            self.dom, self.cod, self.nodes, self.offsets)

    def tensor(self, other):
        assert isinstance(other, PlanarDiagram)
        dom, cod = self.dom + other.dom, self.cod + other.cod
        nodes = self.nodes + other.nodes
        offsets = self.offsets + [n + len(self.cod) for n in other.offsets]
        return PlanarDiagram(dom, cod, nodes, offsets)

    def compose(self, other):
        assert isinstance(other, PlanarDiagram) and self.cod == other.dom
        dom, cod, nodes = self.dom, other.cod, self.nodes + other.nodes
        offsets = self.offsets + other.offsets
        return PlanarDiagram(dom, cod, nodes, offsets)


class PlanarIdentity(Identity, PlanarDiagram):
    def __init__(self, x):
        self.dom, self.cod, self.nodes, self.offsets = x, x, [], []


class PlanarNode(Node, PlanarDiagram):
    def __init__(self, dom, cod, name):
        self.dom, self.cod = dom, cod
        self.nodes, self.offsets = [self], [0]
        self.name = name

    def __repr__(self):
        return "PlanarNode({}, {}, \'{}\')".format(
            self.dom, self.cod, self.name)


assert Node('x', 'y', 'f').compose(Node('y', 'z', 'g')) ==\
    Node('x', 'y', 'f').compose(Identity('y')).compose(Node('y', 'z', 'g')) ==\
    Diagram('x', 'z', [Node('x', 'y', 'f'), Node('y', 'z', 'g')])

assert repr(Node('x', 'y', 'f').compose(Node('y', 'z', 'g'))) ==\
    repr(Diagram('x', 'z', [Node('x', 'y', 'f'), Node('y', 'z', 'g')])) ==\
    "Diagram('x', 'z', [Node('x', 'y', \'f\'), Node('y', 'z', \'g\')])"

x, y, z = 'x', 'y', 'z'
idx, idz = PlanarIdentity([x]), PlanarIdentity([z])
f, g, h = (PlanarNode(dom, cod, name) for (dom, cod, name) in
           [([x], [y, z], "f"), ([x, y], [z], "g"), ([z, z], [x], "h")])
d1 = PlanarDiagram([x, x], [x], [f, g, h], [1, 0, 0])
d2 = idx.tensor(f).compose(g.tensor(idz)).compose(h)

assert d1 == d2
assert repr(d1) == "PlanarDiagram(['x', 'x'], ['x'], "\
    "[PlanarNode(['x'], ['y', 'z'], 'f'), PlanarNode(['x', 'y'], ['z'], 'g'), "\
    "PlanarNode(['z', 'z'], ['x'], 'h')], [1, 0, 0])"
