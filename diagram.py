import numpy as np
from category import Arrow, Identity, Generator, Functor

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

    def then(self, other):
        assert isinstance(other, Diagram) and self.cod == other.dom
        dom, cod, nodes = self.dom, other.cod, self.nodes + other.nodes
        offsets = self.offsets + other.offsets
        return Diagram(dom, cod, nodes, offsets)

class Wire(Identity, Diagram):
    def __init__(self, x):
        xs = x if isinstance(x, list) else [x]
        self.dom, self.cod, self.nodes, self.offsets = xs, xs, [], []

class Node(Generator, Diagram):
    def __init__(self, name, dom, cod):
        self.dom, self.cod, self.nodes, self.offsets = dom, cod, [self], [0]
        self.name = name

    def __repr__(self):
        return "Node('{}', {}, {})".format(self.name, self.dom, self.cod)

class NumpyFunctor(Functor):
    def __call__(self, d):
        if not isinstance(d, Diagram):  # d must be an object
            xs = d if isinstance(d, list) else [d]
            return [self.ob[x] for x in xs]

        if isinstance(d, Node):
            return self.ar[d].reshape(self(d.dom) + self(d.cod))

        arr = 1
        for x in d.dom:
            arr = np.tensordot(arr, np.identity(self(x)[0]), 0)
        arr = np.moveaxis(arr, [2 * i for i in range(len(d.dom))],
                               [i for i in range(len(d.dom))])  # bureaucracy!

        for f, n in zip(d.nodes, d.offsets):
            source = range(len(d.dom) + n, len(d.dom) + n + len(f.dom))
            target = range(len(f.dom))
            arr = np.tensordot(arr, self(f), (source, target))

            source = range(len(arr.shape) - len(f.cod), len(arr.shape))
            destination = range(len(d.dom) + n, len(d.dom) + n +len(f.cod))
            arr = np.moveaxis(arr, source, destination)  # more bureaucracy!

        return arr


x, y, z, w = 'x', 'y', 'z', 'w'
f, g, h = Node('f', [x], [y, z]), Node('g', [z, x], [w]), Node('h', [y, w], [x])
d = f.tensor(Wire(x)).then(Wire(y).tensor(g))

Fo = NumpyFunctor({x: 1, y: 2, z: 3, w: 4}, None)
F = NumpyFunctor(Fo.ob, {a: np.zeros(Fo(a.dom) + Fo(a.cod)) for a in [f, g, h]})

assert F(d).shape == tuple(F(d.dom) + F(d.cod)) == tuple(F(x)+F(x)+F(y)+F(w))
assert F(d.then(h)) == np.tensordot(F(d), F(h), 2)
