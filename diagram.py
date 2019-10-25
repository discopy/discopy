import numpy as np
import pyzx as zx
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

class Box(Generator, Diagram):
    def __init__(self, name, dom, cod):
        self.dom, self.cod, self.nodes, self.offsets = dom, cod, [self], [0]
        self.name = name

    def __repr__(self):
        return "Box('{}', {}, {})".format(self.name, self.dom, self.cod)

class MonoidalFunctor(Functor):
    def __call__(self, d):
        if not isinstance(d, Diagram):  # d must be an object
            xs = d if isinstance(d, list) else [d]
            return [self.ob[x] for x in xs]

        elif isinstance(d, Box):
            return self.ar[d]

        u = d.dom
        r = Wire(self(u))

        for f, n in zip(d.nodes, d.offsets):
            r = r.then(Wire(self(u[:n])).tensor(self(f))\
                 .tensor(Wire(self(u[n + len(f.dom):]))))
            u = u[:n] + f.cod + u[n + len(f.dom):]

        return r


x, y, z, w = 'x', 'y', 'z', 'w'
f, g, h = Box('f', [x], [x, y]), Box('g', [y, z], [w]), Box('h', [x, w], [x])
d = f.tensor(Wire(z)).then(Wire(x).tensor(g))

idF = MonoidalFunctor({o: o for o in [x, y, z, w]},
                      {a: a for a in [f, g, h]})

assert idF(d.then(h)) == idF(d).then(idF(h)) == d.then(h)
