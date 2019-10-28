import numpy as np
import pyzx as zx
from category import Object, Arrow, Identity, Generator, Functor

class Type(list):
    def __init__(self, t=[]):
        if not isinstance(t, list):  # t is a generating object
            super().__init__([t if isinstance(t, Object) else Object(t)])
        super().__init__([x if isinstance(x, Object) else Object(x) for x in t])

    def __add__(self, other):
        return Type(list(self) + list(other))

    def __repr__(self):
        if not self:
            return 'Type()'
        return ' + '.join(repr(x) for x in self)

    def __hash__(self):
        return hash(repr(self))

class Diagram(Arrow):
    def __init__(self, dom, cod, boxes, offsets):
        assert isinstance(dom, Type)
        assert isinstance(cod, Type)
        assert isinstance(boxes, list)
        assert isinstance(offsets, list)
        assert len(boxes) == len(offsets)
        assert all(isinstance(f, Diagram) for f in boxes)
        assert all(isinstance(n, int) for n in offsets)
        self._dom, self._cod = dom, cod
        self._boxes, self._offsets = boxes, offsets
        scan = dom
        for f, n in zip(boxes, offsets):
            assert Type(scan[n : n + len(f.dom)]) == f.dom
            scan = Type(scan[: n]) + f.cod + Type(scan[n + len(f.dom) :])
        assert scan == cod
        list.__init__(self, zip(boxes, offsets))

    @property
    def boxes(self):
        return self._boxes

    @property
    def offsets(self):
        return self._offsets

    def __repr__(self):
        return "Diagram({}, {}, {}, {})".format(
            self.dom, self.cod, self.boxes, self.offsets)

    def tensor(self, other):
        assert isinstance(other, Diagram)
        dom, cod = self.dom + other.dom, self.cod + other.cod
        boxes = self.boxes + other.boxes
        offsets = self.offsets + [n + len(self.cod) for n in other.offsets]
        return Diagram(dom, cod, boxes, offsets)

    def then(self, other):
        assert isinstance(other, Diagram) and self.cod == other.dom
        dom, cod, boxes = self.dom, other.cod, self.boxes + other.boxes
        offsets = self.offsets + other.offsets
        return Diagram(dom, cod, boxes, offsets)

class Wire(Identity, Diagram):
    def __init__(self, x):
        assert isinstance(x, Type)
        Diagram.__init__(self, x, x, [], [])

class Box(Generator, Diagram):
    def __init__(self, name, dom, cod):
        self._dom, self._cod, self._boxes, self._offsets = dom, cod, [self], [0]
        self._name = name
        Diagram.__init__(self, dom, cod, [self], [0])

    def __repr__(self):
        return "Box('{}', {}, {})".format(self.name, self.dom, self.cod)

class MonoidalFunctor(Functor):
    def __init__(self, ob, ar):
        assert all(isinstance(x, Type) and len(x) == 1 for x in ob.keys())
        assert all(isinstance(y, Type) for y in ob.values())
        assert all(isinstance(a, Box) for a in ar.keys())
        assert all(isinstance(b, Diagram) for b in ar.values())
        self._ob, self._ar = {x.pop(): ob[x] for x in ob.keys()}, ar

    def __call__(self, d):
        if isinstance(d, Type):
            return Type(self.ob[x] for x in d)

        elif isinstance(d, Box):
            return self.ar[d]

        u = d.dom
        r = Wire(self(u))

        for f, n in zip(d.boxes, d.offsets):
            r = r.then(Wire(self(u[:n])).tensor(self(f))\
                 .tensor(Wire(self(u[n + len(f.dom):]))))
            u = u[:n] + f.cod + u[n + len(f.dom):]

        return r


x, y, z, w = Type('x'), Type('y'), Type('z'), Type('w')
f, g, h = Box('f', x, x + y), Box('g', y + z, w), Box('h', x + w, x)
d = f.tensor(Wire(z)).then(Wire(x).tensor(g))

# IdF = MonoidalFunctor({o: o for o in [x, y, z, w]},
#                       {a: a for a in [f, g, h]})
#
# assert IdF(d.then(h)) == IdF(d).then(IdF(h)) == d.then(h)
