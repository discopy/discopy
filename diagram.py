import numpy as np
import pyzx as zx
from category import Object, Arrow, Identity, Generator, Functor

class Type(list):
    def __init__(self, t=[]):
        if not isinstance(t, list):  # t is a generating object
            assert isinstance(t, str)
            super().__init__([Object(t)])
        else:
            for x in t:
                assert isinstance(x, Object)
            super().__init__(t)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Type(super().__getitem__(key))
        return super().__getitem__(key)

    def __add__(self, other):
        return Type(list(self) + list(other))

    def __radd__(self, other):  # allows to compute sums of types
        if not other:
            return self
        return other + self

    def __repr__(self):
        if not self:
            return 'Type()'
        return ' + '.join(x.name for x in self)

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
            assert scan[n : n + len(f.dom)] == f.dom
            scan = scan[: n] + f.cod + scan[n + len(f.dom) :]
        assert scan == cod
        self._data = list(zip(boxes, offsets))  # used by the category module
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
        dom, cod = self.dom, other.cod
        boxes = self.boxes + other.boxes
        offsets = self.offsets + other.offsets
        return Diagram(dom, cod, boxes, offsets)

class Wire(Identity, Diagram):
    def __init__(self, x):
        assert isinstance(x, Type)
        Diagram.__init__(self, x, x, [], [])

class Box(Generator, Diagram):
    def __init__(self, name, dom, cod):
        assert isinstance(name, str)
        assert isinstance(dom, Type)
        assert isinstance(cod, Type)
        self._dom, self._cod, self._boxes, self._offsets = dom, cod, [self], [0]
        self._name = name
        Diagram.__init__(self, dom, cod, [self], [0])

    def __repr__(self):
        return "Box('{}', {}, {})".format(self.name, self.dom, self.cod)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, Box):
            return repr(self) == repr(other)
        elif isinstance(other, Diagram):
            return len(other) == 1 and other.boxes[0] == self
        return False

class MonoidalFunctor(Functor):
    def __init__(self, ob, ar):
        assert all(isinstance(x, Type) and len(x) == 1 for x in ob.keys())
        assert all(isinstance(y, Type) for y in ob.values())
        assert all(isinstance(a, Box) for a in ar.keys())
        assert all(isinstance(b, Diagram) for b in ar.values())
        self._ob, self._ar = {x[0]: y for x, y in ob.items()}, ar

    def __call__(self, d):
        if isinstance(d, Type):
            return sum(self.ob[x] for x in d) or Type()
        elif isinstance(d, Box):
            return self.ar[d]
        scan = d.dom
        result = Wire(self(scan))
        for f, n in zip(d.boxes, d.offsets):
            result = result.then(Wire(self(scan[:n])).tensor(self(f))\
                           .tensor(Wire(self(scan[n + len(f.dom):]))))
            scan = scan[:n] + f.cod + scan[n + len(f.dom):]
        return result


x, y, z, w = Type('x'), Type('y'), Type('z'), Type('w')
assert x + y != y + x
assert (x + y) + z == x + y + z == x + (y + z) == sum([x, y, z])
f, g, h = Box('f', x, x + y), Box('g', y + z, w), Box('h', x + w, x)
d = f.tensor(Wire(z)).then(Wire(x).tensor(g))

IdF = MonoidalFunctor({o: o for o in [x, y, z, w]},
                      {a: a for a in [f, g, h]})

assert IdF(d.then(h)) == IdF(d).then(IdF(h)) == d.then(h)
