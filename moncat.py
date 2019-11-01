import numpy as np
import pyzx as zx
from cat import Object, Arrow, Generator, Functor


class Type(list):
    def __init__(self, t=[]):
        if not isinstance(t, list):  # t is a generating object
            super().__init__([Object(t)])
        else:
            assert all(isinstance(x, Object) for x in t)
            super().__init__(t)

    def __add__(self, other):
        return Type(list(self) + list(other))

    def __radd__(self, other):  # allows to compute sums of types
        return self if not other else other + self

    def __getitem__(self, key):  # allows to compute slices of types
        if isinstance(key, slice):
            return Type(super().__getitem__(key))
        return super().__getitem__(key)

    def __repr__(self):
        return 'Type()' if not self else ' + '.join(str(x.name) for x in self)

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

    def __matmul__(self, other):
        return self.tensor(other)

    def then(self, other):
        assert isinstance(other, Diagram) and self.cod == other.dom
        dom, cod = self.dom, other.cod
        boxes = self.boxes + other.boxes
        offsets = self.offsets + other.offsets
        return Diagram(dom, cod, boxes, offsets)

    def __rshift__(self, other):
        return self.then(other)

    def __lshift__(self, other):
        return other.then(self)

    @staticmethod
    def id(x):
        assert isinstance(x, Type)
        return Diagram(x, x, [], [])

def Id(x):
    return Diagram.id(x)

class Box(Generator, Diagram):
    def __init__(self, name, dom, cod):
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
        scan, result = d.dom, d.id(self(d.dom))
        for f, n in d:
            result = result.then(d.id(self(scan[:n])).tensor(self(f))\
                           .tensor(d.id(self(scan[n + len(f.dom):]))))
            scan = scan[:n] + f.cod + scan[n + len(f.dom):]
        return result

class NumpyFunctor(MonoidalFunctor):
    def __init__(self, ob, ar):
        assert all(isinstance(x, Type) and len(x) == 1 for x in ob.keys())
        assert all(isinstance(y, int) for y in ob.values())
        assert all(isinstance(a, Box) for a in ar.keys())
        assert all(isinstance(b, np.ndarray) for b in ar.values())
        self._ob, self._ar = {x[0]: y for x, y in ob.items()}, ar

    def __call__(self, d):
        if isinstance(d, Object):
            return self.ob[d]
        elif isinstance(d, Type):
            return tuple(self.ob[x] for x in d)
        elif isinstance(d, Box):
            return self.ar[d].reshape(self(d.dom) + self(d.cod))
        arr = 1
        for x in d.dom:
            arr = np.tensordot(arr, np.identity(self(x)), 0)
        arr = np.moveaxis(arr, [2 * i for i in range(len(d.dom))],
                               [i for i in range(len(d.dom))])  # bureaucracy!
        for f, n in d:
            source = range(len(d.dom) + n, len(d.dom) + n + len(f.dom))
            target = range(len(f.dom))
            arr = np.tensordot(arr, self(f), (source, target))
            source = range(len(arr.shape) - len(f.cod), len(arr.shape))
            destination = range(len(d.dom) + n, len(d.dom) + n +len(f.cod))
            arr = np.moveaxis(arr, source, destination)  # more bureaucracy!
        return arr


x, y, z, w = Type('x'), Type('y'), Type('z'), Type('w')
assert x + y != y + x
assert (x + y) + z == x + y + z == x + (y + z) == sum([x, y, z])
f, g, h = Box('f', x, x + y), Box('g', y + z, w), Box('h', x + w, x)
d = Id(x) @ g << f @ Id(z)

IdF = MonoidalFunctor({o: o for o in [x, y, z, w]},
                      {a: a for a in [f, g, h]})

assert IdF(d >> h) == IdF(d) >> IdF(h) == d >> h

F0 = NumpyFunctor({x: 1, y: 2, z: 3, w: 4}, dict())
F = NumpyFunctor({x: 1, y: 2, z: 3, w: 4},
                 {a: np.zeros(F0(a.dom) + F0(a.cod)) for a in [f, g, h]})

assert F(d).shape == tuple(F(d.dom) + F(d.cod))
assert F(d >> h).shape == np.tensordot(F(d), F(h), 2).shape
assert np.all(F(d >> h) == np.tensordot(F(d), F(h), 2))
