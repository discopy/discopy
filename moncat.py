""" Implements free monoidal categories and monoidal functors.
Naturality needs to be computed explicitly with interchangers.

>>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
>>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
>>> assert (f0 @ f1).interchange(0, 1) == Id(x) @ f1 >> f0 @ Id(w)
>>> assert (f0 @ f1).interchange(0, 1).interchange(0, 1) == f0 @ f1
>>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
>>> assert s0 @ s1 == s0 >> s1 == (s1 @ s0).interchange(0, 1)
>>> assert s1 @ s0 == s1 >> s0 == (s0 @ s1).interchange(0, 1)
>>> assert x + y != y + x
>>> assert (x + y) + z == x + y + z == x + (y + z) == sum([x, y, z], Ty())
"""

from discopy.cat import FAST, Ob, Arrow, Generator, Functor


class Ty(list):
    def __init__(self, *t):
        super().__init__(x if isinstance(x, Ob) else Ob(x) for x in t)

    def __add__(self, other):
        return Ty(*(super().__add__(other)))

    def __getitem__(self, key):  # allows to compute slices of types
        if isinstance(key, slice):
            return Ty(*super().__getitem__(key))
        return super().__getitem__(key)

    def __repr__(self):
        return "Ty({})".format(', '.join(repr(x.name) for x in self))

    def __str__(self):
        return ' + '.join(map(str, self)) or 'Ty()'

    def __hash__(self):
        return hash(repr(self))

class Diagram(Arrow):
    def __init__(self, dom, cod, boxes, offsets):
        assert isinstance(dom, Ty)
        assert isinstance(cod, Ty)
        assert isinstance(boxes, list)
        assert isinstance(offsets, list)
        assert len(boxes) == len(offsets)
        assert all(isinstance(f, Diagram) for f in boxes)
        assert all(isinstance(n, int) for n in offsets)
        self._dom, self._cod = dom, cod
        self._boxes, self._offsets = boxes, offsets
        self._data = list(zip(boxes, offsets))  # used by the Arrow class
        list.__init__(self, zip(boxes, offsets))
        if not FAST:
            scan = dom
            for f, n in zip(boxes, offsets):
                assert scan[n : n + len(f.dom)] == f.dom
                scan = scan[: n] + f.cod + scan[n + len(f.dom) :]
            assert scan == cod

    @property
    def boxes(self):
        return self._boxes

    @property
    def offsets(self):
        return self._offsets

    def __repr__(self):
        if not self:  # i.e. self is identity.
            return repr(Id(self.dom))
        return "Diagram(dom={}, cod={}, boxes={}, offsets={})".format(
            repr(self.dom), repr(self.cod),
            repr(self.boxes), repr(self.offsets))

    def __str__(self):
        if not self:  # i.e. self is identity.
            return str(self.id(self.dom))
        def line(scan, box, off):
            left = "{} @ ".format(self.id(scan[:off])) if scan[:off] else ""
            right = " @ {}".format(self.id(scan[off + len(box.dom):]))\
                                   if scan[off + len(box.dom):] else ""
            return left + str(box) + right
        box, off = self.boxes[0], self.offsets[0]
        result = line(self.dom, box, off)
        scan = self.dom[:off] + box.cod + self.dom[off + len(box.dom):]
        for box, off in zip(self.boxes[1:], self.offsets[1:]):
            result = "{} >> {}".format(result, line(scan, box, off))
            scan = scan[:off] + box.cod + scan[off + len(box.dom):]
        return result

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

    def dagger(self):
        return Diagram(self.cod, self.dom,
            [f.dagger() for f in self.boxes[::-1]], self.offsets[::-1])

    @staticmethod
    def id(x):
        return Id(x)

    def interchange(self, k0, k1):
        assert k0 + 1 == k1
        box0, box1 = self.boxes[k0], self.boxes[k1]
        off0, off1 = self.offsets[k0], self.offsets[k1]
        if off1 >= off0 + len(box0.cod):  # box0 left of box1
            off1 = off1 - len(box0.cod) + len(box0.dom)
        elif off0 >= off1 + len(box1.dom):  # box1 left of box0
            off0 = off0 - len(box1.dom) + len(box1.cod)
        else:
            raise Exception("Interchange not allowed."
                            "Boxes ({}, {}) are connected.".format(box0, box1))
        return Diagram(self.dom, self.cod,
                       self.boxes[:k0] + [box1, box0] + self.boxes[k0 + 2:],
                       self.offsets[:k0] + [off1, off0] + self.offsets[k0 + 2:])

class Id(Diagram):
    def __init__(self, x):
        super().__init__(x, x, [], [])

    def __repr__(self):
        return "Id({})".format(repr(self.dom))

    def __str__(self):
        return "Id({})".format(str(self.dom))

class Box(Generator, Diagram):
    def __init__(self, name, dom, cod, dagger=False):
        assert isinstance(dom, Ty)
        assert isinstance(cod, Ty)
        self._dom, self._cod, self._boxes, self._offsets = dom, cod, [self], [0]
        self._name, self._dagger = name, dagger
        Diagram.__init__(self, dom, cod, [self], [0])

    def dagger(self):
        return Box(self.name, self.cod, self.dom, not self._dagger)

    def __repr__(self):
        if self._dagger:
            return "Box(name={}, dom={}, cod={}).dagger()".format(
                *map(repr, [self.name, self.cod, self.dom]))
        return "Box(name={}, dom={}, cod={})".format(
            *map(repr, [self.name, self.dom, self.cod]))

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, Box):
            return repr(self) == repr(other)
        elif isinstance(other, Diagram):
            return len(other) == 1 and other.boxes[0] == self

class MonoidalFunctor(Functor):
    def __init__(self, ob, ar):
        assert all(isinstance(x, Ty) and len(x) == 1 for x in ob.keys())
        assert all(isinstance(a, Box) for a in ar.keys())
        self._objects, self._arrows = ob, ar
        self._ob = {x[0]: y for x, y in ob.items()}
        self._ar = {f.name: g for f, g in ar.items()}

    def __repr__(self):
        return "MonoidalFunctor(ob={}, ar={})".format(
                                self._objects, self._arrows)

    def __call__(self, d):
        if isinstance(d, Ty):
            return sum([self.ob[x] for x in d], Ty())
        elif isinstance(d, Box):
            return self.ar[d.name].dagger() if d._dagger else self.ar[d.name]
        scan, result = d.dom, d.id(self(d.dom))
        for f, n in d:
            result = result.then(d.id(self(scan[:n])).tensor(self(f))\
                           .tensor(d.id(self(scan[n + len(f.dom):]))))
            scan = scan[:n] + f.cod + scan[n + len(f.dom):]
        return result
