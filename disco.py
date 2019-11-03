import numpy as np
from moncat import Ob, Ty, Diagram, Box, NumpyFunctor


class Adjoint(Ob):
    def __init__(self, b, z):
        assert isinstance(z, int)
        self._b, self._z = b, z
        super().__init__((b, z))

    @property
    def b(self):
        return self._b

    @property
    def z(self):
        return self._z

    @property
    def l(self):
        return Adjoint(self.b, self.z - 1)

    @property
    def r(self):
        return Adjoint(self.b, self.z + 1)

    def __repr__(self):
        return "Adjoint({}, {})".format(repr(self.b), repr(self.z))

    def __str__(self):
        return str(self.b) + (- self.z * '.l' if self.z < 0 else self.z * '.r')

    def __iter__(self):
        yield self.b
        yield self.z

class Pregroup(Ty):
    def __init__(self, *t):
        if not isinstance(t, list):  # t is a basic type
            super().__init__(Adjoint(t, 0))
        else:
            assert all(isinstance(x, Adjoint) for x in t)
            super().__init__(*t)

    def __add__(self, other):
        return Pregroup(*super().__add__(other))

    def __getitem__(self, key):  # allows to compute slices of types
        if isinstance(key, slice):
            return Pregroup(*super().__getitem__(key))
        return super().__getitem__(key)

    def __repr__(self):
        if not self:
            return 'Pregroup()'
        return ' + '.join(map(repr, self))

    @property
    def l(self):
        return Pregroup(*[t.l for t in self[::-1]])

    @property
    def r(self):
        return Pregroup(*[t.r for t in self[::-1]])

    @property
    def is_basic(self):
        return len(self) == 1 and self[0].z == 0

class PDiagram(Diagram):
    def __init__(self, dom, cod, nodes, offsets):
        assert all(isinstance(x, Pregroup) for x in [dom, cod])
        super().__init__(dom, cod, nodes, offsets)

    def tensor(self, other):
        r = super().tensor(other)
        r._dom, r._cod = Pregroup(r.dom), Pregroup(r.cod)
        return r

class Wire(PDiagram):
    def __init__(self, x):
        assert isinstance(x, Pregroup)
        super().__init__(x, x, [], [])

class Word(PDiagram, Box):
    def __init__(self, w, t):
        assert isinstance(w, str)
        assert isinstance(t, Pregroup)
        self._word, self._type = w, t
        Box.__init__(self, (w, t), Pregroup(w), t)

    @property
    def word(self):
        return self._word

    @property
    def type(self):
        return self._type

    def __repr__(self):
        return "Word({}, {})".format(self.word, self.type)

class Cup(PDiagram, Box):
    def __init__(self, x):
        assert isinstance(x, Pregroup) and len(x) == 1
        Box.__init__(self, 'cup_{}'.format(x), x + x.r, Pregroup())

class Cap(PDiagram, Box):
    def __init__(self, x):
        assert isinstance(x, Pregroup) and len(x) == 1
        Box.__init__(self, 'cap_{}'.format(x), Pregroup(), x + x.l)

class Parse(Diagram):
    def __init__(self, words, cups):
        dom = sum(w.dom for w in words)
        nodes = words[::-1]  # words are backwards to make offsets easier
        offsets = [len(words) - i - 1 for i in range(len(words))] + cups
        cod = sum(w.cod for w in words)
        for i in cups:
            assert cod[i].r == cod[i + 1]
            nodes.append(Cup(cod[i: i + 1]))
            cod = cod[:i] + cod[i + 2:]
        super().__init__(dom, cod, nodes, offsets)

class Model(NumpyFunctor):
    def __init__(self, ob, ar):
        assert all(isinstance(x, Pregroup) and x.is_basic for x in ob.keys())
        assert all(isinstance(y, int) for y in ob.values())
        assert all(isinstance(a, Word) for a in ar.keys())
        assert all(isinstance(b, np.ndarray) for b in ar.values())
        # rigid functors are defined by their image on basic types
        ob = {x[0].b: ob[x] for x in ob.keys()}
        # we assume the images for word boxes are all states
        ob.update({w.dom[0].b: 1 for w in ar.keys()})
        self._ob, self._ar = ob, ar

    def __call__(self, d):
        if isinstance(d, Adjoint):
            return self.ob[d.b]
        if isinstance(d, Pregroup):
            return [self(x) for x in d]
        if isinstance(d, Cup):
            return np.identity(self(d.dom[0]))
        if isinstance(d, Cap):
            return np.identity(self(d.cod[0]))
        return super().__call__(d)


s, n = Pregroup('s'), Pregroup('n')

alice, bob = Word('Alice', n), Word('Bob', n)
loves = Word('loves', n.r + s + n.l)
sentence = Parse([alice, loves, bob], [0, 1])

F = Model({s: 1, n: 2},
          {alice : [1, 0],
           bob : [0, 1],
           loves : [0, 1, 1, 0]})

assert F(sentence) == True

snake_l = Cap(n) @ Wire(n) >> Wire(n) @ Cup(n.l)
snake_r = Wire(n) @ Cap(n.r) >> Cup(n) @ Wire(n)
assert (F(snake_l) == F(Wire(n))).all()
assert (F(Wire(n)) == F(snake_r)).all()
