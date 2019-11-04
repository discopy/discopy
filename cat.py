FAST = False
from functools import reduce as fold


class Ob(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __eq__(self, other):
        if not isinstance(other, Ob):
            return False
        return self.name == other.name

    def __repr__(self):
        return "Ob({})".format(repr(self.name))

    def __str__(self):
        return str(self.name)

    def __hash__(self):
        return hash(repr(self))

class Arrow(list):
    def __init__(self, dom, cod, data):
        if not FAST:
            assert isinstance(dom, Ob)
            assert isinstance(cod, Ob)
            assert isinstance(data, list)
            assert all(isinstance(f, Arrow) for f in data)
            u = dom
            for f in data:
                assert f.data  # i.e. f is not the identity arrow
                assert u == f.dom
                u = f.cod
            assert u == cod
        self._dom, self._cod, self._data = dom, cod, data
        super().__init__(data)

    @property
    def dom(self):
        return self._dom

    @property
    def cod(self):
        return self._cod

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return "Arrow({}, {}, {})".format(
            repr(self.dom), repr(self.cod), repr(self.data))

    def __str__(self):
        return " >> ".join(map(str, self))

    def __eq__(self, other):
        if not isinstance(other, Arrow):
            return False
        return self.dom == other.dom and self.cod == other.cod\
            and all(x == y for x, y in zip(self.data, other.data))

    def then(self, other):
        assert isinstance(other, Arrow)
        assert self.cod == other.dom
        return Arrow(self.dom, other.cod, self.data + other.data)

    def __rshift__(self, other):
        return self.then(other)

    def __lshift__(self, other):
        return other.then(self)

    def dagger(self):
        return Arrow(self.cod, self.dom, [f.dagger() for f in self.data[::-1]])

    @staticmethod
    def id(x):
        assert isinstance(x, Ob)
        return Arrow(x, x, [])

class Generator(Arrow):
    def __init__(self, name, dom, cod, dagger=False):
        assert isinstance(dom, Ob)
        assert isinstance(cod, Ob)
        self._dagger = dagger
        self._name, self._dom, self._cod, self._data = name, dom, cod, [self]
        super().__init__(dom, cod, [self])

    @property
    def name(self):
        return self._name

    def dagger(self):
        return Generator(self.name, self.cod, self.dom, not self._dagger)

    def __repr__(self):
        if self._dagger:
            return "Generator(name={}, dom={}, cod={}).dagger()".format(
                *map(repr, [self.name, self.cod, self.dom]))
        return "Generator(name={}, dom={}, cod={})".format(
            *map(repr, [self.name, self.dom, self.cod]))

    def __str__(self):
        return str(self.name) + (".dagger()" if self._dagger else '')

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if not isinstance(other, Arrow):
            return False
        if isinstance(other, Generator):
            return repr(self) == repr(other)
        return len(other) == 1 and other.data[0] == self

class Function(Generator):
    def __init__(self, f, dom, cod):
        assert isinstance(dom, type)
        assert isinstance(cod, type)
        self._name, self._dom, self._cod = f, dom, cod

    def __call__(self, x):
        assert isinstance(x, self.dom)
        y = self.name(x)
        assert isinstance(y, self.cod)
        return y

    def then(self, other):
        return Function(lambda x: other(self(x)), self.dom, other.cod)

class Functor:
    def __init__(self, ob, ar):
        assert all(isinstance(x, Ob) for x in ob.keys())
        assert all(isinstance(y, type) for y in ob.values())
        assert all(isinstance(a, Generator) for a in ar.keys())
        assert all(isinstance(b, Function) for b in ar.values())
        self._ob, self._ar = ob, ar

    @property
    def ob(self):
        return self._ob

    @property
    def ar(self):
        return self._ar

    def __eq__(self, other):
        return self.ob == other.ob and self.ar == other.ar

    def __repr__(self):
        return "Functor(ob={}, ar={})".format(repr(self.ob), repr(self.ar))

    def __call__(self, f):
        if isinstance(f, Ob):
            return self.ob[f]
        if isinstance(f, Generator):
            return self.ar[f]
        assert isinstance(f, Arrow)
        unit = Function(lambda x: x, self(f.dom), self(f.dom))
        return fold(lambda g, h: g.then(self(h)), f, unit)
