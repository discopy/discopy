from functools import reduce as fold


class Object(object):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __eq__(self, other):
        if not isinstance(other, Object):
            return False
        return self.name == other.name

    def __repr__(self):
        return "Object({})".format(repr(self.name))

    def __hash__(self):
        return hash(repr(self))

class Arrow(list):
    def __init__(self, dom, cod, data):
        assert isinstance(dom, Object)
        assert isinstance(cod, Object)
        assert isinstance(data, list)
        assert all(isinstance(g, Arrow) for g in data)
        u = dom
        for g in data:
            assert g.data  # i.e. f is not the identity arrow
            assert u == g.dom
            u = g.cod
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
        return "Arrow({}, {}, {})".format(self.dom, self.cod, self.data)

    def __eq__(self, other):
        if not isinstance(other, Arrow):
            return False
        return all(x == y for x, y in zip(self.data, other.data))

    def then(self, other):
        assert isinstance(other, Arrow)
        assert self.cod == other.dom
        return Arrow(self.dom, other.cod, self.data + other.data)

class Identity(Arrow):
    def __init__(self, x):
        assert isinstance(x, Object)
        super().__init__(x, x, [])

class Generator(Arrow):
    def __init__(self, name, dom, cod):
        assert isinstance(dom, Object)
        assert isinstance(cod, Object)
        self._name, self._dom, self._cod, self._data = name, dom, cod, [self]
        super().__init__(dom, cod, [self])

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return "Generator('{}', {}, {})".format(
            self.name, self.dom, self.cod)

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
        assert isinstance(f, type(lambda x: x))
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
        assert all(isinstance(x, Object) for x in ob.keys())
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

    def __call__(self, f):
        if isinstance(f, Object):
            return self.ob[f]
        if isinstance(f, Generator):
            return self.ar[f]
        assert isinstance(f, Arrow)
        unit = Function(lambda x: x, self(f.dom), self(f.dom))
        return fold(lambda g, h: g.then(self(h)), f, unit)


a, b, c = Object('a'), Object('a'), 'c'
assert a == b and b != c

x, y, z = Object('x'), Object('y'), Object('z')
f, g, h = Generator('f', x, y), Generator('g', y, z), Generator('h', z, x)
assert Identity(x).then(f) == f == f.then(Identity(y))
assert (f.then(g)).dom == f.dom and (f.then(g)).cod == g.cod
assert f.then(g).then(h) == f.then(g.then(h)) == Arrow(x, x, [f, g, h])

a = f.then(g).then(h)
F = Functor({x: int, y:tuple, z:int}, {
    f: Function(lambda x: (x, x), int, tuple),
    g: Function(lambda x: x[0] + x[1], tuple, int),
    h: Function(lambda x: x // 2, int, int)})
assert F(Identity(x))(42) == Function(lambda x: x, int, int)(42) == 42
assert F(f.then(g))(42) == F(g)(F(f)(42))
assert F(a)(42) == F(h)(F(g)(F(f)(42))) == F(Identity(x))(42) == 42
