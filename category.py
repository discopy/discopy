class Arrow:
    def __init__(self, dom, cod, nodes):
        self.dom, self.cod, self.nodes = dom, cod, nodes
        u = dom
        for f in nodes:
            assert u == f.dom
            u = f.cod
        assert u == cod

    def __eq__(self, other):
        assert isinstance(other, Arrow)
        return all(x.name == y.name for x, y in zip(self.nodes, other.nodes))

    def __repr__(self):
        return "Arrow('{}', '{}', {})".format(self.dom, self.cod, self.nodes)

    def then(self, other):
        assert isinstance(other, Arrow) and self.cod == other.dom
        return Arrow(self.dom, other.cod, self.nodes + other.nodes)

class Identity(Arrow):
    def __init__(self, x):
        self.dom, self.cod, self.nodes = x, x, []

class Generator(Arrow):
    def __init__(self, name, dom, cod):
        self.dom, self.cod = dom, cod
        self.nodes, self.name = [self], name

    def __repr__(self):
        return "Generator('{}', '{}', '{}')".format(
            self.name, self.dom, self.cod)

    def __eq__(self, other):
        return self.dom == other.dom and self.cod == other.cod\
                                     and self.name == other.name

    def __hash__(self):
        return hash(str(self.name))

class Functor:
    def __init__(self, ob, ar):
        self.ob, self.ar = ob, ar

    def __call__(self, a):
        if not isinstance(a, Arrow):  # a must be an object
            return self.ob[a]

        if isinstance(a, Generator):
            return self.ar[a]

        r = lambda x: x
        compose = lambda f, g: (lambda x: g(f(x)))
        for g in a.nodes:
            r = compose(r, self(g))
        return r

x, y, z = 'x', 'y', 'z'
f, g = Generator('f', x, y), Generator('g', y, z)
F = Functor(None, {f: lambda x: x**2, g: lambda x: x + 1})

assert f.then(g) == f.then(Identity(y)).then(g) == Arrow(x, z, [f, g])
assert F(f.then(g))(2) == F(g)(F(f)(2)) == 5
