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
        return self.dom == other.dom and self.cod == other.cod\
                                     and self.nodes == other.nodes

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

    def __hash__(self):
        return hash(str(self.name))

class Functor:
    def __init__(self, data_ob, data_ar):
        self.data_ob, self.data_ar = data_ob, data_ar

    def ob(self, x):
        return self.data_ob[x]

    def ar(self, x):
        return self.data_ar[x]

    def apply(self, f):
        assert isinstance(f, Arrow)
        r = lambda x: x
        then = lambda f, g: (lambda x: g(f(x)))
        for g in f.nodes:
            r = then(r, self.ar(g))
        return r

x, y, z = 'x', 'y', 'z'
f, g = Generator('f', x, y), Generator('g', y, z)
F = Functor({x: None, y: None, z: None},
            {f: lambda x: x**2, g: lambda x: x + 1})

assert f.then(g) == f.then(Identity(y)).then(g) == Arrow(x, z, [f, g])
assert F.apply(f.then(g))(2) == 5


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
        self.dom, self.cod, self.nodes, self.offsets = x, x, [], []

class Node(Generator, Diagram):
    def __init__(self, name, dom, cod):
        self.dom, self.cod, self.nodes, self.offsets = dom, cod, [self], [0]
        self.name = name

    def __repr__(self):
        return "Node('{}', {}, {})".format(self.name, self.dom, self.cod)


f, g, h = Node('f', [x], [y, z]), Node('g', [x, y], [z]), Node('h', [z, z], [x])
d1 = Diagram([x, x], [x], [f, g, h], [1, 0, 0])
d2 = Wire([x]).tensor(f).then(g.tensor(Wire([z]))).then(h)
assert d1 == d2


#
# data = {'f': lambda *xs: [xs[0], xs[0] + 1],
#         'g': lambda *xs: [xs[0] + xs[1]],
#         'h': lambda *xs: [xs[0] * xs[1]]}
#
# class MonoidalFunctor(Functor):
#     def apply(self, d):
#         assert isinstance(d, Diagram)
#         then = lambda f, g: (lambda *xs: g(f(xs)))
#         next = lambda f, n: (lambda *xs:
#             xs[:n] + self.data[f.name](xs[n: n + len(f.dom)]) + xs[n + len(f.dom):])
#         r = lambda *xs: xs
#         for f, n in zip(d.nodes, d.offsets):
#             r = then(r, next(f, n))
#         return r
