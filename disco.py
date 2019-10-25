import numpy as np
from diagram import Diagram, Box, Wire, MonoidalFunctor


class Type(list):
    def __init__(self, t):
        if isinstance(t, str):  # t is a basic type
            super().__init__([(t, 0)])
        else:
            assert all(isinstance(z, int) for b, z in t)
            super().__init__(t)

    def __add__(self, other):
        return Type(list(self) + list(other))

    def __repr__(self):
        f = lambda z: - z * '.l' if z < 0 else z * '.r'
        return ' + '.join(b + f(z) for b, z in self)

    def __hash__(self):
        return hash(repr(self))

    @property
    def l(self):
        return Type([(b, z - 1) for b, z in self[::-1]])

    @property
    def r(self):
        return Type([(b, z + 1) for b, z in self[::-1]])

class Word(Box):
    def __init__(self, w, t):
        assert isinstance(w, str)
        assert isinstance(t, Type)
        super().__init__((w, t), [(w, t)], t)

    def __repr__(self):
        return "Word" + str(self.name)

class Cup(Box):
    def __init__(self, x, y):
        assert x[0] == y[0] and y[1] - x[1] == 1  # x and y are adjoints
        super().__init__('cup_{}'.format(x[0]), Type([x, y]), [])

class Parse(Diagram):
    def __init__(self, words, cups):
        dom = [w.name for w in words]
        nodes = words[::-1]  # words are backwards to make offsets easier
        offsets = [len(words) - i - 1 for i in range(len(words))] + cups
        cod = Type([(b, z) for w in words for b, z in w.cod])
        for i in cups:
            nodes.append(Cup(cod[i], cod[i + 1]))
            cod = Type(cod[:i] + cod[i + 2:])
        super().__init__(dom, cod, nodes, offsets)

class NumpyFunctor(MonoidalFunctor):
    def __call__(self, d):
        if not isinstance(d, Diagram):  # d must be an object
            xs = d if isinstance(d, list) else [d]
            return [self.ob[x] for x in xs]

        if isinstance(d, Box):
            return self.ar[d].reshape(self(d.dom) + self(d.cod))

        arr = 1
        for x in d.dom:
            arr = np.tensordot(arr, np.identity(self.ob[x]), 0)
        arr = np.moveaxis(arr, [2 * i for i in range(len(d.dom))],
                               [i for i in range(len(d.dom))])  # bureaucracy!

        for f, n in zip(d.nodes, d.offsets):
            source = range(len(d.dom) + n, len(d.dom) + n + len(f.dom))
            target = range(len(f.dom))
            arr = np.tensordot(arr, self(f), (source, target))

            source = range(len(arr.shape) - len(f.cod), len(arr.shape))
            destination = range(len(d.dom) + n, len(d.dom) + n +len(f.cod))
            arr = np.moveaxis(arr, source, destination)  # more bureaucracy!

        return arr

class Model(NumpyFunctor):
    def __init__(self, ob, ar):
        ob.update({w.name: 1 for w in ar.keys()})
        self.ob, self.ar = ob, ar

    def __call__(self, d):
        if not isinstance(d, Diagram):  # d is an object
            if isinstance(d, Type):  # we forget adjoints
                return super().__call__([Type([(b, 0)]) for b, z in d])
            return super().__call__(d)

        if isinstance(d, Cup):
            return np.identity(self(d.dom)[0])

        return super().__call__(d)


x, y, z, w = 'x', 'y', 'z', 'w'
f, g, h = Box('f', [x], [x, y]), Box('g', [y, z], [w]), Box('h', [x, w], [x])
d = f.tensor(Wire(z)).then(Wire(x).tensor(g))

F0 = NumpyFunctor({x: 1, y: 2, z: 3, w: 4}, None)
F = NumpyFunctor(F0.ob, {a: np.zeros(F0(a.dom) + F0(a.cod)) for a in [f, g, h]})

assert F(d).shape == tuple(F(d.dom) + F(d.cod))
assert np.all(F(d.then(h)) == np.tensordot(F(d), F(h), 2))


s, n = Type('s'), Type('n')
alice, bob = Word('Alice', n), Word('Bob', n)
loves = Word('loves', n.r + s + n.l)
sentence = Parse([alice, loves, bob], [0, 1])

F = Model({s: 1, n: 2},
          {alice : np.array([1, 0]),
           bob : np.array([0, 1]),
           loves : np.array([0, 1, 1, 0])})

assert F(sentence)
