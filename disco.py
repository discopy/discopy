from numpy import array, identity
from diagram import Diagram, Node, Wire, NumpyFunctor


class Type(list):
    def __init__(self, t):
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

class Word(Node):
    def __init__(self, w, t):
        assert isinstance(w, str)
        assert isinstance(t, Type)
        super().__init__((w, t), [(w, t)], t)

    def __repr__(self):
        return str(self.name)

class Cup(Node):
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

class Model(NumpyFunctor):
    def __init__(self, ob, ar):
        ob.update({w.name: 1 for w in ar.keys()})
        self.ob, self.ar = ob, ar

    def __call__(self, d):
        if not isinstance(d, Diagram):  # d is an object
            if isinstance(d, Type):  # we forget adjoints
                return super().__call__([Type([(b, 0)]) for b, z in d])
            return super().__call__(d if isinstance(d, list) else [d])

        if isinstance(d, Cup):
            return identity(self(d.dom)[0])

        return super().__call__(d)


s, n = Type([('s', 0)]), Type([('n', 0)])
alice, bob = Word('Alice', n), Word('Bob', n)
loves = Word('loves', n.r + s + n.l)
sentence = Parse([alice, loves, bob], [0, 1])

F = Model({s: 1, n: 2},
          {alice : array([1, 0]),
           bob : array([0, 1]),
           loves : array([0, 1, 1, 0])})

assert F(sentence)
