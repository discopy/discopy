from numpy import array, identity
from diagram import Diagram, Node, Wire, NumpyFunctor


s, n = ('s', 0), ('n', 0)
l = lambda b, z: (b, z - 1)
r = lambda b, z: (b, z + 1)

class Word(Node):
    def __init__(self, w, t):
        assert isinstance(w, str)
        for b, z in t:
            assert isinstance(z, int)
        super().__init__((w, t), [w], t)

    def __repr__(self):
        return self.name[0]

class Cup(Node):
    def __init__(self, x, y):
        assert x[0] == y[0] and x[1] - y[1] == 1  # x and y are adjoints
        super().__init__('cup_{}'.format(x[0]), [x, y], [])

class Parse(Diagram):
    def __init__(self, words, cups):
        dom = [w.dom[0] for w in words]
        nodes = words[::-1]  # words are backwards to make offsets easier
        offsets = [len(words) - i - 1 for i in range(len(words))] + cups
        cod = [x for w in words for x in w.cod]
        for i in cups:
            nodes.append(Cup(cod[i], cod[i + 1]))
            cod = cod[:i] + cod[i + 2:]
        super().__init__(dom, cod, nodes, offsets)

class Model(NumpyFunctor):
    def __init__(self, vocab, ob, ar):
        self.vocab, self.ob, self.ar = vocab, ob, ar

    def __call__(self, d):
        if not isinstance(d, Diagram):
            xs = d if isinstance(d, list) else [d]
            if xs in [[w] for w in self.vocab]:
                return [1]  # words are states
            return super().__call__([(b, 0) for b, z in xs])  # forget adjoints

        if isinstance(d, Cup):
            return identity(self(d.dom)[0])

        return super().__call__(d)


alice, bob = Word('Alice', [n]), Word('Bob', [n])
loves = Word('loves', [l(*n), s, r(*n)])
sentence = Parse([alice, loves, bob], [0, 1])

F = Model(['Alice', 'loves', 'Bob'], {s: 1, n: 2},
          {alice : array([0, 1]),
           bob : array([1, 0]),
           loves : array([[0, 1], [1, 0]])})

assert F(sentence)
