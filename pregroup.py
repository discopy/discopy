import numpy as np
from functools import reduce as fold
from diagram import Diagram, Node, Functor

class NumpyFunctor(Functor):
    def apply(self, d):
        assert isinstance(d, Diagram)
        def identity(dims):
            return np.identity(int(np.prod(dims)))
        dim = [self.ob(x) for x in d.dom]
        arr = identity(dim)
        for f, n in zip(d.nodes, d.offsets):
            fdom = int(np.prod([self.ob(x) for x in f.dom])) #dimension of domain
            fcod = int(np.prod([self.ob(x) for x in f.cod])) #dimension of codomain
            fmatrix = np.reshape(self.ar(f), (fdom, fcod)) #make f into a square matrix
            a = np.kron( np.kron( identity(dim[0:n]), fmatrix), identity(dim[n+len(f.dom):]))
            arr = np.dot(arr, a)
            dim = dim[0:n] + [self.ob(x) for x in f.cod] + dim[n+len(f.dom):]
        return arr

x, y, z = 'x', 'y', 'z'
f, g, h = Node('f', [x], [y, z]), Node('g', [x, y], [z]), Node('h', [z, z], [x])
d = Diagram([x, x], [x], [f, g, h], [1, 0, 0])
F = NumpyFunctor({x: 1, y:2, z:3},
                 {f: np.array([[[1, 1, 0], [0, 0, 1]]]),
                  g: np.array([[[1, 1, 1], [1, 1, 1]]]),
                  h: np.array([[[1],[1],[1]],[[0],[0],[0]],[[2],[2],[2]]])
                 })
F.apply(d)

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
        super().__init__('cup', [x, y], [])

class Parsing(Diagram):
    def __init__(self, words, cups):
        dom = [w.dom[0] for w in words]
        nodes = words[::-1]  # words are backwards to make offsets easier
        offsets = [len(words) - i - 1 for i in range(len(words))] + cups
        cod = [x for w in words for x in w.cod]
        for i in cups:
            nodes.append(Cup(cod[i], cod[i + 1]))
            cod = cod[:i] + cod[i + 2:]
        super().__init__(dom, cod, nodes, offsets)

class DisCo(NumpyFunctor):
    def apply(self, p):
        assert isinstance(p, Parsing)
        return super().apply(p)

    def ob(self, x):
        if isinstance(x, str):  # x is a word
            return 1
        else:  # x is a simple type (b, z)
            return self.data_ob[(x[0], 0)]  # we forget adjoints

s, n = ('s', 0), ('n', 0)
l = lambda b, z: (b, z - 1)
r = lambda b, z: (b, z + 1)
alice, bob = Word('Alice', [n]), Word('Bob', [n])
loves = Word('loves', [l(*n), s, r(*n)])
p = Parsing([alice, loves, bob], [0, 1])

F = DisCo({s: 1, n: 2},
          {alice : np.array([0, 1]),
           bob : np.array([1, 0]),
           loves : np.array([[0, 1], [1, 0]])})
