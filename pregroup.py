from diagram import Diagram, Node


class Word(Node):
    def __init__(self, w, t):
        assert isinstance(w, str)
        for b, z in t:
            assert isinstance(z, int)
        super().__init__(w, [w], t)

class Cup(Node):
    def __init__(self, x, y):
        assert x[0] == y[0] and x[1] - y[1] == 1  # x and y are adjoints
        super().__init__('cup', [x, y], [])

class Parsing(Diagram):
    def __init__(self, words, cups):
        dom = [w.name for w in words]
        nodes = words[::-1]  # words are backwards to make offsets easier
        offsets = [len(words) - i - 1 for i in range(len(words))] + cups
        cod = [x for w in words for x in w.cod]
        for i in cups:
            nodes.append(Cup(cod[i], cod[i + 1]))
            cod = cod[:i] + cod[i + 2:]
        super().__init__(dom, cod, nodes, offsets)

s, n = [('s', 0)], [('n', 0)]
l = lambda t: [(b, z - 1) for b, z in t[::-1]]
r = lambda t: [(b, z + 1) for b, z in t[::-1]]
alice, bob = Word('Alice', n), Word('Bob', n)
loves = Word('loves', l(n) + s + r(n))
p = Parsing([alice, loves, bob], [0, 1])
