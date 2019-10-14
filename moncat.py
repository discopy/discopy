from copy import copy

def unzip(l):
    return zip(*l)

class StringDiagram:
    def __init__(self, generators, offsets, source):
        assert(len(generators) == len(offsets))
        self.gen, self.dom = zip(generators, offsets), source

        for f, n in self.gen:
            assert(source[n : n + len(f.dom)] == f.dom)
            source = source[: n] + f.cod + source[n + len(f.dom) :]

        self.cod = source

    def __eq__(self, other):
        return self.dom == other.dom and self.gen == other.gen

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "StringDiagram" + str(tuple(unzip(self.gen) + [self.dom]))

    def tensor(self, other):
        assert(isinstance(other, StringDiagram))

        r = copy(self)
        r.dom, r.cod = self.dom + other.dom, self.cod + other.cod
        r.gen += [(f, n + len(self.cod)) for f, n in other.gen]

        return r

    def compose(self, other):
        assert(isinstance(other, StringDiagram) and self.cod == other.dom)
        generators, offsets = unzip(self.gen + other.gen)
        return StringDiagram(generators, offsets, self.dom)


class Arrow(StringDiagram):
    def __init__(self, domain, codomain, name):
        self.dom, self.cod, self.gen = domain, codomain, [(self, 0)]
        self.name = name

    def __repr__(self):
        return self.name

class Identity(StringDiagram):
    def __init__(self, objects):
        self.dom, self.cod, self.gen = objects, objects, []


x, y, z = "x", "y", "z"
f, g, h = Arrow([x], [y, z], "f"), Arrow([x, y], [z], "g"), Arrow([z, z], [x], "h")
d1 = StringDiagram([f, g, h], [1, 0, 0], [x, x])
d2 = Identity([x]).tensor(f).compose(g.tensor(Identity([z]))).compose(h)

assert(d1==d2)

class Signature:
    def __init__(self, objects, arrows):
        for f in arrows:
            assert(isinstance(f, StringDiagram)
                   and set(f.dom + f.cod).issubset(objects))
        self.ob, self.ar = objects, arrows

class MonoidalFunctor:
    def __init__(self, sigma, val):
        assert(isinstance(sigma, Signature)
               and isinstance(val, Signature))
        assert(len(sigma.ob) == len(val.ob)
               and len(sigma.ar == len(val.ar)))

        for f, v in zip(sigma.ar, val.ar):
            assert(isinstance(f, Arrow) and isinstance(v,StringDiagram))
