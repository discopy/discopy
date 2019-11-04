import numpy as np
from moncat import Ob, Ty, Diagram, Box, NumpyFunctor


class Adjoint(Ob):
    def __init__(self, basic, z):
        assert isinstance(z, int)
        self._basic, self._z = basic, z
        super().__init__((basic, z))

    @property
    def l(self):
        return Adjoint(self._basic, self._z - 1)

    @property
    def r(self):
        return Adjoint(self._basic, self._z + 1)

    def __repr__(self):
        return "Adjoint({}, {})".format(repr(self._basic), repr(self._z))

    def __str__(self):
        return str(self._basic) + (
            - self._z * '.l' if self._z < 0 else self._z * '.r')

    def __iter__(self):
        yield self._basic
        yield self._z

class Pregroup(Ty):
    def __init__(self, *t):
        t = [x if isinstance(x, Adjoint) else Adjoint(x, 0) for x in t]
        super().__init__(*t)

    def __add__(self, other):
        return Pregroup(*super().__add__(other))

    def __getitem__(self, key):  # allows to compute slices of types
        if isinstance(key, slice):
            return Pregroup(*super().__getitem__(key))
        return super().__getitem__(key)

    def __repr__(self):
        return "Pregroup({})".format(', '.join(
            repr(x if x._z else x._basic) for x in self))

    def __str__(self):
        return ' + '.join(map(str, self)) or "Pregroup()"

    @property
    def l(self):
        return Pregroup(*[x.l for x in self[::-1]])

    @property
    def r(self):
        return Pregroup(*[x.r for x in self[::-1]])

    @property
    def is_basic(self):
        return len(self) == 1 and not self[0]._z

class Grammar(Diagram):
    def __init__(self, dom, cod, boxes, offsets):
        assert isinstance(dom, Pregroup) and isinstance(cod, Pregroup)
        super().__init__(dom, cod, boxes, offsets)

    def then(self, other):
        r = super().then(other)
        return Grammar(Pregroup(*r.dom), Pregroup(*r.cod), r.boxes, r.offsets)

    def tensor(self, other):
        r = super().tensor(other)
        return Grammar(Pregroup(*r.dom), Pregroup(*r.cod), r.boxes, r.offsets)

    def dagger(self):
        return Grammar(self.cod, self.dom,
            [f.dagger() for f in self.boxes[::-1]], self.offsets[::-1])

    @staticmethod
    def id(t):
        return Wire(t)

    def __repr__(self):
        return "Grammar(dom={}, cod={}, boxes={}, offsets={})".format(
            *map(repr, [self.dom, self.cod, self.boxes, self.offsets]))

    def __str__(self):
        return repr(self)

class Wire(Grammar):
    def __init__(self, t):
        if isinstance(t, Word):
            t = t.dom
        assert isinstance(t, Pregroup)
        super().__init__(t, t, [], [])

    def __repr__(self):
        return "Wire({})".format(repr(self.dom))

    def __str__(self):
        return "Wire({})".format(str(self.dom))

class Cup(Grammar, Box):
    def __init__(self, x, dagger=False):
        if isinstance(x, Pregroup):
            assert len(x) == 1
            x = x[0]
        elif not isinstance(x, Adjoint):
            x = Adjoint(x, 0)
        dom, cod = Pregroup(x, x.l) if dagger else Pregroup(x, x.r), Pregroup()
        Box.__init__(self, 'cup_{}'.format(x), dom, cod, dagger)

    def dagger(self):
        return Cap(self.dom[0], not self._dagger)

    def __repr__(self):
        return "Cup({}{})".format(repr(
            self.dom[0] if self.dom[0]._z else self.dom[0]._basic),
            ", dagger=True" if self._dagger else "")

    def __str__(self):
        return "Cup({}{})".format(str(self.dom[0]),
                                ", dagger=True" if self._dagger else "")

class Cap(Grammar, Box):
    def __init__(self, x, dagger=False):
        if isinstance(x, Pregroup):
            assert len(x) == 1
            x = x[0]
        elif not isinstance(x, Adjoint):
            x = Adjoint(x, 0)
        dom, cod = Pregroup(), Pregroup(x, x.r) if dagger else Pregroup(x, x.l)
        Box.__init__(self, 'cap_{}'.format(x), dom, cod, dagger)

    def dagger(self):
        return Cup(self.cod[0], not self._dagger)

    def __repr__(self):
        return "Cap({}{})".format(repr(
            self.cod[0] if self.cod[0]._z else self.cod[0]._basic),
            ", dagger=True" if self._dagger else "")

    def __str__(self):
        return "Cap({}{})".format(str(self.cod[0]),
                                ", dagger=True" if self._dagger else "")

class Word(Grammar, Box):
    def __init__(self, w, t, dagger=False):
        assert isinstance(w, str)
        assert isinstance(t, Pregroup)
        self._word, self._type = w, t
        dom, cod = (t, Pregroup(w)) if dagger else (Pregroup(w), t)
        Box.__init__(self, (w, t), dom, cod, dagger)

    def dagger(self):
        return Word(self._word, self._type, not self._dagger)

    @property
    def word(self):
        return self._word

    @property
    def type(self):
        return self._type

    def __repr__(self):
        return "Word({}, {}){}".format(repr(self.word), repr(self.type),
                                       ".dagger()" if self._dagger else "")

    def __str__(self):
        return str(self.word)

class Parse(Grammar):
    def __init__(self, words, cups):
        self._words, self._cups = words, cups
        self._type = sum((w.type for w in words), Pregroup())
        dom = sum((w.dom for w in words), Pregroup())
        boxes = words[::-1]  # words are backwards to make offsets easier
        offsets = [len(words) - i - 1 for i in range(len(words))] + cups
        cod = self._type
        for i in cups:
            assert cod[i].r == cod[i + 1]
            boxes.append(Cup(cod[i]))
            cod = cod[:i] + cod[i + 2:]
        super().__init__(dom, cod, boxes, offsets)

    def __str__(self):
        return "{} >> {}".format(" @ ".join(self._words), Grammar(self._type,
            self.cod, self.boxes[len(self._words):], self._cups))

class Model(NumpyFunctor):
    def __init__(self, ob, ar):
        assert all(isinstance(x, Pregroup) and x.is_basic for x in ob.keys())
        assert all(isinstance(a, Word) for a in ar.keys())
        self._types, self._vocab = ob, ar
        # rigid functors are defined by their image on basic types
        ob = {x[0]._basic: ob[x] for x in self._types.keys()}
        # we assume the images for word boxes are all states
        ob.update({w.dom[0]._basic: 1 for w in self._vocab.keys()})
        self._ob, self._ar = ob, {f.name: n for f, n in ar.items()}

    def __repr__(self):
        return "Model(ob={}, ar={})".format(self._types, self._vocab)

    def __call__(self, d):
        if isinstance(d, Adjoint):
            return int(self.ob[d._basic])
        if isinstance(d, Pregroup):
            return [self(x) for x in d]
        if isinstance(d, Cup):
            return np.identity(self(d.dom[0]))
        if isinstance(d, Cap):
            return np.identity(self(d.cod[0]))
        return super().__call__(d)


if __name__ == '__main__':
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r + s + n.l)
    grammar = Cup(n) @ Wire(s) @ Cup(n.l)
    sentence = grammar << Alice @ loves @ Bob
    assert sentence == Parse([Alice, loves, Bob], [0, 1]).interchange(0, 1)\
                                                         .interchange(1, 2)\
                                                         .interchange(0, 1)
    F = Model({s: 1, n: 2},
              {Alice: [1, 0],
               loves: [0, 1, 1, 0],
               Bob: [0, 1]})
    assert F(sentence) == True

    snake_l = Cap(n) @ Wire(n) >> Wire(n) @ Cup(n.l)
    snake_r = Wire(n) @ Cap(n.r) >> Cup(n) @ Wire(n)
    assert (F(snake_l) == F(Wire(n))).all()
    assert (F(Wire(n)) == F(snake_r)).all()
