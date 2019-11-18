""" Implements free rigid categories and distributional compositional models.

>>> s, n = Pregroup('s'), Pregroup('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r + s + n.l)
>>> grammar = Cup(n) @ Wire(s) @ Cup(n.l)
>>> sentence = grammar << Alice @ loves @ Bob
>>> ob = {s: 1, n: 2}
>>> ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
>>> F = Model(ob, ar)
>>> assert F(sentence) == True
"""

import numpy as np
from discopy.moncat import Ob, Ty, Diagram, Box
from discopy.matrix import Dim, Matrix, Id, MatrixFunctor


class Adjoint(Ob):
    """
    Implements basic types and their iterated adjoints, also known as simple types.

    >>> a = Adjoint('a', 0)
    >>> a
    Adjoint('a', 0)
    >>> a.l
    Adjoint('a', -1)
    >>> a.r
    Adjoint('a', 1)
    >>> a.r.r
    Adjoint('a', 2)
    >>> assert a.l.r == a.r.l == a and a != a.l.l != a.r.r
    """
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
    """ Implements pregroup types as lists of adjoints.

    >>> s, n = Pregroup('s'), Pregroup('n')
    >>> s
    Pregroup('s')
    >>> print(s)
    s
    >>> s.l
    Pregroup(Adjoint('s', -1))
    >>> s @ n.l
    Pregroup('s', Adjoint('n', -1))
    >>> (s @ n).l
    Pregroup(Adjoint('n', -1), Adjoint('s', -1))
    >>> assert n.l.r == n == n.r.l
    >>> assert (s @ n).l == n.l @ s.l and (s @ n).r == n.r @ s.r
    """
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
    """ Implements diagrams in free rigid categories, by checking that the domain
    and codomain are pregroup types.

    >>> n, b = Pregroup('n'), Pregroup('b')
    >>> Alice = Word('b', n)
    >>> snake_l = Cap(n) @ Wire(n) >> Wire(n) @ Cup(n.l)
    >>> snake_r = Wire(n) @ Cap(n.r) >> Cup(n) @ Wire(n)

    We take take transposes of any morphism.

    >>> assert Alice.transpose_r() == Cap(b.r) @ Wire(n.r) >> Wire(b.r) @ Alice @ Wire(n.r) >> Wire(b.r) @ Cup(n)
    >>> assert Wire(n.l).transpose_r() == snake_l
    >>> assert Wire(n.r).transpose_l() == snake_r
    """
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

    def transpose_r(self):
        a = self.dom
        b = self.cod
        return Grammar.cap(a.r) @ Wire(b.r) >> Wire(a.r) @ self @ Wire(b.r) >> \
               Wire(a.r) @ Grammar.cup(b)

    def transpose_l(self):
        a = self.dom
        b = self.cod
        return Wire(b.l) @ Grammar.cap(a) >> Wire(b.l) @ self @ Wire(a.l) >> \
               Grammar.cup(b.l) @ Wire(a.l)

    @staticmethod
    def id(t):
        return Wire(t)

    @staticmethod
    def cup(t):
        """ Construct cups for pregroup types.

        >>> n, s = Pregroup('n'), Pregroup('s')
        >>> Grammar.cup(n @ s @ n).dom
        Pregroup('n', 's', 'n', Adjoint('n', 1), Adjoint('s', 1), Adjoint('n', 1))
        >>> Grammar.cup(n @ s @ n).boxes
        [Cup('n'), Cup('s'), Cup('n')]
        >>> Grammar.cup(n @ s @ n).offsets
        [2, 1, 0]
        >>> Grammar.cup(n @ s @ n).cod
        Pregroup()
        >>> assert Grammar.cup(n) == Cup('n')
        """
        assert isinstance(t, Pregroup)
        dom = t @ t.r
        cod = Pregroup()
        boxes = [Cup(b) for b in t[::-1]]
        offsets = list(range(len(t)))[::-1]
        return Grammar(dom, cod, boxes, offsets)

    @staticmethod
    def cap(t):
        """ Construct caps for pregroup types.

        >>> n, s = Pregroup('n'), Pregroup('s')
        >>> Grammar.cap(n @ s).dom
        Pregroup()
        >>> Grammar.cap(n @ s).boxes
        [Cap('n'), Cap('s')]
        >>> Grammar.cap(n @ s).offsets
        [0, 1]
        >>> Grammar.cap(n @ s).cod
        Pregroup('n', 's', Adjoint('s', -1), Adjoint('n', -1))
        >>> assert Grammar.cap(n) == Cap('n')
        """
        assert isinstance(t, Pregroup)
        dom = Pregroup()
        cod = t @ t.l
        boxes = [Cap(b) for b in t]
        offsets = list(range(len(t)))
        return Grammar(dom, cod, boxes, offsets)

    def __repr__(self):
        return "Grammar(dom={}, cod={}, boxes={}, offsets={})".format(
            *map(repr, [self.dom, self.cod, self.boxes, self.offsets]))

    def __str__(self):
        return repr(self)

class Wire(Grammar):
    """ Define an identity arrow in a free rigid category

    >>> assert Wire(Pregroup('x')) == Grammar(Pregroup('x'), Pregroup('x'), [], [])
    """
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
    """ Defines cups for simple types.

    >>> Cup('n').dom
    Pregroup('n', Adjoint('n', 1))
    >>> Cup('n').cod
    Pregroup()
    """
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
    """ Defines caps for simple types.

    >>> Cap('n').dom
    Pregroup()
    >>> Cap('n').cod
    Pregroup('n', Adjoint('n', -1))
    """
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
    """ Encodes words with their pregroup type as diagrams in free rigid categories

    >>> s, n = Pregroup('s'), Pregroup('n')
    >>> Alice = Word('Alice', n)
    >>> Alice.dom
    Pregroup('Alice')
    >>> Alice.cod
    Pregroup('n')
    >>> loves = Word('loves', n.r + s + n.l)
    >>> loves.dom
    Pregroup('loves')
    >>> loves.cod
    Pregroup(Adjoint('n', 1), 's', Adjoint('n', -1))
    """
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
    """ Produces the diagram in a free rigid category corresponding to a pregroup parsing.

    >>> s, n = Pregroup('s'), Pregroup('n')
    >>> Alice, Bob, jokes = Word('Alice', n), Word('Bob', n), Word('jokes', n)
    >>> loves, tells = Word('loves', n.r + s + n.l), Word('tells', n.r + s + n.l)
    >>> who = Word('who', n.r + n.l.r + s.l + n)

    A parse is given by a list of words and a list of offsets for the cups.

    >>> parse = Parse(words = [Alice, loves, Bob], cups = [0, 1])
    >>> parse1 = Parse([Alice, loves, Bob, who, tells, jokes], [0, 2, 1, 2, 1, 1])

    A sentence u is grammatical if there is a parsing with domain u and codomain the sentence type s.

    >>> parse.dom
    Pregroup('Alice', 'loves', 'Bob')
    >>> parse.cod
    Pregroup('s')
    >>> parse._type
    Pregroup('n', Adjoint('n', 1), 's', Adjoint('n', -1), 'n')
    """
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
            self.cod, self.boxes[len(self._words):], ))

class Model(MatrixFunctor):
    """ Implements functors from pregroup grammars to matrices

    >>> n = Pregroup('n')
    >>> F = Model({n: 2}, {})
    >>> snake_l = Cap(n) @ Wire(n) >> Wire(n) @ Cup(n.l)
    >>> snake_r = Wire(n) @ Cap(n.r) >> Cup(n) @ Wire(n)
    >>> assert (F(snake_l) == F(Wire(n))).all()
    >>> assert (F(Wire(n)) == F(snake_r)).all()
    """
    def __init__(self, ob, ar):
        assert all(isinstance(x, Pregroup) and x.is_basic for x in ob.keys())
        assert all(isinstance(a, Word) for a in ar.keys())
        self._types, self._vocab = ob, ar
        #  Rigid functors are defined by their image on basic types.
        ob = {x[0]._basic: ob[x] for x in self._types.keys()}
        #  We assume the images for word boxes are all states.
        ob.update({w.dom[0]._basic: 1 for w in self._vocab.keys()})
        self._ob, self._ar = ob, {f: array for f, array in ar.items()}

    def __repr__(self):
        return "Model(ob={}, ar={})".format(self._types, self._vocab)

    def __call__(self, d):
        if isinstance(d, Pregroup):
            return Dim(*(self.ob[x._basic] for x in d))
        if isinstance(d, Cup):
            return Matrix(self(d.dom), Dim(), Id(self(d.dom[:1])).array)
        if isinstance(d, Cap):
            return Matrix(Dim(), self(d.cod), Id(self(d.cod[:1])).array)
        return super().__call__(d)
