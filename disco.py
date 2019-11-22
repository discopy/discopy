""" Implements free rigid categories and distributional compositional models.

>>> s, n = Pregroup('s'), Pregroup('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r @ s @ n.l)

# >>> grammar = Cup(n) @ Wire(s) @ Cup(n.l)
# >>> sentence = grammar << Alice @ loves @ Bob
# >>> ob = {s: 1, n: 2}
# >>> ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
# >>> F = Model(ob, ar)
# >>> assert F(sentence) == True
"""

import numpy as np
from discopy import moncat
from discopy.moncat import Ob, Ty, Diagram, Box
from discopy.matrix import Dim, Matrix, Id, MatrixFunctor
from discopy.circuit import CircuitFunctor


class Adjoint(Ob):
    """
    Implements simple types: basic types and their iterated adjoints.

    >>> a = Adjoint('a', 0)
    >>> assert a.l.r == a.r.l == a and a != a.l.l != a.r.r
    """
    def __init__(self, basic, z):
        """
        >>> a = Adjoint('a', 0)
        >>> a.name
        ('a', 0)
        """
        if not isinstance(z, int):
            raise ValueError("Expected int, got {} instead".format(repr(z)))
        self._basic, self._z = basic, z
        super().__init__((basic, z))

    @property
    def l(self):
        """
        >>> Adjoint('a', 0).l
        Adjoint('a', -1)
        """
        return Adjoint(self._basic, self._z - 1)

    @property
    def r(self):
        """
        >>> Adjoint('a', 0).r
        Adjoint('a', 1)
        """
        return Adjoint(self._basic, self._z + 1)

    def __repr__(self):
        """
        >>> Adjoint('a', 42)
        Adjoint('a', 42)
        """
        return "Adjoint({}, {})".format(repr(self._basic), repr(self._z))

    def __str__(self):
        """
        >>> a = Adjoint('a', 0)
        >>> print(a)
        a
        >>> print(a.r)
        a.r
        >>> print(a.l)
        a.l
        """
        return str(self._basic) + (
            - self._z * '.l' if self._z < 0 else self._z * '.r')

class Pregroup(Ty):
    """ Implements pregroup types as lists of adjoints.

    >>> s, n = Pregroup('s'), Pregroup('n')
    >>> assert n.l.r == n == n.r.l
    >>> assert (s @ n).l == n.l @ s.l and (s @ n).r == n.r @ s.r
    """
    def __init__(self, *t):
        """
        >>> Pregroup('s', 'n')
        Pregroup('s', 'n')
        """
        t = [x if isinstance(x, Adjoint) else Adjoint(x, 0) for x in t]
        super().__init__(*t)

    def __add__(self, other):
        """
        >>> s, n = Pregroup('s'), Pregroup('n')
        >>> assert n.r @ s @ n.l == n.r + s + n.l
        """
        return Pregroup(*super().__add__(other))

    def __getitem__(self, key):
        """
        >>> Pregroup('s', 'n')[1]
        Adjoint('n', 0)
        >>> Pregroup('s', 'n')[1:]
        Pregroup('n')
        """
        if isinstance(key, slice):
            return Pregroup(*super().__getitem__(key))
        return super().__getitem__(key)

    def __repr__(self):
        """
        >>> s, n = Pregroup('s'), Pregroup('n')
        >>> n.r @ s @ n.l
        Pregroup(Adjoint('n', 1), 's', Adjoint('n', -1))
        """
        return "Pregroup({})".format(', '.join(
            repr(x if x._z else x._basic) for x in self))

    def __str__(self):
        """
        >>> s, n = Pregroup('s'), Pregroup('n')
        >>> print(n.r @ s @ n.l)
        n.r @ s @ n.l
        """
        return ' @ '.join(map(str, self)) or "Pregroup()"

    @property
    def l(self):
        """
        >>> s, n = Pregroup('s'), Pregroup('n')
        >>> (s @ n.r).l
        Pregroup('n', Adjoint('s', -1))
        """
        return Pregroup(*[x.l for x in self[::-1]])

    @property
    def r(self):
        """
        >>> s, n = Pregroup('s'), Pregroup('n')
        >>> (s @ n.l).r
        Pregroup('n', Adjoint('s', 1))
        """
        return Pregroup(*[x.r for x in self[::-1]])

    @property
    def is_basic(self):
        """
        >>> s, n = Pregroup('s'), Pregroup('n')
        >>> assert s.is_basic and not s.l.is_basic and not (s @ n).is_basic
        """
        return len(self) == 1 and not self[0]._z

class Diagram(moncat.Diagram):
    """ Implements diagrams in free rigid categories, by checking that the domain
    and codomain are pregroup types.

    >>> n, b = Pregroup('n'), Pregroup('b')
    >>> Alice = Word('b', n)

    # >>> snake_l = Cap(n) @ Wire(n) >> Wire(n) @ Cup(n.l)
    # >>> snake_r = Wire(n) @ Cap(n.r) >> Cup(n) @ Wire(n)

    We take take transposes of any morphism.
    #
    # >>> assert Alice.transpose_r() == Cap(b.r) @ Wire(n.r) >> Wire(b.r) @ Alice @ Wire(n.r) >> Wire(b.r) @ Cup(n)
    # >>> assert Wire(n.l).transpose_r() == snake_l
    # >>> assert Wire(n.r).transpose_l() == snake_r
    """
    def __init__(self, dom, cod, boxes, offsets):
        """
        # >>> n, s = Pregroup('n'), Pregroup('s')
        # >>> Alice, jokes = Word('Alice', n), Word('jokes', n.l @ s)
        # >>> boxes, offsets = [Alice, jokes, Cup(n)], [0, 1, 0]
        # >>> Diagram(Alice.dom @ jokes.dom, s, boxes, offsets)
        """
        if not isinstance(dom, Pregroup):
            raise ValueError("Domain of type Pregroup expected, got {} "
                             "of type {} instead.".format(repr(dom), type(dom)))
        if not isinstance(cod, Pregroup):
            raise ValueError("Codomain of type Pregroup expected, got {} "
                             "of type {} instead.".format(repr(cod), type(cod)))
        super().__init__(dom, cod, boxes, offsets)

    def then(self, other):
        r = super().then(other)
        return Diagram(Pregroup(*r.dom), Pregroup(*r.cod), r.boxes, r.offsets)

    def tensor(self, other):
        r = super().tensor(other)
        return Diagram(Pregroup(*r.dom), Pregroup(*r.cod), r.boxes, r.offsets)

    def dagger(self):
        return Diagram(self.cod, self.dom,
            [f.dagger() for f in self.boxes[::-1]], self.offsets[::-1])

    def __repr__(self):
        return "Diagram(dom={}, cod={}, boxes={}, offsets={})".format(
        *map(repr, [self.dom, self.cod, self.boxes, self.offsets]))

    def __str__(self):
        return repr(self)

    def transpose_r(self):
        a = self.dom
        b = self.cod
        return Diagram.cap(a.r) @ Wire(b.r) >> Wire(a.r) @ self @ Wire(b.r) >> \
               Wire(a.r) @ Cup(b)

    def transpose_l(self):
        a = self.dom
        b = self.cod
        return Wire(b.l) @ Diagram.cap(a) >> Wire(b.l) @ self @ Wire(a.l) >> \
               Cup(b.l) @ Wire(a.l)

    @staticmethod
    def id(t):
        return Wire(t)

    # @staticmethod
    # def cup(t):
    #     """
    #     >>> n, s = Pregroup('n'), Pregroup('s')
    #     >>> Diagram.cup(n @ s @ n).dom
    #     Pregroup('n', 's', 'n', Adjoint('n', 1), Adjoint('s', 1), Adjoint('n', 1))
    #     >>> Diagram.cup(n @ s @ n).boxes
    #     [Cup('n'), Cup('s'), Cup('n')]
    #     >>> Diagram.cup(n @ s @ n).offsets
    #     [2, 1, 0]
    #     >>> Cup(n @ s @ n).cod
    #     Pregroup()
    #     >>> assert Diagram.cup(n) == Diagram.cup('n')
    #     """
    #
    #     if not isinstance(t, Pregroup):
    #         raise ValueError("Input of type Pregroup expected, got {} "
    #                          "of type {} instead.".format(repr(t), type(t)))
    #     boxes = [Cup(b) for b in t[::-1]]
    #     offsets = list(range(len(t)))[::-1]
    #     return Diagram(t @ t.r, Pregroup(), boxes, offsets)

    # @staticmethod
    # def cap(t):
    #     """ Construct caps for pregroup types.
    #
    #     >>> n, s = Pregroup('n'), Pregroup('s')
    #     >>> Diagram.cap(n @ s).dom
    #     Pregroup()
    #     >>> Diagram.cap(n @ s).boxes
    #     [Cap('n'), Cap('s')]
    #     >>> Diagram.cap(n @ s).offsets
    #     [0, 1]
    #     >>> Diagram.cap(n @ s).cod
    #     Pregroup('n', 's', Adjoint('s', -1), Adjoint('n', -1))
    #     >>> assert Diagram.cap(n) == Cap('n')
    #     """
    #     if not isinstance(t, Pregroup):
    #         raise ValueError("Input of type Pregroup expected, got {} "
    #                          "of type {} instead.".format(repr(t), type(t)))
    #     dom = Pregroup()
    #     cod = t @ t.l
    #     boxes = [Cap(b) for b in t[::-1]]
    #     offsets = list(range(len(t)))[::-1]
    #     return Diagram(dom, cod, boxes, offsets)

class AxiomError(moncat.AxiomError):
    """
    >>> Cup(Pregroup('n'), Pregroup('n'))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    disco.AxiomError: n and n are not adjoints.
    >>> Cup(Pregroup('n'), Pregroup('s'))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    disco.AxiomError: n and s are not adjoints.
    >>> Cup(Pregroup('n'), Pregroup('n').l.l)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    disco.AxiomError: n and n.l.l are not adjoints.
    """
    pass

class Wire(Diagram):
    """ Define an identity arrow in a free rigid category

    >>> assert Wire(Pregroup('x')) == Diagram(Pregroup('x'), Pregroup('x'), [], [])
    """
    def __init__(self, t):
        """
        >>> wire = Wire(Pregroup('n') @ Pregroup('s'))
        >>> wire.dom
        Pregroup('n', 's')
        >>> wire.cod
        Pregroup('n', 's')
        """
        if isinstance(t, Word):
            t = t.dom
        if not isinstance(t, Pregroup):
            raise ValueError("Input of type Pregroup expected, got {} "
                             "of type {} instead.".format(repr(t), type(t)))
        super().__init__(t, t, [], [])

    def __repr__(self):
        """
        >>> Wire(Pregroup('n'))
        Wire(Pregroup('n'))
        """
        return "Wire({})".format(repr(self.dom))

    def __str__(self):
        """
        >>> print(Wire(Pregroup('n')))
        Wire(n)
        """
        return "Wire({})".format(str(self.dom))

class Cup(Diagram, Box):
    """ Defines cups for simple types.

    >>> n = Pregroup('n')
    >>> Cup(n, n.l)
    Cup(Pregroup('n'), Pregroup(Adjoint('n', -1)))
    >>> Cup(n, n.r)
    Cup(Pregroup('n'), Pregroup(Adjoint('n', 1)))
    >>> Cup(n.l.l, n.l)
    Cup(Pregroup(Adjoint('n', -2)), Pregroup(Adjoint('n', -1)))
    """
    def __init__(self, x, y):
        """
        >>> Cup(Pregroup('n', 's'), Pregroup('n').l)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Simple type expected, got Pregroup('n', 's') instead.
        >>> Cup(Pregroup('n'), Pregroup())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Simple type expected, got Pregroup() instead.
        >>> Cup(Pregroup('n'), Pregroup('n').l)
        Cup(Pregroup('n'), Pregroup(Adjoint('n', -1)))
        """
        err = "Simple type expected, got {} instead."
        if not isinstance(x, Pregroup) or not len(x) == 1:
            raise ValueError(err.format(repr(x)))
        if not isinstance(y, Pregroup) or not len(y) == 1:
            raise ValueError(err.format(repr(y)))
        if x[0]._basic != y[0]._basic or not x[0]._z - y[0]._z in [-1, +1]:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        Box.__init__(self, 'Cup', x @ y, Pregroup())

    def dagger(self):
        """
        >>> n = Pregroup('n')
        >>> Cup(n, n.l).dagger()
        Cap(Pregroup('n'), Pregroup(Adjoint('n', -1)))
        >>> assert Cup(n, n.l) == Cup(n, n.l).dagger().dagger()
        """
        return Cap(self.dom[:1], self.dom[1:])

    def __repr__(self):
        """
        >>> n = Pregroup('n')
        >>> Cup(n, n.l)
        Cup(Pregroup('n'), Pregroup(Adjoint('n', -1)))
        """
        return "Cup({}, {})".format(repr(self.dom[:1]), repr(self.dom[1:]))

    def __str__(self):
        """
        >>> n = Pregroup('n')
        >>> print(Cup(n, n.l))
        Cup(n, n.l)
        """
        return "Cup({}, {})".format(self.dom[:1], self.dom[1:])

class Cap(Diagram, Box):
    """ Defines cups for simple types.

    >>> n = Pregroup('n')
    >>> print(Cap(n, n.l).cod)
    n @ n.l
    >>> print(Cap(n, n.r).cod)
    n @ n.r
    >>> print(Cap(n.l.l, n.l).cod)
    n.l.l @ n.l
    """
    def __init__(self, x, y):
        """
        >>> Cap(Pregroup('n', 's'), Pregroup('n').l)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Simple type expected, got Pregroup('n', 's') instead.
        >>> Cap(Pregroup('n'), Pregroup())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Simple type expected, got Pregroup() instead.
        >>> Cap(Pregroup('n'), Pregroup('n').l)
        Cap(Pregroup('n'), Pregroup(Adjoint('n', -1)))
        """
        err = "Simple type expected, got {} instead."
        if not isinstance(x, Pregroup) or not len(x) == 1:
            raise ValueError(err.format(repr(x)))
        if not isinstance(y, Pregroup) or not len(y) == 1:
            raise ValueError(err.format(repr(y)))
        if not x[0]._z - y[0]._z in [-1, +1]:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        Box.__init__(self, 'Cap', Pregroup(), x @ y)

    def dagger(self):
        """
        >>> n = Pregroup('n')
        >>> Cap(n, n.l).dagger()
        Cup(Pregroup('n'), Pregroup(Adjoint('n', -1)))
        >>> assert Cap(n, n.l) == Cap(n, n.l).dagger().dagger()
        """
        return Cup(self.cod[:1], self.cod[1:])

    def __repr__(self):
        """
        >>> n = Pregroup('n')
        >>> Cap(n, n.l)
        Cap(Pregroup('n'), Pregroup(Adjoint('n', -1)))
        """
        return "Cap({}, {})".format(repr(self.cod[:1]), repr(self.cod[1:]))

    def __str__(self):
        """
        >>> n = Pregroup('n')
        >>> print(Cap(n, n.l))
        Cap(n, n.l)
        """
        return "Cap({}, {})".format(self.cod[:1], self.cod[1:])

class Word(Diagram, Box):
    """ Encodes words with their pregroup type as diagrams in free rigid categories

    """
    def __init__(self, w, t, _dagger=False):
        """
        >>> n, s = Pregroup('n'), Pregroup('s')
        >>> Alice = Word('Alice', Pregroup('n'))
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
        if not isinstance(w, str):
            raise ValueError("Expected str, got {} of type {} instead."
                             .format(repr(w), type(w)))
        if not isinstance(t, Pregroup):
            raise ValueError("Input of type Pregroup expected, got {} "
                             "of type {} instead.".format(repr(t), type(t)))
        self._word, self._type = w, t
        dom, cod = (t, Pregroup(w)) if _dagger else (Pregroup(w), t)
        Box.__init__(self, (w, t), dom, cod, _dagger=_dagger)

    def dagger(self):
        """
        >>> Alice = Word('Alice', Pregroup('n')).dagger()
        >>> Alice.dom
        Pregroup('n')
        >>> Alice.cod
        Pregroup('Alice')
        """
        return Word(self._word, self._type, not self._dagger)

    @property
    def word(self):
        """
        >>> Word('Alice', Pregroup('n')).word
        'Alice'
        """
        return self._word

    @property
    def type(self):
        """
        >>> Word('loves', Pregroup('n').r @ Pregroup('s') @ Pregroup('n').l).type
        Pregroup(Adjoint('n', 1), 's', Adjoint('n', -1))
        """
        return self._type

    def __repr__(self):
        """
        >>> Word('Alice', Pregroup('n'))
        Word('Alice', Pregroup('n'))
        >>> Word('Alice', Pregroup('n')).dagger()
        Word('Alice', Pregroup('n')).dagger()
        """
        return "Word({}, {}){}".format(repr(self.word), repr(self.type),
                                       ".dagger()" if self._dagger else "")

    def __str__(self):
        """
        >>> print(Word('Alice', Pregroup('n')))
        Alice
        """
        return str(self.word)

class Model(MatrixFunctor):
    """ Implements functors from pregroup grammars to matrices

    >>> n = Pregroup('n')
    >>> F = Model({n: 2}, {})

    # >>> snake_l = Cap(n) @ Wire(n) >> Wire(n) @ Cup(n.l)
    # >>> snake_r = Wire(n) @ Cap(n.r) >> Cup(n) @ Wire(n)
    # >>> assert (F(snake_l) == F(Wire(n))).all()
    # >>> assert (F(Wire(n)) == F(snake_r)).all()
    """
    def __init__(self, ob, ar):
        for x in ob.keys():
            if not isinstance(x, Pregroup) or not x.is_basic:
                raise ValueError(
                    "Expected a basic type, got {} instead.".format(repr(x)))

        for a in ar.keys():
            if not isinstance(a, Word):
                raise ValueError("Expected Word, got {} of type {} instead."
                                 .format(repr(a), type(a)))
        self._types, self._vocab = ob, ar
        #  Rigid functors are defined by their image on basic types.
        ob = {x[0]._basic: ob[x] for x in self._types.keys()}
        #  We assume the images for word boxes are all states.
        ob.update({w.dom[0]._basic: 1 for w in self._vocab.keys()})
        self._ob, self._ar = ob, {f: g for f, g in ar.items()}

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


class CircuitModel(CircuitFunctor, Model):
    """
    >>> from discopy.circuit import *
    >>> s, n = Pregroup('s'), Pregroup('n')
    >>> Alice = Word('Alice', n)
    >>> loves = Word('loves', n.r @ s @ n.l)
    >>> Bob = Word('Bob', n)

    # >>> grammar = Cup(n) @ Wire(s) @ Cup(n.l)
    # >>> sentence = grammar << Alice @ loves @ Bob
    # >>> ob = {s: 0, n: 1}
    # >>> ar = {Alice: Ket(0),
    # ...       loves: CX << H @ X << Ket(0, 0),
    # ...       Bob: Ket(1)}
    # >>> F = CircuitModel(ob, ar)
    # >>> BornRule = lambda c: np.absolute(c.eval().array) ** 2
    # >>> assert 2**3 * BornRule(F(sentence))
    """
    def __init__(self, ob, ar):
        Model.__init__(self, ob, ar)

    def __call__(self, x):
        if isinstance(x, Cup):
            self(x.dom[0]) / 2
            return GCX(n) >> HAD(n) @ Circuit.id(n)
        CircuitFunctor.__call__(self, x)

def parse(words, cups):
    """ Produces the diagram in a free rigid category corresponding to a pregroup parsing.

    >>> s, n = Pregroup('s'), Pregroup('n')
    >>> Alice, Bob, jokes = Word('Alice', n), Word('Bob', n), Word('jokes', n)
    >>> loves, tells = Word('loves', n.r + s + n.l), Word('tells', n.r + s + n.l)
    >>> who = Word('who', n.r + n.l.r + s.l + n)

    A parse is given by a list of words and a list of offsets for the cups.

    >>> parse0 = parse([Alice, loves, Bob], [0, 1])
    >>> parse([Alice, loves, Bob], [0, 2])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    IndexError: list index out of range
    >>> parse([Box('Alice', Pregroup('Alice'), Pregroup('n')), loves, Bob], [0, 1])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Word expected, got Box(...) of type <class 'discopy.moncat.Box'> instead.
    >>> parse1 = parse([Alice, loves, Bob, who, tells, jokes], [0, 2, 1, 2, 1, 1])

    A sentence u is grammatical if there is a parse with domain u and codomain s.

    >>> parse0.dom
    Pregroup('Alice', 'loves', 'Bob')
    >>> parse1.dom
    Pregroup('Alice', 'loves', 'Bob', 'who', 'tells', 'jokes')
    >>> assert parse0.cod == Pregroup('s') == parse1.cod
    """
    for w in words:
        if not isinstance(w, Word):
            raise ValueError("Word expected, got {} instead.".format(repr(w)))
    dom = sum((w.dom for w in words), Pregroup())
    boxes = words[::-1]  # words are backwards to make offsets easier
    offsets = [len(words) - i - 1 for i in range(len(words))] + cups
    cod = sum((w.type for w in words), Pregroup())
    for i in cups:
        if cod[i].r != cod[i + 1]:
            raise AxiomError("There can be no Cup of type {}."
                                   .format(cod[i: i + 2]))
        boxes.append(Cup(cod[i]))
        cod = cod[:i] + cod[i + 2:]
    return Diagram(dom, cod, boxes, offsets)
