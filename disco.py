""" Implements free rigid categories and distributional compositional models.

>>> s, n = Pregroup('s'), Pregroup('n')
>>> Alice, Bob = Word('Alice', n), Word('Bob', n)
>>> loves = Word('loves', n.r @ s @ n.l)
>>> grammar = Cup(n, n.r) @ Wire(s) @ Cup(n.l, n)
>>> sentence = grammar << Alice @ loves @ Bob
>>> ob = {s: 1, n: 2}
>>> ar = {Alice: [1, 0], loves: [0, 1, 1, 0], Bob: [0, 1]}
>>> F = Model(ob, ar)

# >>> assert F(sentence) == True
"""

import numpy as np
from discopy import cat, moncat
from discopy.moncat import Ob, Ty, Diagram
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
    """ Implements diagrams in free dagger pivotal categories.

    >>> n, s = Pregroup('n'), Pregroup('s')
    >>> Alice, jokes = Word('Alice', n), Word('jokes', n.l @ s)
    >>> boxes, offsets = [Alice, jokes, Cup(n, n.l)], [0, 1, 0]
    >>> print(Diagram(Alice.dom @ jokes.dom, s, boxes, offsets))
    Alice >> Wire(n) @ jokes >> Cup(n, n.l) @ Wire(s)
    """
    def __init__(self, dom, cod, boxes, offsets):
        """
        >>> a, b = Pregroup('a'), Pregroup('b')
        >>> f, g = Box('f', a, a.l @ b.r), Box('g', b.r, b.r)
        >>> print(Diagram(a, a, [f, g, f.dagger()], [0, 1, 0]))
        f >> Wire(a.l) @ g >> f.dagger()
        """
        if not isinstance(dom, Pregroup):
            raise ValueError("Domain of type Pregroup expected, got {} "
                             "of type {} instead.".format(repr(dom), type(dom)))
        if not isinstance(cod, Pregroup):
            raise ValueError("Codomain of type Pregroup expected, got {} "
                             "of type {} instead.".format(repr(cod), type(cod)))
        super().__init__(dom, cod, boxes, offsets)

    def then(self, other):
        """
        >>> a, b = Pregroup('a'), Pregroup('b')
        >>> f = Box('f', a, a.l @ b.r)
        >>> print(f >> f.dagger() >> f)
        f >> f.dagger() >> f
        """
        r = super().then(other)
        return Diagram(Pregroup(*r.dom), Pregroup(*r.cod), r.boxes, r.offsets)

    def tensor(self, other):
        """
        >>> a, b = Pregroup('a'), Pregroup('b')
        >>> f = Box('f', a, a.l @ b.r)
        >>> print(f.dagger() @ f)
        f.dagger() @ Wire(a) >> Wire(a) @ f
        """
        r = super().tensor(other)
        return Diagram(Pregroup(*r.dom), Pregroup(*r.cod), r.boxes, r.offsets)

    def dagger(self):
        """
        >>> a, b = Pregroup('a'), Pregroup('b')
        >>> f = Box('f', a, a.l @ b.r).dagger()
        >>> assert f.dagger() >> f == (f.dagger() >> f).dagger()
        """
        return Diagram(self.cod, self.dom,
            [f.dagger() for f in self.boxes[::-1]], self.offsets[::-1])

    def __repr__(self):
        """
        >>> Diagram(Pregroup('a'), Pregroup('a'), [], [])
        Diagram(dom=Pregroup('a'), cod=Pregroup('a'), boxes=[], offsets=[])
        """
        return "Diagram(dom={}, cod={}, boxes={}, offsets={})".format(
        *map(repr, [self.dom, self.cod, self.boxes, self.offsets]))

    @staticmethod
    def id(t):
        """
        >>> assert Diagram.id(Pregroup('s')) == Wire(Pregroup('s'))
        """
        return Wire(t)

class Box(cat.Gen, Diagram):
    """ Implements generators of dagger pivotal diagrams.

    >>> a, b = Pregroup('a'), Pregroup('b')
    >>> Box('f', a, b.l @ b, data={42})
    Box('f', Pregroup('a'), Pregroup(Adjoint('b', -1), 'b'), data={42})
    """
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        """
        >>> a, b = Pregroup('a'), Pregroup('b')
        >>> Box('f', a, b.l @ b)
        Box('f', Pregroup('a'), Pregroup(Adjoint('b', -1), 'b'))
        """
        self._dom, self._cod, self._boxes, self._offsets = dom, cod, [self], [0]
        self._name, self._dagger, self._data = name, _dagger, data
        Diagram.__init__(self, dom, cod, [self], [0])

    def dagger(self):
        """
        >>> a, b = Pregroup('a'), Pregroup('b')
        >>> Box('f', a, b.l @ b).dagger()
        Box('f', Pregroup('a'), Pregroup(Adjoint('b', -1), 'b')).dagger()
        """
        return Box(self.name, self.cod, self.dom,
                   _dagger=not self._dagger, data=self.data)

    def __repr__(self):
        """
        >>> a, b = Pregroup('a'), Pregroup('b')
        >>> Box('f', a, b.l @ b)
        Box('f', Pregroup('a'), Pregroup(Adjoint('b', -1), 'b'))
        >>> Box('f', a, b.l @ b).dagger()
        Box('f', Pregroup('a'), Pregroup(Adjoint('b', -1), 'b')).dagger()
        """
        if self._dagger:
            return repr(self.dagger()) + ".dagger()"
        return "Box({}, {}, {}{})".format(
            *map(repr, [self.name, self.dom, self.cod]),
            ", data=" + repr(self.data) if self.data else '')

    def __hash__(self):
        """
        >>> a, b = Pregroup('a'), Pregroup('b')
        >>> f = Box('f', a, b.l @ b)
        >>> {f: 42}[f]
        42
        """
        return hash(repr(self))

    def __eq__(self, other):
        """
        >>> a, b = Pregroup('a'), Pregroup('b')
        >>> f = Box('f', a, b.l @ b)
        >>> assert f == Diagram(a, b.l @ b, [f], [0])
        """
        if isinstance(other, Box):
            return repr(self) == repr(other)
        elif isinstance(other, Diagram):
            return len(other) == 1 and other.boxes[0] == self
        return False

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

    >>> t = Pregroup('a', 'b', 'c')
    >>> assert Wire(t) == Diagram(t, t, [], [])
    """
    def __init__(self, t):
        """
        >>> Wire(Pregroup('n') @ Pregroup('s'))
        Wire(Pregroup('n', 's'))
        >>> Wire('n')  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Input of type Pregroup expected, got 'n' instead.
        """
        if isinstance(t, Word):
            t = t.dom
        if not isinstance(t, Pregroup):
            raise ValueError("Input of type Pregroup expected, got {} instead."
                             .format(repr(t), type(t)))
        super().__init__(t, t, [], [])

    def __repr__(self):
        """
        >>> Wire(Pregroup('n'))
        Wire(Pregroup('n'))
        """
        return "Wire({})".format(repr(self.dom))

    def __str__(self):
        """
        >>> n = Pregroup('n')
        >>> print(Wire(n))
        Wire(n)
        """
        return "Wire({})".format(str(self.dom))

class Cup(Box):
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

class Cap(Box):
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

class Word(Box):
    """ Encodes words with their pregroup type as diagrams in free rigid categories

    >>> Alice = Word('Alice', Pregroup('n'))
    >>> loves = Word('loves', Pregroup('n').r @ Pregroup('s') @ Pregroup('n').l)
    >>> Alice
    Word('Alice', Pregroup('n'))
    >>> loves
    Word('loves', Pregroup(Adjoint('n', 1), 's', Adjoint('n', -1)))
    """
    def __init__(self, w, t, _dagger=False):
        """
        >>> Word('Alice', Pregroup('n'))
        Word('Alice', Pregroup('n'))
        """
        if not isinstance(w, str):
            raise ValueError("Expected str, got {} of type {} instead."
                             .format(repr(w), type(w)))
        if not isinstance(t, Pregroup):
            raise ValueError("Input of type Pregroup expected, got {} "
                             "of type {} instead.".format(repr(t), type(t)))
        self._word, self._type = w, t
        dom, cod = (t, Pregroup()) if _dagger else (Pregroup(), t)
        Box.__init__(self, (w, t), dom, cod, _dagger=_dagger)

    def dagger(self):
        """
        >>> Word('Alice', Pregroup('n')).dagger()
        Word('Alice', Pregroup('n')).dagger()
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

    >>> n, s = Pregroup('n'), Pregroup('s')
    >>> Alice, jokes = Word('Alice', n), Word('jokes', n.l @ s)
    >>> F = Model({s: 1, n: 2}, {Alice: [0, 1], jokes: [0, 1]})
    >>> parsing = Alice @ jokes >> Cup(n, n.l) @ Wire(s)
    >>> assert F(parsing).array
    """
    def __init__(self, ob, ar):
        for x in ob.keys():
            if not isinstance(x, Pregroup) or not x.is_basic:
                raise ValueError(
                    "Expected a basic type, got {} instead.".format(repr(x)))
        super().__init__(ob, ar)

    def __repr__(self):
        return super().__repr__().replace("MatrixFunctor", "Model")

    def __call__(self, d):
        if isinstance(d, Pregroup):
            return sum([self.ob[Pregroup(x._basic)] for x in d], Dim(1))
        elif isinstance(d, Cup):
            return Matrix(self(d.dom), Dim(), Matrix.id(self(d.dom[:1])).array)
        elif isinstance(d, Cap):
            return Matrix(Dim(), self(d.cod), Matrix.id(self(d.cod[:1])).array)
        elif isinstance(d, Box):
            if d._dagger:
                return self(d.dagger()).dagger()
            return Matrix(self(d.dom), self(d.cod), self.ar[d])
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
