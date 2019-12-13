# -*- coding: utf-8 -*-

"""
Implements free dagger pivotal and rigid monoidal categories.
The objects are given by the free pregroup, the arrows by planar diagrams.

>>> unit, s, n = Ty(), Ty('s'), Ty('n')
>>> t = n.r @ s @ n.l
>>> assert t @ unit == t == unit @ t
>>> assert t.l.r == t == t.r.l
>>> snake_l = Cap(n, n.l) @ Wire(n) >> Wire(n) @ Cup(n.l, n)
>>> snake_r = Wire(n) @ Cap(n.r, n) >> Cup(n, n.r) @ Wire(n)
>>> assert snake_l.dagger().dagger() == snake_l
>>> assert (snake_l >> snake_r).dagger()\\
...         == snake_l.dagger() << snake_r.dagger()
"""

from discopy import cat, moncat


class Ob(cat.Ob):
    """
    Implements simple pregroup types: basic types and their iterated adjoints.

    >>> a = Ob('a')
    >>> assert a.l.r == a.r.l == a and a != a.l.l != a.r.r
    """
    def __init__(self, name, z=0):
        """
        >>> print(Ob('a'))
        a
        >>> print(Ob('a', z=-2))
        a.l.l
        """
        if not isinstance(z, int):
            raise ValueError("Expected int, got {} instead".format(repr(z)))
        self._z = z
        super().__init__(name)

    @property
    def z(self):
        """
        >>> Ob('a').z
        0
        """
        return self._z

    @property
    def l(self):
        """
        >>> Ob('a').l
        Ob('a', z=-1)
        """
        return Ob(self.name, self.z - 1)

    @property
    def r(self):
        """
        >>> Ob('a').r
        Ob('a', z=1)
        """
        return Ob(self.name, self.z + 1)

    def __eq__(self, other):
        """
        >>> assert Ob('a') == Ob('a').l.r
        """
        if not isinstance(other, Ob):
            return False
        return (self.name, self.z) == (other.name, other.z)

    def __repr__(self):
        """
        >>> Ob('a', z=42)
        Ob('a', z=42)
        """
        return "Ob({}{})".format(
            repr(self.name), ", z=" + repr(self.z) if self.z else '')

    def __str__(self):
        """
        >>> a = Ob('a')
        >>> print(a)
        a
        >>> print(a.r)
        a.r
        >>> print(a.l)
        a.l
        """
        return str(self.name) + (
            - self.z * '.l' if self.z < 0 else self.z * '.r')


class Ty(moncat.Ty):
    """ Implements pregroup types as lists of simple types.

    >>> s, n = Ty('s'), Ty('n')
    >>> assert n.l.r == n == n.r.l
    >>> assert (s @ n).l == n.l @ s.l and (s @ n).r == n.r @ s.r
    """
    def __init__(self, *t):
        """
        >>> Ty('s', 'n')
        Ty('s', 'n')
        """
        t = [x if isinstance(x, Ob) else Ob(x) for x in t]
        super().__init__(*t)

    def __matmul__(self, other):
        """
        >>> s, n = Ty('s'), Ty('n')
        >>> assert n.r @ s == Ty(Ob('n', z=1), 's')
        """
        return Ty(*super().__matmul__(other))

    def __getitem__(self, key):
        """
        >>> Ty('s', 'n')[1]
        Ob('n')
        >>> Ty('s', 'n')[1:]
        Ty('n')
        """
        if isinstance(key, slice):
            return Ty(*super().__getitem__(key))
        return super().__getitem__(key)

    def __repr__(self):
        """
        >>> s, n = Ty('s'), Ty('n')
        >>> n.r @ s @ n.l
        Ty(Ob('n', z=1), 's', Ob('n', z=-1))
        """
        return "Ty({})".format(', '.join(
            repr(x if x.z else x.name) for x in self.objects))

    def __str__(self):
        """
        >>> s, n = Ty('s'), Ty('n')
        >>> print(n.r @ s @ n.l)
        n.r @ s @ n.l
        """
        return ' @ '.join(map(str, self.objects)) or "Ty()"

    @property
    def l(self):
        """
        >>> s, n = Ty('s'), Ty('n')
        >>> (s @ n.r).l
        Ty('n', Ob('s', z=-1))
        """
        return Ty(*[x.l for x in self.objects[::-1]])

    @property
    def r(self):
        """
        >>> s, n = Ty('s'), Ty('n')
        >>> (s @ n.l).r
        Ty('n', Ob('s', z=1))
        """
        return Ty(*[x.r for x in self.objects[::-1]])

    @property
    def is_basic(self):
        """
        >>> s, n = Ty('s'), Ty('n')
        >>> assert s.is_basic and not s.l.is_basic and not (s @ n).is_basic
        """
        return len(self) == 1 and not self.objects[0].z


class Diagram(moncat.Diagram):
    """ Implements diagrams in free dagger pivotal categories.

    >>> I, n, s = Ty(), Ty('n'), Ty('s')
    >>> Alice, jokes = Box('Alice', I, n), Box('jokes', I, n.l @ s)
    >>> boxes, offsets = [Alice, jokes, Cup(n, n.l)], [0, 1, 0]
    >>> print(Diagram(Alice.dom @ jokes.dom, s, boxes, offsets))
    Alice >> Wire(n) @ jokes >> Cup(n, n.l) @ Wire(s)
    """
    def __init__(self, dom, cod, boxes, offsets, fast=False):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> f, g = Box('f', a, a.l @ b.r), Box('g', b.r, b.r)
        >>> print(Diagram(a, a, [f, g, f.dagger()], [0, 1, 0]))
        f >> Wire(a.l) @ g >> f.dagger()
        """
        if not isinstance(dom, Ty):
            raise ValueError(
                "Domain of type Ty expected, got {} of type {} instead."
                .format(repr(dom), type(dom)))
        if not isinstance(cod, Ty):
            raise ValueError(
                "Codomain of type Ty expected, got {} of type {}"
                " instead.".format(repr(cod), type(cod)))
        super().__init__(dom, cod, boxes, offsets, fast=fast)

    def then(self, other):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> f = Box('f', a, a.l @ b.r)
        >>> print(f >> f.dagger() >> f)
        f >> f.dagger() >> f
        """
        r = super().then(other)
        return Diagram(Ty(*r.dom), Ty(*r.cod), r.boxes, r.offsets, fast=True)

    def tensor(self, other):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> f = Box('f', a, a.l @ b.r)
        >>> print(f.dagger() @ f)
        f.dagger() @ Wire(a) >> Wire(a) @ f
        """
        r = super().tensor(other)
        return Diagram(Ty(*r.dom), Ty(*r.cod), r.boxes, r.offsets, fast=True)

    def dagger(self):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> f = Box('f', a, a.l @ b.r).dagger()
        >>> assert f.dagger() >> f == (f.dagger() >> f).dagger()
        """
        return Diagram(
            self.cod, self.dom,
            [f.dagger() for f in self.boxes[::-1]], self.offsets[::-1],
            fast=True)

    def __repr__(self):
        """
        >>> Diagram(Ty('a'), Ty('a'), [], [])
        Diagram(dom=Ty('a'), cod=Ty('a'), boxes=[], offsets=[])
        """
        return "Diagram(dom={}, cod={}, boxes={}, offsets={})".format(
            *map(repr, [self.dom, self.cod, self.boxes, self.offsets]))

    @staticmethod
    def id(t):
        """
        >>> assert Diagram.id(Ty('s')) == Wire(Ty('s'))
        """
        return Wire(t)


class Box(moncat.Box, Diagram):
    """ Implements generators of dagger pivotal diagrams.

    >>> a, b = Ty('a'), Ty('b')
    >>> Box('f', a, b.l @ b, data={42})
    Box('f', Ty('a'), Ty(Ob('b', z=-1), 'b'), data={42})
    """
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> Box('f', a, b.l @ b)
        Box('f', Ty('a'), Ty(Ob('b', z=-1), 'b'))
        """
        self._dom, self._cod = dom, cod
        self._boxes, self._offsets = [self], [0]
        self._name, self._dagger, self._data = name, _dagger, data
        moncat.Box.__init__(self, name, dom, cod, data=data, _dagger=_dagger)
        Diagram.__init__(self, dom, cod, [self], [0], fast=True)

    def dagger(self):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> Box('f', a, b.l @ b).dagger()
        Box('f', Ty('a'), Ty(Ob('b', z=-1), 'b')).dagger()
        """
        return Box(self.name, self.cod, self.dom,
                   _dagger=not self._dagger, data=self.data)

    def __hash__(self):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> f = Box('f', a, b.l @ b)
        >>> {f: 42}[f]
        42
        """
        return hash(repr(self))


class AxiomError(moncat.AxiomError):
    """
    >>> Cup(Ty('n'), Ty('n'))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    pregroup.AxiomError: n and n are not adjoints.
    >>> Cup(Ty('n'), Ty('s'))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    pregroup.AxiomError: n and s are not adjoints.
    >>> Cup(Ty('n'), Ty('n').l.l)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    pregroup.AxiomError: n and n.l.l are not adjoints.
    """


class Wire(Diagram):
    """ Define an identity arrow in a free rigid category

    >>> t = Ty('a', 'b', 'c')
    >>> assert Wire(t) == Diagram(t, t, [], [])
    """
    def __init__(self, t):
        """
        >>> Wire(Ty('n') @ Ty('s'))
        Wire(Ty('n', 's'))
        >>> Wire('n')  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Input of type Ty expected, got 'n' instead.
        """
        if not isinstance(t, Ty):
            raise ValueError(
                "Input of type Ty expected, got {} instead.".format(repr(t)))
        super().__init__(t, t, [], [], fast=True)

    def __repr__(self):
        """
        >>> Wire(Ty('n'))
        Wire(Ty('n'))
        """
        return "Wire({})".format(repr(self.dom))

    def __str__(self):
        """
        >>> n = Ty('n')
        >>> print(Wire(n))
        Wire(n)
        """
        return "Wire({})".format(str(self.dom))


class Cup(Box):
    """ Defines cups for simple types.

    >>> n = Ty('n')
    >>> Cup(n, n.l)
    Cup(Ty('n'), Ty(Ob('n', z=-1)))
    >>> Cup(n, n.r)
    Cup(Ty('n'), Ty(Ob('n', z=1)))
    >>> Cup(n.l.l, n.l)
    Cup(Ty(Ob('n', z=-2)), Ty(Ob('n', z=-1)))
    """
    def __init__(self, x, y):
        """
        >>> Cup(Ty('n', 's'), Ty('n').l)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Simple type expected, got Ty('n', 's') instead.
        >>> Cup(Ty('n'), Ty())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Simple type expected, got Ty() instead.
        >>> Cup(Ty('n'), Ty('n').l)
        Cup(Ty('n'), Ty(Ob('n', z=-1)))
        """
        err = "Simple type expected, got {} instead."
        if not isinstance(x, Ty) or not len(x) == 1:
            raise ValueError(err.format(repr(x)))
        if not isinstance(y, Ty) or not len(y) == 1:
            raise ValueError(err.format(repr(y)))
        if x[0].name != y[0].name or not x[0].z - y[0].z in [-1, +1]:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        super().__init__('Cup', x @ y, Ty())

    def dagger(self):
        """
        >>> n = Ty('n')
        >>> Cup(n, n.l).dagger()
        Cap(Ty('n'), Ty(Ob('n', z=-1)))
        >>> assert Cup(n, n.l) == Cup(n, n.l).dagger().dagger()
        """
        return Cap(self.dom[:1], self.dom[1:])

    def __repr__(self):
        """
        >>> n = Ty('n')
        >>> Cup(n, n.l)
        Cup(Ty('n'), Ty(Ob('n', z=-1)))
        """
        return "Cup({}, {})".format(repr(self.dom[:1]), repr(self.dom[1:]))

    def __str__(self):
        """
        >>> n = Ty('n')
        >>> print(Cup(n, n.l))
        Cup(n, n.l)
        """
        return "Cup({}, {})".format(self.dom[:1], self.dom[1:])


class Cap(Box):
    """ Defines cups for simple types.

    >>> n = Ty('n')
    >>> print(Cap(n, n.l).cod)
    n @ n.l
    >>> print(Cap(n, n.r).cod)
    n @ n.r
    >>> print(Cap(n.l.l, n.l).cod)
    n.l.l @ n.l
    """
    def __init__(self, x, y):
        """
        >>> Cap(Ty('n', 's'), Ty('n').l)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Simple type expected, got Ty('n', 's') instead.
        >>> Cap(Ty('n'), Ty())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Simple type expected, got Ty() instead.
        >>> Cap(Ty('n'), Ty('n').l)
        Cap(Ty('n'), Ty(Ob('n', z=-1)))
        """
        err = "Simple type expected, got {} instead."
        if not isinstance(x, Ty) or not len(x) == 1:
            raise ValueError(err.format(repr(x)))
        if not isinstance(y, Ty) or not len(y) == 1:
            raise ValueError(err.format(repr(y)))
        if not x[0].z - y[0].z in [-1, +1]:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        super().__init__('Cap', Ty(), x @ y)

    def dagger(self):
        """
        >>> n = Ty('n')
        >>> Cap(n, n.l).dagger()
        Cup(Ty('n'), Ty(Ob('n', z=-1)))
        >>> assert Cap(n, n.l) == Cap(n, n.l).dagger().dagger()
        """
        return Cup(self.cod[:1], self.cod[1:])

    def __repr__(self):
        """
        >>> n = Ty('n')
        >>> Cap(n, n.l)
        Cap(Ty('n'), Ty(Ob('n', z=-1)))
        """
        return "Cap({}, {})".format(repr(self.cod[:1]), repr(self.cod[1:]))

    def __str__(self):
        """
        >>> n = Ty('n')
        >>> print(Cap(n, n.l))
        Cap(n, n.l)
        """
        return "Cap({}, {})".format(self.cod[:1], self.cod[1:])
