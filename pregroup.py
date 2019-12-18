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

    def __pow__(self, other):
        """
        >>> Ty('x') ** 3
        Ty('x', 'x', 'x')
        """
        if not isinstance(other, int):
            raise ValueError(
                "Expected int, got {} instead.".format(repr(other)))
        return sum(other * (self, ), Ty())

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
        result = super().then(other)
        return Diagram(Ty(*result.dom), Ty(*result.cod),
                       result.boxes, result.offsets, fast=True)

    def tensor(self, other):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> f = Box('f', a, a.l @ b.r)
        >>> print(f.dagger() @ f)
        f.dagger() @ Wire(a) >> Wire(a) @ f
        """
        result = super().tensor(other)
        return Diagram(Ty(*result.dom), Ty(*result.cod),
                       result.boxes, result.offsets, fast=True)

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
    def id(x):
        """
        >>> assert Diagram.id(Ty('s')) == Wire(Ty('s'))
        """
        return Wire(x)

    @staticmethod
    def cups(x, y):
        """ Constructs nested cups witnessing adjointness of x and y

        >>> a, b = Ty('a'), Ty('b')
        >>> Diagram.cups(a @ b @ a, a.r @ b.r) #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        pregroup.AxiomError: a @ b @ a and a.r @ b.r are not adjoints.
        >>> assert Diagram.cups(a, a.r) == Cup(a, a.r)
        >>> assert Diagram.cups(a @ b, (a @ b).l) == (Cup(a, a.l)
        ...                 << Wire(a) @ Cup(b, b.l) @ Wire(a.l))
        """
        if x.r != y and x != y.r:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        cups = Wire(x @ y)
        for i in range(len(x)):
            j = len(x) - i - 1
            cups = cups >> Wire(x[:j]) @ Cup(x[j], y[i]) @ Wire(y[i + 1:])
        return cups

    @staticmethod
    def caps(x, y):
        """ Constructs nested cups witnessing adjointness of x and y

        >>> a, b = Ty('a'), Ty('b')
        >>> Diagram.caps( a @ b @ a, a.l @ b.l) #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        pregroup.AxiomError: a @ b @ a and a.l @ b.l are not adjoints.
        >>> assert Diagram.caps(a, a.r) == Cap(a, a.r)
        >>> assert Diagram.caps(a @ b, (a @ b).l) == (Cap(a, a.l)
        ...                 >> Wire(a) @ Cap(b, b.l) @ Wire(a.l))
        """
        if x.r != y and x != y.r:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        caps = Wire(x @ y)
        for i in range(len(x)):
            j = len(x) - i - 1
            caps = caps << Wire(x[:j]) @ Cap(x[j], y[i]) @ Wire(y[i + 1:])
        return caps

    def transpose_r(self):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> snake_r = Wire(a).transpose_r()
        """
        return Diagram.caps(self.dom.r, self.dom) @ Wire(self.cod.r)\
            >> Wire(self.dom.r) @ self @ Wire(self.cod.r)\
            >> Wire(self.dom.r) @ Diagram.cups(self.dom, self.dom.r)

    def interchange(self, i, j, left=False):
        """
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x.r, y.l)
        >>> d = (f @ f.dagger()).interchange(0, 1)
        >>> assert d == Wire(x.r) @ f.dagger() >> f @ Wire(x.r)
        >>> print((Cup(x, x.l) >> Cap(x, x.r)).interchange(0, 1))
        Cap(x, x.r) @ Wire(x @ x.l) >> Wire(x @ x.r) @ Cup(x, x.l)
        >>> print((Cup(x, x.l) >> Cap(x, x.r)).interchange(0, 1, left=True))
        Wire(x @ x.l) @ Cap(x, x.r) >> Cup(x, x.l) @ Wire(x @ x.r)
        """
        result = super().interchange(i, j, left=left)
        return Diagram(Ty(*result.dom), Ty(*result.cod),
                       result.boxes, result.offsets, fast=True)

    def normal_form(self):
        """
        Implements the normalisation of rigid monoidal categories,
        see arxiv:1601.05372, definition 2.12.

        >>> n, a = Ty('n'), Ty('a')
        >>> cup = Diagram.cups(n ** 5, (n ** 5).r)
        >>> cap = Diagram.caps((n ** 5).r, n ** 5)
        >>> snake = cup @ Wire(n ** 5) << Wire(n ** 5) @ cap
        >>> assert snake.normal_form() == Wire(n ** 5)
        >>> assert snake.dagger().normal_form() == Wire(n ** 5)
        >>> cup, cap = Cup(n, n.r), Cap(n.r, n)
        >>> f_n = Wire(n)
        >>> for _ in range(2):
        ...     f_n = f_n >> Box('f0', n, n @ n) >> Box('f1', n @ n, n)
        >>> g, h = Box('g', a @ n, n), Box('h', n, n @ a)
        >>> d0 = g @ cap >> f_n.dagger() @ Wire(n.r) @ f_n >> cup @ h
        >>> d1 = g >> f_n.dagger() >> f_n >> h
        >>> assert d0.normal_form() == d1
        >>> assert d0.dagger().normal_form() == d1.dagger()
        """
        self = super().normal_form()

        def unsnake(diagram, cup, cap):
            return Diagram(diagram.dom, diagram.cod,
                           diagram.boxes[:cap] + diagram.boxes[cup + 1:],
                           diagram.offsets[:cap] + diagram.offsets[cup + 1:])

        def left_unsnake(diagram, cup, cap,
                         left_obstruction, right_obstruction):
            for left in left_obstruction:
                diagram = diagram.interchange(cap, left)
                for i, right in enumerate(right_obstruction):
                    if right < left:
                        right_obstruction[i] += 1
                cap += 1
            for right in right_obstruction[::-1]:
                diagram = diagram.interchange(right, cup)
                cup -= 1
            return unsnake(diagram, cup, cap)

        def right_unsnake(diagram, cup, cap,
                          left_obstruction, right_obstruction):
            for left in left_obstruction[::-1]:
                diagram = diagram.interchange(left, cup)
                for i, right in enumerate(right_obstruction):
                    if right > left:
                        right_obstruction[i] -= 1
                cup -= 1
            for right in right_obstruction:
                diagram = diagram.interchange(cap, right)
                cap += 1
            return unsnake(diagram, cup, cap)

        for i, (box0, off0) in enumerate(zip(self.boxes, self.offsets)):
            if isinstance(box0, Cap):
                for scan in [off0, off0 + 1]:
                    rewrite = left_unsnake if scan == off0 else right_unsnake
                    left_obstruction, right_obstruction = [], []
                    for j in range(i + 1, len(self)):
                        box1, off1 = self.boxes[j], self.offsets[j]
                        if off1 <= scan < off1 + len(box1.dom):
                            if not isinstance(box1, Cup):
                                break
                            return rewrite(
                                self, j, i,
                                left_obstruction, right_obstruction)\
                                .normal_form()
                        elif off1 < scan:
                            scan += len(box1.cod) - len(box1.dom)
                            left_obstruction.append(j)
                        else:
                            right_obstruction.append(j)
        return self


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


class Cup(Box, Diagram):
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
        if isinstance(x, Ob):
            x = Ty(x)
        if isinstance(y, Ob):
            y = Ty(y)
        if not isinstance(x, Ty) or not len(x) == 1:
            raise ValueError(err.format(repr(x)))
        if not isinstance(y, Ty) or not len(y) == 1:
            raise ValueError(err.format(repr(y)))
        if x[0].name != y[0].name or not x[0].z - y[0].z in [-1, +1]:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        self._x, self._y = x, y
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
        if isinstance(x, Ob):
            x = Ty(x)
        if isinstance(y, Ob):
            y = Ty(y)
        err = "Simple type expected, got {} instead."
        if not isinstance(x, Ty) or not len(x) == 1:
            raise ValueError(err.format(repr(x)))
        if not isinstance(y, Ty) or not len(y) == 1:
            raise ValueError(err.format(repr(y)))
        if not x[0].z - y[0].z in [-1, +1]:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        self._x, self._y = x, y
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


class RigidFunctor(moncat.MonoidalFunctor):
    """Implements functors between rigid categories preserving cups and caps

    >>> s, n, a = Ty('s'), Ty('n'), Ty('a')
    >>> loves = Box('loves', Ty(), n.r @ s @ n.l)
    >>> love_box = Box('loves', a @ a, s)
    >>> ob = {s: s, n: a, a: n @ n}
    >>> ar = {loves: Cap(a.r, a) @ Cap(a, a.l)
    ...              >> Wire(a.r) @ love_box @ Wire(a.l)}
    >>> F = RigidFunctor(ob, ar)
    >>> assert F(Cap(n.r, n)) == Cap(Ty(Ob('a', z=1)), Ty('a'))
    >>> assert F(Cup(a, a.l)) == Diagram.cups(n @ n, (n @ n).l)
    """
    def __call__(self, diagram):
        if isinstance(diagram, Ob):
            result = self.ob[Ty(diagram.name)]
            if diagram.z < 0:
                for i in range(-diagram.z):
                    result = result.l
            elif diagram.z > 0:
                for i in range(diagram.z):
                    result = result.r
            return result
        if isinstance(diagram, Ty):
            return sum([self(b) for b in diagram.objects], Ty())
        if isinstance(diagram, Cup):
            return Diagram.cups(self(diagram._x), self(diagram._y))
        if isinstance(diagram, Cap):
            return Diagram.caps(self(diagram._x), self(diagram._y))
        if isinstance(diagram, Diagram):
            return super().__call__(diagram)
