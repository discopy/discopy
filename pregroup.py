# -*- coding: utf-8 -*-

"""
Implements free dagger pivotal and rigid monoidal categories.
The objects are given by the free pregroup, the arrows by planar diagrams.

>>> unit, s, n = Ty(), Ty('s'), Ty('n')
>>> t = n.r @ s @ n.l
>>> assert t @ unit == t == unit @ t
>>> assert t.l.r == t == t.r.l
>>> snake_l = Cap(n, n.l) @ Id(n) >> Id(n) @ Cup(n.l, n)
>>> snake_r = Id(n) @ Cap(n.r, n) >> Cup(n, n.r) @ Id(n)
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
    def z(self):  # pylint: disable=invalid-name
        """
        >>> Ob('a').z
        0
        """
        return self._z

    @property
    def l(self):  # pylint: disable=invalid-name
        """
        >>> Ob('a').l
        Ob('a', z=-1)
        """
        return Ob(self.name, self.z - 1)

    @property
    def r(self):  # pylint: disable=invalid-name
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

    @property
    def l(self):  # pylint: disable=invalid-name
        """
        >>> s, n = Ty('s'), Ty('n')
        >>> (s @ n.r).l
        Ty('n', Ob('s', z=-1))
        """
        return Ty(*[x.l for x in self.objects[::-1]])

    @property
    def r(self):  # pylint: disable=invalid-name
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
    Alice >> Id(n) @ jokes >> Cup(n, n.l) @ Id(s)
    """
    def __init__(self, dom, cod, boxes, offsets, _fast=False):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> f, g = Box('f', a, a.l @ b.r), Box('g', b.r, b.r)
        >>> print(Diagram(a, a, [f, g, f.dagger()], [0, 1, 0]))
        f >> Id(a.l) @ g >> f.dagger()
        """
        if not isinstance(dom, Ty):
            raise ValueError(
                "Domain of type Ty expected, got {} of type {} instead."
                .format(repr(dom), type(dom)))
        if not isinstance(cod, Ty):
            raise ValueError(
                "Codomain of type Ty expected, got {} of type {}"
                " instead.".format(repr(cod), type(cod)))
        super().__init__(dom, cod, boxes, offsets, _fast=_fast)

    def then(self, other):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> f = Box('f', a, a.l @ b.r)
        >>> print(f >> f.dagger() >> f)
        f >> f.dagger() >> f
        """
        result = super().then(other)
        return Diagram(Ty(*result.dom), Ty(*result.cod),
                       result.boxes, result.offsets, _fast=True)

    def tensor(self, other):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> f = Box('f', a, a.l @ b.r)
        >>> print(f.dagger() @ f)
        f.dagger() @ Id(a) >> Id(a) @ f
        """
        result = super().tensor(other)
        return Diagram(Ty(*result.dom), Ty(*result.cod),
                       result.boxes, result.offsets, _fast=True)

    def dagger(self):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> f = Box('f', a, a.l @ b.r).dagger()
        >>> assert f.dagger() >> f == (f.dagger() >> f).dagger()
        """
        result = super().dagger()
        return Diagram(Ty(*result.dom), Ty(*result.cod),
                       result.boxes, result.offsets, _fast=True)

    @staticmethod
    def id(x):
        """
        >>> assert Diagram.id(Ty('s')) == Id(Ty('s'))
        >>> print(Diagram.id(Ty('s')))
        Id(s)
        """
        return Id(x)

    @staticmethod
    def cups(x, y):  # pylint: disable=invalid-name
        """ Constructs nested cups witnessing adjointness of x and y

        >>> a, b = Ty('a'), Ty('b')
        >>> Diagram.cups(a @ b @ a, a.r @ b.r) #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        pregroup.AxiomError: a @ b @ a and a.r @ b.r are not adjoints.
        >>> assert Diagram.cups(a, a.r) == Cup(a, a.r)
        >>> assert Diagram.cups(a @ b, (a @ b).l) == (Cup(a, a.l)
        ...                 << Id(a) @ Cup(b, b.l) @ Id(a.l))
        """
        if x.r != y and x != y.r:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        cups = Id(x @ y)
        for i in range(len(x)):
            j = len(x) - i - 1
            cups = cups\
                >> Id(x[:j]) @ Cup(x[j:j + 1], y[i:i + 1]) @ Id(y[i + 1:])
        return cups

    @staticmethod
    def caps(x, y):  # pylint: disable=invalid-name
        """ Constructs nested cups witnessing adjointness of x and y

        >>> a, b = Ty('a'), Ty('b')
        >>> Diagram.caps( a @ b @ a, a.l @ b.l) #doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        pregroup.AxiomError: a @ b @ a and a.l @ b.l are not adjoints.
        >>> assert Diagram.caps(a, a.r) == Cap(a, a.r)
        >>> assert Diagram.caps(a @ b, (a @ b).l) == (Cap(a, a.l)
        ...                 >> Id(a) @ Cap(b, b.l) @ Id(a.l))
        """
        if x.r != y and x != y.r:
            raise AxiomError("{} and {} are not adjoints.".format(x, y))
        caps = Id(x @ y)
        for i in range(len(x)):
            j = len(x) - i - 1
            caps = caps\
                << Id(x[:j]) @ Cap(x[j:j + 1], y[i:i + 1]) @ Id(y[i + 1:])
        return caps

    def transpose_r(self):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> double_snake = Id(a @ b).transpose_r()
        >>> two_snakes = Id(b).transpose_r() @ Id(a).transpose_r()
        >>> double_snake == two_snakes
        False
        >>> two_snakes_nf = moncat.Diagram.normal_form(two_snakes)
        >>> assert double_snake == two_snakes_nf
        """
        return Diagram.caps(self.dom.r, self.dom) @ Id(self.cod.r)\
            >> Id(self.dom.r) @ self @ Id(self.cod.r)\
            >> Id(self.dom.r) @ Diagram.cups(self.cod, self.cod.r)

    def transpose_l(self):
        """
        >>> a, b = Ty('a'), Ty('b')
        >>> double_snake = Id(a @ b).transpose_l()
        >>> two_snakes = Id(b).transpose_l() @ Id(a).transpose_l()
        >>> double_snake == two_snakes
        False
        >>> two_snakes_nf = moncat.Diagram.normal_form(two_snakes, left=True)
        >>> assert double_snake == two_snakes_nf
        """
        return Id(self.cod.l) @ Diagram.caps(self.dom, self.dom.l)\
            >> Id(self.cod.l) @ self @ Id(self.dom.l)\
            >> Diagram.cups(self.cod.l, self.cod) @ Id(self.dom.l)

    def interchange(self, i, j, left=False):
        """
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x.r, y.l)
        >>> d = (f @ f.dagger()).interchange(0, 1)
        >>> assert d == Id(x.r) @ f.dagger() >> f @ Id(x.r)
        >>> print((Cup(x, x.l) >> Cap(x, x.r)).interchange(0, 1))
        Cap(x, x.r) @ Id(x @ x.l) >> Id(x @ x.r) @ Cup(x, x.l)
        >>> print((Cup(x, x.l) >> Cap(x, x.r)).interchange(0, 1, left=True))
        Id(x @ x.l) @ Cap(x, x.r) >> Cup(x, x.l) @ Id(x @ x.r)
        """
        result = super().interchange(i, j, left=left)
        return Diagram(Ty(*result.dom), Ty(*result.cod),
                       result.boxes, result.offsets, _fast=True)

    def normalize(self, left=False):
        """
        Return a generator which yields normalization steps.

        >>> n, s = Ty('n'), Ty('s')
        >>> cup, cap = Cup(n, n.r), Cap(n.r, n)
        >>> f, g, h = Box('f', n, n), Box('g', s @ n, n), Box('h', n, n @ s)
        >>> diagram = g @ cap >> f.dagger() @ Id(n.r) @ f >> cup @ h
        >>> assert next(diagram.normalize()) == diagram
        >>> for d in diagram.normalize(): print(d)  # doctest: +ELLIPSIS
        g >> Id(n) @ Cap(n.r, n) >> ... >> Cup(n, n.r) @ Id(n) >> h
        g >> f.dagger() >> ... >> Cup(n, n.r) @ Id(n) >> h
        g >> f.dagger() >> Id(n) @ Cap(n.r, n) >> Cup(n, n.r) @ Id(n) >> f >> h
        g >> f.dagger() >> f >> h
        """
        def unsnake(diagram, cup, cap):
            """
            Given a diagram and the indices for an adjacent cup and cap pair,
            returns a new diagram with the snake removed.
            """
            if not cup - cap == 1:
                raise ValueError
            if not isinstance(diagram.boxes[cup], Cup):
                raise ValueError
            if not isinstance(diagram.boxes[cap], Cap):
                raise ValueError
            if not diagram.offsets[cap] - diagram.offsets[cup] in [-1, 1]:
                raise ValueError
            return Diagram(diagram.dom, diagram.cod,
                           diagram.boxes[:cap] + diagram.boxes[cup + 1:],
                           diagram.offsets[:cap] + diagram.offsets[cup + 1:],
                           _fast=True)

        def left_unsnake(diagram, cup, cap,
                         left_obstruction, right_obstruction):
            """
            Given a diagram, the indices for a yankable cup and cap, together
            with the lists of box indices obstructing on the left and right,
            returns a new diagram with the snake removed.

            A left snake is one of the form Id @ Cap >> Cup @ Id.
            """
            for left in left_obstruction:
                diagram = diagram.interchange(cap, left)
                yield diagram
                for i, right in enumerate(right_obstruction):
                    if right < left:
                        right_obstruction[i] += 1
                cap += 1
            for right in right_obstruction[::-1]:
                diagram = diagram.interchange(right, cup)
                yield diagram
                cup -= 1
            yield unsnake(diagram, cup, cap)

        def right_unsnake(diagram, cup, cap,
                          left_obstruction, right_obstruction):
            """
            A right snake is one of the form Cap @ Id >> Id @ Cup.
            """
            for left in left_obstruction[::-1]:
                diagram = diagram.interchange(left, cup)
                yield diagram
                for i, right in enumerate(right_obstruction):
                    if right > left:
                        right_obstruction[i] -= 1
                cup -= 1
            for right in right_obstruction:
                diagram = diagram.interchange(cap, right)
                yield diagram
                cap += 1
            yield unsnake(diagram, cup, cap)

        def follow_wire(diagram, i, wire):
            """
            Given a diagram, the index of a box i and the offset of an output
            wire, returns (j, wire, left_obstruction, right_obstruction).
            j is the index of the box which takes this wire as input, or
            len(diagram) if it is connected to the bottom boundary.
            wire is the offset of the wire at its bottom end.
            """
            left_obstruction, right_obstruction = [], []
            for j in range(i + 1, len(diagram)):
                box, off = diagram.boxes[j], diagram.offsets[j]
                if off <= wire < off + len(box.dom):
                    return j, wire, left_obstruction, right_obstruction
                if off <= wire:
                    wire += len(box.cod) - len(box.dom)
                    left_obstruction.append(j)
                else:
                    right_obstruction.append(j)
            return len(diagram), wire, left_obstruction, right_obstruction

        diagram = self
        yield diagram
        while True:
            before = diagram
            for cap in range(len(diagram)):
                if not isinstance(diagram.boxes[cap], Cap):
                    continue
                for left_snake, wire in [(True, diagram.offsets[cap]),
                                         (False, diagram.offsets[cap] + 1)]:
                    cup, wire, left_obstruction, right_obstruction =\
                        follow_wire(diagram, cap, wire)
                    # We found what the cap is connected to,
                    # if it's not yankable we try with the other leg.
                    if cup == len(diagram):
                        continue
                    if not isinstance(diagram.boxes[cup], Cup):
                        continue
                    if left_snake and diagram.offsets[cup] + 1 != wire:
                        continue
                    if not left_snake and diagram.offsets[cup] != wire:
                        continue
                    # We rewrite self and call normal_form recursively
                    # on a smaller diagram (with one snake removed).
                    rewrite = left_unsnake if left_snake else right_unsnake
                    for _diagram in rewrite(
                            diagram, cup, cap,
                            left_obstruction, right_obstruction):
                        yield _diagram
                        diagram = _diagram
                    break
                break
            if diagram == before:
                break
        gen = moncat.Diagram.normalize(diagram, left=left)
        for _diagram in gen:
            if _diagram != diagram:
                yield _diagram

    def normal_form(self, left=False):
        """
        Implements the normalisation of rigid monoidal categories,
        see arxiv:1601.05372, definition 2.12.

        >>> a, b, c = Ty('a'), Ty('b'), Ty('c')
        >>> f = Box('f', a @ b.l, c.r)
        >>> transpose = f.transpose_r().transpose_l().transpose_r()\\
        ...              .transpose_l().transpose_r().transpose_l()
        >>> assert transpose.normal_form() == f

        >>> x = Ty('x')
        >>> unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
        >>> d = Cap(x, x.l) @ unit >> counit @ Cup(x.l, x)
        >>> assert d.normal_form() == unit >> counit
        """
        *_, diagram = self.normalize(left=left)
        return diagram


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
        moncat.Box.__init__(self, name, dom, cod, data=data, _dagger=_dagger)
        Diagram.__init__(self, dom, cod, [self], [0], _fast=True)


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


class Id(Diagram):
    """ Define an identity arrow in a free rigid category

    >>> t = Ty('a', 'b', 'c')
    >>> assert Id(t) == Diagram(t, t, [], [])
    """
    def __init__(self, t):
        """
        >>> Id(Ty('n') @ Ty('s'))
        Id(Ty('n', 's'))
        >>> Id('n')  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Input of type Ty expected, got 'n' instead.
        """
        if not isinstance(t, Ty):
            raise ValueError(
                "Input of type Ty expected, got {} instead.".format(repr(t)))
        super().__init__(t, t, [], [], _fast=True)

    def __repr__(self):
        """
        >>> Id(Ty('n'))
        Id(Ty('n'))
        """
        return "Id({})".format(repr(self.dom))

    def __str__(self):
        """
        >>> n = Ty('n')
        >>> print(Id(n))
        Id(n)
        """
        return "Id({})".format(str(self.dom))


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
    ...              >> Id(a.r) @ love_box @ Id(a.l)}
    >>> F = RigidFunctor(ob, ar)
    >>> assert F(Cap(n.r, n)) == Cap(Ty(Ob('a', z=1)), Ty('a'))
    >>> assert F(Cup(a, a.l)) == Diagram.cups(n @ n, (n @ n).l)
    """
    def __call__(self, diagram):
        if isinstance(diagram, Ob):
            result = self.ob[Ty(diagram.name)]
            if diagram.z < 0:
                for _ in range(-diagram.z):
                    result = result.l
            elif diagram.z > 0:
                for _ in range(diagram.z):
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
        raise ValueError("Expected pregroup.Diagram, got {} of type {} instead"
                         .format(repr(diagram), type(diagram)))
