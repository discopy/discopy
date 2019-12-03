# -*- coding: utf-8 -*-

"""
Implements free monoidal categories and (dagger) monoidal functors.

We can check the axioms for dagger monoidal categories, up to interchanger.

>>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
>>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
>>> d = Id(x) @ f1 >> f0 @ Id(w)
>>> assert d == (f0 @ f1).interchange(0, 1)
>>> assert f0 @ f1 == d.interchange(0, 1)
>>> assert (f0 @ f1).dagger().dagger() == f0 @ f1
>>> assert (f0 @ f1).dagger().interchange(0, 1) == f0.dagger() @ f1.dagger()

We can check the Eckerman-Hilton argument, up to interchanger.

>>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
>>> assert s0 @ s1 == s0 >> s1 == (s1 @ s0).interchange(0, 1)
>>> assert s1 @ s0 == s1 >> s0 == (s0 @ s1).interchange(0, 1)
"""

from discopy import cat, config
from discopy.cat import Ob, Arrow, Gen, Functor


class Ty(list):
    """
    Implements a type as a list of objects, used as dom and cod of diagrams.
    Types are the free monoid on objects with product @ and unit Ty().

    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> assert x @ y != y @ x
    >>> assert x @ Ty() == x == Ty() @ x
    >>> assert (x @ y) @ z == x @ y @ z == x @ (y @ z)
    """
    def __init__(self, *t):
        """
        >>> t = Ty('x', 'y', 'z')
        >>> list(t)
        [Ob('x'), Ob('y'), Ob('z')]
        """
        super().__init__(x if isinstance(x, Ob) else Ob(x) for x in t)

    def __add__(self, other):
        """
        >>> sum([Ty('x'), Ty('y'), Ty('z')], Ty())
        Ty('x', 'y', 'z')
        """
        return Ty(*(super().__add__(other)))

    def __matmul__(self, other):
        """
        >>> Ty('x') @ Ty('y')
        Ty('x', 'y')
        """
        return self + other

    def __getitem__(self, key):
        """
        >>> t = Ty('x', 'y', 'z')
        >>> t[0]
        Ob('x')
        >>> t[:1]
        Ty('x')
        >>> t[1:]
        Ty('y', 'z')
        """
        if isinstance(key, slice):
            return Ty(*super().__getitem__(key))
        return super().__getitem__(key)

    def __repr__(self):
        """
        >>> Ty('x', 'y')
        Ty('x', 'y')
        """
        return "Ty({})".format(', '.join(repr(x.name) for x in self))

    def __str__(self):
        """
        >>> print(Ty('x', 'y'))
        x @ y
        """
        return ' @ '.join(map(str, self)) or 'Ty()'

    def __hash__(self):
        """
        >>> {Ty('x', 'y', 'z'): 42}[Ty('x', 'y', 'z')]
        42
        """
        return hash(repr(self))


class Diagram(Arrow):
    """ Implements a diagram with dom, cod, a list of boxes and offsets.

    >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    >>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
    >>> d = Diagram(x @ z, y @ w, [f0, f1], [0, 1])
    >>> assert d == f0 @ f1
    """
    def __init__(self, dom, cod, boxes, offsets):
        """
        >>> Diagram(Ty('x'), Ty('y'), [Box('f', Ty('x'), Ty('y'))], [0])
        ... # doctest: +ELLIPSIS
        Diagram(dom=Ty('x'), cod=Ty('y'), boxes=[Box(...)], offsets=[0])
        >>> Diagram('x', Ty('x'), [], [])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Domain of type Ty expected, got 'x' ... instead.
        >>> Diagram(Ty('x'), 'x', [], [])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Codomain of type Ty expected, got 'x' ... instead.
        >>> Diagram(Ty('x'), Ty('x'), [], [1])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Boxes and offsets must have the same length.
        >>> Diagram(Ty('x'), Ty('x'), [1], [1])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Box of type Diagram expected, got 1 ... instead.
        >>> Diagram(Ty('x'), Ty('x'), [Box('f', Ty('x'), Ty('y'))], [Ty('x')])
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Offset of type int expected, got Ty('x') ... instead.
        """
        if not isinstance(dom, Ty):
            raise ValueError("Domain of type Ty expected, got {} of type {} "
                             "instead.".format(repr(dom), type(dom)))
        if not isinstance(cod, Ty):
            raise ValueError("Codomain of type Ty expected, got {} of type {} "
                             "instead.".format(repr(cod), type(cod)))
        if len(boxes) != len(offsets):
            raise ValueError("Boxes and offsets must have the same length.")
        if not config.fast:
            scan = dom
            for f, n in zip(boxes, offsets):
                if not isinstance(f, Diagram):
                    raise ValueError(
                        "Box of type Diagram expected, got {} of type {} "
                        "instead.".format(repr(f), type(f)))
                if not isinstance(n, int):
                    raise ValueError(
                        "Offset of type int expected, got {} of type {} "
                        "instead.".format(repr(n), type(n)))
                if scan[n: n + len(f.dom)] != f.dom:
                    raise AxiomError(
                        "Domain {} expected, got {} instead."
                        .format(scan[n: n + len(f.dom)], f.dom))
                scan = scan[: n] + f.cod + scan[n + len(f.dom):]
            if scan != cod:
                raise AxiomError(
                    "Codomain {} expected, got {} instead.".format(cod, scan))
        self._dom, self._cod = dom, cod
        self._boxes, self._offsets = boxes, offsets
        list.__init__(self, zip(boxes, offsets))

    @property
    def boxes(self):
        """
        >>> Diagram(Ty('x'), Ty('x'), [], []).boxes
        []
        """
        return self._boxes

    @property
    def offsets(self):
        """
        >>> Diagram(Ty('x'), Ty('x'), [], []).offsets
        []
        """
        return self._offsets

    def __eq__(self, other):
        """
        >>> Diagram(Ty('x'), Ty('x'), [], []) == Ty('x')
        False
        >>> Diagram(Ty('x'), Ty('x'), [], []) == Id(Ty('x'))
        True
        """
        if not isinstance(other, Diagram):
            return False
        return all(self.__getattribute__(attr) == other.__getattribute__(attr)
                   for attr in ['dom', 'cod', 'boxes', 'offsets'])

    def __repr__(self):
        """
        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> Diagram(x, x, [], [])
        Id(Ty('x'))
        >>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
        >>> Diagram(x, y, [f0], [0])  # doctest: +ELLIPSIS
        Diagram(dom=Ty('x'), cod=Ty('y'), boxes=[Box(...)], offsets=[0])
        >>> Diagram(x @ z, y @ w, [f0, f1], [0, 1])  # doctest: +ELLIPSIS
        Diagram(dom=Ty('x', 'z'), cod=Ty('y', 'w'), boxes=..., offsets=[0, 1])
        """
        if not self.boxes:  # i.e. self is identity.
            return repr(Id(self.dom))
        return "Diagram(dom={}, cod={}, boxes={}, offsets={})".format(
            repr(self.dom), repr(self.cod),
            repr(self.boxes), repr(self.offsets))

    def __str__(self):
        """
        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> print(Diagram(x, x, [], []))
        Id(x)
        >>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
        >>> print(Diagram(x, y, [f0], [0]))
        f0
        >>> print(f0 @ f1)
        f0 @ Id(z) >> Id(y) @ f1
        >>> print(f0 @ Id(z) >> Id(y) @ f1)
        f0 @ Id(z) >> Id(y) @ f1
        """
        if not self.boxes:  # i.e. self is identity.
            return str(self.id(self.dom))

        def line(scan, box, off):
            left = "{} @ ".format(self.id(scan[:off])) if scan[:off] else ""
            right = " @ {}".format(self.id(scan[off + len(box.dom):]))\
                if scan[off + len(box.dom):] else ""
            return left + str(box) + right
        box, off = self.boxes[0], self.offsets[0]
        result = line(self.dom, box, off)
        scan = self.dom[:off] + box.cod + self.dom[off + len(box.dom):]
        for box, off in zip(self.boxes[1:], self.offsets[1:]):
            result = "{} >> {}".format(result, line(scan, box, off))
            scan = scan[:off] + box.cod + scan[off + len(box.dom):]
        return result

    def tensor(self, other):
        """
        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
        >>> assert f0.tensor(f1) == f0.tensor(Id(z)) >> Id(y).tensor(f1)
        """
        if not isinstance(other, Diagram):
            raise ValueError("Expected Diagram, got {} of type {} instead."
                             .format(repr(other), type(other)))
        dom, cod = self.dom + other.dom, self.cod + other.cod
        boxes = self.boxes + other.boxes
        offsets = self.offsets + [n + len(self.cod) for n in other.offsets]
        return Diagram(dom, cod, boxes, offsets)

    def __matmul__(self, other):
        """
        >>> Id(Ty('x')) @ Id(Ty('y'))
        Id(Ty('x', 'y'))
        >>> assert Id(Ty('x')) @ Id(Ty('y')) == Id(Ty('x')).tensor(Id(Ty('y')))
        """
        return self.tensor(other)

    def then(self, other):
        """
        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
        >>> print(Id(x) @ f1 >> f0 @ Id(w))
        Id(x) @ f1 >> f0 @ Id(w)
        """
        if not isinstance(other, Diagram):
            raise ValueError("Expected Diagram, got {} of type {} instead."
                             .format(repr(other), type(other)))
        if self.cod != other.dom:
            raise AxiomError("{} does not compose with {}."
                             .format(repr(self), repr(other)))
        dom, cod = self.dom, other.cod
        boxes = self.boxes + other.boxes
        offsets = self.offsets + other.offsets
        return Diagram(dom, cod, boxes, offsets)

    def dagger(self):
        """
        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> d = Box('f0', x, y) @ Box('f1', z, w)
        >>> print(d.dagger())
        Id(y) @ f1.dagger() >> f0.dagger() @ Id(z)
        """
        return Diagram(self.cod, self.dom,
                       [f.dagger() for f in self.boxes[::-1]],
                       self.offsets[::-1])

    @staticmethod
    def id(x):
        """
        >>> assert Diagram.id(Ty('x')) == Diagram(Ty('x'), Ty('x'), [], [])
        """
        return Id(x)

    def interchange(self, k0, k1):
        """
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x, y)
        >>> d = f @ f.dagger()
        >>> print(d.interchange(0, 0))
        f @ Id(y) >> Id(y) @ f.dagger()
        >>> print(d.interchange(0, 1))
        Id(x) @ f.dagger() >> f @ Id(x)
        >>> print((d >> d.dagger()).interchange(0, 2))
        Id(x) @ f.dagger() >> Id(x) @ f >> f @ Id(y) >> f.dagger() @ Id(y)
        """
        if k0 == k1:
            return self
        elif k1 < k0:
            return self.interchange(k1, k0)
        elif k1 > k0 + 1:
            result = self
            for i in range(k1 - k0):
                result = result.interchange(k0 + i, k0 + i + 1)
            return result
        box0, box1 = self.boxes[k0], self.boxes[k1]
        off0, off1 = self.offsets[k0], self.offsets[k1]
        if off1 >= off0 + len(box0.cod):  # box0 left of box1
            off1 = off1 - len(box0.cod) + len(box0.dom)
        elif off0 >= off1 + len(box1.dom):  # box0 right of box1
            off0 = off0 - len(box1.dom) + len(box1.cod)
        else:
            raise InterchangerError("Boxes {} and {} are connected."
                                    .format(repr(box0), repr(box1)))
        return Diagram(
            self.dom, self.cod,
            self.boxes[:k0] + [box1, box0] + self.boxes[k0 + 2:],
            self.offsets[:k0] + [off1, off0] + self.offsets[k0 + 2:])

    def normal_form(self):
        """
        >>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
        >>> (s1 @ s0).normal_form()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        NotImplementedError: Diagram(...) is not connected.
        >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
        >>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
        >>> d = Id(x) @ f1 >> f0 @ Id(w)
        >>> assert d.normal_form() == f0 @ f1
        """
        d, cache = self, [self]
        while True:
            for i in range(len(d)):
                try:
                    off0, off1 = self.offsets[i], self.offsets[i + 1]
                    left, right = (off0, off1)\
                        if off1 >= off0 + len(self.boxes[i].cod)\
                        else (off1, off0)
                    d = d.interchange(right, left)
                    if d in cache:
                        raise NotImplementedError(
                            "{} is not connected.".format(repr(self)))
                    cache.append(d)
                    break
                except InterchangerError:
                    pass
            break
        return d


class AxiomError(cat.AxiomError):
    """
    >>> Diagram(Ty('x'), Ty('x'), [Box('f', Ty('x'), Ty('y'))], [0])
    ... # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    moncat.AxiomError: Codomain x expected, got y instead.
    >>> Diagram(Ty('y'), Ty('y'), [Box('f', Ty('x'), Ty('y'))], [0])
    ... # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    moncat.AxiomError: Domain y expected, got x instead.
    """
    pass


class InterchangerError(AxiomError):
    """
    >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
    >>> d = Box('f', x, y) >> Box('g', y, z)
    >>> d.interchange(0, 1)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    moncat.InterchangerError: Boxes ... are connected.
    """
    pass


class Id(Diagram):
    """ Implements the identity diagram of a given type.

    >>> s, t = Ty('x', 'y'), Ty('z', 'w')
    >>> f = Box('f', s, t)
    >>> assert f >> Id(t) == f == Id(s) >> f
    """
    def __init__(self, x):
        """
        >>> assert Id(Ty('x')) == Diagram.id(Ty('x'))
        """
        super().__init__(x, x, [], [])

    def __repr__(self):
        """
        >>> Id(Ty('x'))
        Id(Ty('x'))
        """
        return "Id({})".format(repr(self.dom))

    def __str__(self):
        """
        >>> print(Id(Ty('x')))
        Id(x)
        """
        return "Id({})".format(str(self.dom))


class Box(Gen, Diagram):
    """ Implements a box as a diagram with a name and itself as box.

    Note that as for composition, when we tensor an empty diagram with a box,
    we get a diagram that is defined as equal to the original box.

    >>> f = Box('f', Ty('x', 'y'), Ty('z'))
    >>> Id(Ty('x', 'y')) >> f  # doctest: +ELLIPSIS
    Diagram(dom=Ty('x', 'y'), cod=Ty('z'), boxes=[Box(...)], offsets=[0])
    >>> Id(Ty()) @ f  # doctest: +ELLIPSIS
    Diagram(dom=Ty('x', 'y'), cod=Ty('z'), boxes=[Box(...)], offsets=[0])
    >>> assert Id(Ty('x', 'y')) >> f == f == f >> Id(Ty('z'))
    >>> assert Id(Ty()) @ f == f == f @ Id(Ty())
    >>> assert f == f.dagger().dagger()
    """
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        """
        >>> f = Box('f', Ty('x', 'y'), Ty('z'), data=42)
        >>> print(f)
        f
        >>> f.name, f.dom, f.cod, f.data
        ('f', Ty('x', 'y'), Ty('z'), 42)
        """
        self._dom, self._cod = dom, cod
        self._boxes, self._offsets = [self], [0]
        self._name, self._dagger, self._data = name, _dagger, data
        Diagram.__init__(self, dom, cod, [self], [0])

    def dagger(self):
        """
        >>> f = Box('f', Ty('x', 'y'), Ty('z'))
        >>> print(f.dagger())
        f.dagger()
        """
        return Box(self.name, self.cod, self.dom,
                   _dagger=not self._dagger, data=self.data)

    def __repr__(self):
        """
        >>> Box('f', Ty('x', 'y'), Ty('z'))
        Box('f', Ty('x', 'y'), Ty('z'))
        >>> Box('f', Ty('x', 'y'), Ty('z')).dagger()
        Box('f', Ty('x', 'y'), Ty('z')).dagger()
        """
        if self._dagger:
            return repr(self.dagger()) + ".dagger()"
        return "Box({}, {}, {}{})".format(
            *map(repr, [self.name, self.dom, self.cod]),
            ", data=" + repr(self.data) if self.data else '')

    def __hash__(self):
        """
        >>> f = Box('f', Ty('x', 'y'), Ty('z'), data=42)
        >>> {f: 42}[f]
        42
        """
        return hash(repr(self))

    def __eq__(self, other):
        """
        >>> f = Box('f', Ty('x', 'y'), Ty('z'), data=42)
        >>> assert f == Diagram(Ty('x', 'y'), Ty('z'), [f], [0])
        """
        if isinstance(other, Box):
            return repr(self) == repr(other)
        elif isinstance(other, Diagram):
            return len(other) == 1 and other.boxes[0] == self
        return False


class MonoidalFunctor(Functor):
    """ Implements a monoidal functor given its image on objects and arrows.

    >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    >>> f0, f1 = Box('f0', x, y, data=[0.1]), Box('f1', z, w, data=[1.1])
    >>> F = MonoidalFunctor({x: z, y: w, z: x, w: y}, {f0: f1, f1: f0})
    >>> assert F(f0) == f1 and F(f1) == f0
    >>> assert F(F(f0)) == f0
    >>> assert F(f0 @ f1) == f1 @ f0
    >>> assert F(f0 >> f0.dagger()) == f1 >> f1.dagger()
    """
    def __init__(self, ob, ar, ob_cls=Ty, ar_cls=Diagram):
        """
        >>> F = MonoidalFunctor({Ty('x'): Ty('y')}, {})
        >>> F(Id(Ty('x')))
        Id(Ty('y'))
        """
        for x in ob.keys():
            if not isinstance(x, Ty) or len(x) != 1:
                raise ValueError(
                    "Expected an atomic type, got {} instead.".format(repr(x)))
        self._ob, self._ar = ob, ar
        self.ob_cls, self.ar_cls = ob_cls, ar_cls

    def __repr__(self):
        """
        >>> MonoidalFunctor({Ty('x'): Ty('y')}, {})
        MonoidalFunctor(ob={Ty('x'): Ty('y')}, ar={})
        """
        return "MonoidalFunctor(ob={}, ar={})".format(self.ob, self.ar)

    def __call__(self, d):
        """
        >>> x, y = Ty('x'), Ty('y')
        >>> f = Box('f', x, y)
        >>> F = MonoidalFunctor({x: y, y: x}, {f: f.dagger()})
        >>> print(F(x))
        y
        >>> print(F(f))
        f.dagger()
        >>> print(F(F(f)))
        f
        >>> print(F(f >> f.dagger()))
        f.dagger() >> f
        >>> print(F(f @ f.dagger()))
        f.dagger() @ Id(x) >> Id(x) @ f
        """
        if isinstance(d, Ty):
            return sum([self.ob[self.ob_cls(x)] for x in d], self.ob_cls())
        elif isinstance(d, Box):
            return self.ar[d.dagger()].dagger() if d._dagger else self.ar[d]
        elif isinstance(d, Diagram):
            scan, result = d.dom, Id(self(d.dom))
            for f, n in zip(d.boxes, d.offsets):
                id_l = self.ar_cls.id(self(scan[:n]))
                id_r = self.ar_cls.id(self(scan[n + len(f.dom):]))
                result = result >> id_l @ self(f) @ id_r
                scan = scan[:n] + f.cod + scan[n + len(f.dom):]
            return result
        else:
            raise ValueError("Diagram expected, got {} of type {} "
                             "instead.".format(repr(d), type(d)))
