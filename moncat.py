"""
Implements free monoidal categories and (dagger) monoidal functors.

We can check the Eckerman-Hilton argument, up to interchanger.

>>> s0, s1 = Box('s0', Ty(), Ty()), Box('s1', Ty(), Ty())
>>> assert s0 @ s1 == s0 >> s1 == (s1 @ s0).interchange(0, 1)
>>> assert s1 @ s0 == s1 >> s0 == (s0 @ s1).interchange(0, 1)

We can check the axioms for dagger monoidal categories, up to interchanger.

>>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
>>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
>>> d = Id(x) @ f1 >> f0 @ Id(w)
>>> assert d == (f0 @ f1).interchange(0, 1)
>>> assert f0 @ f1 == d.interchange(0, 1)
>>> assert (f0 @ f1).dagger().dagger() == f0 @ f1
>>> assert (f0 @ f1).dagger().interchange(0, 1) == f0.dagger() @ f1.dagger()
"""

from discopy.cat import (
    _config, Ob, Arrow, Gen, Functor, Quiver, AxiomError)


class Ty(list):
    """ Implements a type as a list of objects, used as dom and cod of diagrams.
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
        >>> Diagram(Ty('x'), Ty('x'), [Box('f', Ty('x'), Ty('y'))], [0])
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        discopy.cat.AxiomError: Codomain x expected, got y instead.
        >>> Diagram(Ty('y'), Ty('y'), [Box('f', Ty('x'), Ty('y'))], [0])
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        discopy.cat.AxiomError: Domain y expected, got x instead.
        """
        if not isinstance(dom, Ty):
            raise ValueError("Domain of type Ty expected, got {} of type {} "
                             "instead.".format(repr(dom), type(dom)))
        if not isinstance(cod, Ty):
            raise ValueError("Codomain of type Ty expected, got {} of type {} "
                             "instead.".format(repr(cod), type(cod)))
        if len(boxes) != len(offsets):
            raise ValueError("Boxes and offsets must have the same length.")
        self._dom, self._cod = dom, cod
        self._boxes, self._offsets = boxes, offsets
        list.__init__(self, zip(boxes, offsets))
        if not _config.fast:
            scan = dom
            for f, n in zip(boxes, offsets):
                if not isinstance(f, Diagram):
                    raise ValueError(
                        "Box of type Diagram expected, got {} of type {} "
                        "instead.".format(repr(f), type(f)))
                if not f.boxes:
                    raise ValueError(
                        "The identity diagram {} cannot be used as a box."
                        .format(repr(f)))
                if not isinstance(n, int):
                    raise ValueError(
                        "Offset of type int expected, got {} of type {} "
                        "instead.".format(repr(n), type(n)))
                if scan[n : n + len(f.dom)] != f.dom:
                    raise AxiomError(
                        "Domain {} expected, got {} instead."
                        .format(scan[n : n + len(f.dom)], f.dom))
                scan = scan[: n] + f.cod + scan[n + len(f.dom) :]
            if scan != cod:
                raise AxiomError(
                    "Codomain {} expected, got {} instead.".format(cod, scan))

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
        Diagram(dom=Ty('x', 'z'), cod=Ty('y', 'w'), boxes=[...], offsets=[0, 1])
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
        return Diagram(self.cod, self.dom,
            [f.dagger() for f in self.boxes[::-1]], self.offsets[::-1])

    @staticmethod
    def id(x):
        return Id(x)

    def interchange(self, k0, k1):
        if k0 + 1 != k1:
            raise NotImplementedError
        box0, box1 = self.boxes[k0], self.boxes[k1]
        off0, off1 = self.offsets[k0], self.offsets[k1]
        if off1 >= off0 + len(box0.cod):  # box0 left of box1
            off1 = off1 - len(box0.cod) + len(box0.dom)
        elif off0 >= off1 + len(box1.dom):  # box1 left of box0
            off0 = off0 - len(box1.dom) + len(box1.cod)
        else:
            raise InterchangerError("Boxes ({}, {}) are connected."
                                    .format(box0, box1))
        return Diagram(self.dom, self.cod,
                       self.boxes[:k0] + [box1, box0] + self.boxes[k0 + 2:],
                       self.offsets[:k0] + [off1, off0] + self.offsets[k0 + 2:])

class InterchangerError:
    pass

class Id(Diagram):
    """ Implements the identity diagram of a given type.

    >>> assert Id(Ty('x')) == Diagram(Ty('x'), Ty('x'), [], [])
    """
    def __init__(self, x):
        super().__init__(x, x, [], [])

    def __repr__(self):
        return "Id({})".format(repr(self.dom))

    def __str__(self):
        return "Id({})".format(str(self.dom))

class Box(Gen, Diagram):
    """ Implements a box as a diagram with a name and itself as box.

    Note that as for composition, when we tensor an empty diagram with a box,
    we get a diagram that is defined as equal to the original box.

    >>> f = Box('f', Ty('x', 'y'), Ty('z'))
    >>> f
    Box(name='f', dom=Ty('x', 'y'), cod=Ty('z'))
    >>> print(f)
    f
    >>> Id(Ty('x', 'y')) >> f  # doctest: +ELLIPSIS
    Diagram(dom=Ty('x', 'y'), cod=Ty('z'), boxes=[Box(name='f'...], offsets=[0])
    >>> assert Id(Ty('x', 'y')) >> f == f == f >> Id(Ty('z'))
    >>> Id(Ty()) @ f  # doctest: +ELLIPSIS
    Diagram(dom=Ty('x', 'y'), cod=Ty('z'), boxes=[Box(name='f'...], offsets=[0])
    >>> assert Id(Ty()) @ f == f == f @ Id(Ty())
    >>> f.dagger()
    Box(name='f', dom=Ty('x', 'y'), cod=Ty('z')).dagger()
    >>> print(f.dagger())
    f.dagger()
    >>> assert f == f.dagger().dagger()
    """
    def __init__(self, name, dom, cod, dagger=False, data=None):
        self._dom, self._cod, self._boxes, self._offsets = dom, cod, [self], [0]
        self._name, self._dagger, self._data = name, dagger, data
        Diagram.__init__(self, dom, cod, [self], [0])

    def dagger(self):
        return Box(self.name, self.cod, self.dom,
                   dagger=not self._dagger, data=self.data)

    def __repr__(self):
        if self._dagger:
            return "Box(name={}, dom={}, cod={}{}).dagger()".format(
                *map(repr, [self.name, self.cod, self.dom]),
                ", data=" + repr(self.data) if self.data else '')
        return "Box(name={}, dom={}, cod={}{})".format(
            *map(repr, [self.name, self.dom, self.cod]),
            ", data=" + repr(self.data) if self.data else '')

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if isinstance(other, Box):
            return repr(self) == repr(other)
        elif isinstance(other, Diagram):
            return len(other) == 1 and other.boxes[0] == self

class MonoidalFunctor(Functor):
    """ Implements a monoidal functor given its image on objects and arrows.

    >>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
    >>> f0, f1 = Box('f0', x, y, data=[0.1]), Box('f1', z, w, data=[1.1])
    >>> ob = {x: z, y: w, z: x, w: y}
    >>> ar = Quiver(lambda f: f1 if f == f0 else f0 if f == f1 else None)
    >>> F = MonoidalFunctor(ob, ar)
    >>> assert F(f0) == f1 and F(f1) == f0
    >>> assert F(F(f0)) == f0
    >>> F(f0)
    Box(name='f1', dom=Ty('z'), cod=Ty('w'), data=[1.1])
    >>> assert F(f0 @ f1) == f1 @ f0
    >>> assert F(f0 >> f0.dagger()) == f1 >> f1.dagger()
    """
    def __init__(self, ob, ar):
        for x in ob.keys():
            if not isinstance(x, Ty) or len(x) != 1:
                raise ValueError(
                    "Expected an atomic type, got {} instead.".format(repr(x)))
        self._objects, self._arrows = ob, ar
        self._ob, self._ar = {x[0]: y for x, y in ob.items()}, ar

    def __repr__(self):
        return "MonoidalFunctor(ob={}, ar={})".format(
                                self._objects, self._arrows)

    def __call__(self, d):
        if isinstance(d, Ty):
            return sum([self.ob[x] for x in d], Ty())
        elif isinstance(d, Box):
            return self.ar[d.dagger()].dagger() if d._dagger else self.ar[d]
        scan, result = d.dom, Id(self(d.dom))
        for f, n in d:
            result = result >> Id(self(scan[:n])) @ self(f)\
                             @ Id(self(scan[n + len(f.dom):]))
            scan = scan[:n] + f.cod + scan[n + len(f.dom):]
        return result
