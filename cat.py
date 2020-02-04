# -*- coding: utf-8 -*-

"""
Implements free dagger categories and functors.

We can create boxes with objects as domain and codomain:

>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)

We can create arbitrary arrows with composition:

>>> arrow = Arrow(x, x, [f, g, h])
>>> assert arrow == f >> g >> h == h << g << f

We can create dagger functors from the free category to itself:

>>> ob = {x: z, y: y, z: x}
>>> ar = {f: g[::-1], g: f[::-1], h: h[::-1]}
>>> F = Functor(ob, ar)
>>> assert F(arrow) == (h >> f >> g)[::-1]
"""

from functools import reduce as fold
from discopy import messages


class Ob:
    """
    Defines an object in a free category, only distinguished by its name.

    Parameters
    ----------
    name : any
        Name of the object

    Note
    ----
    When printing an object, we only print its name.

    >>> x = Ob('x')
    >>> print(x)
    x

    Objects are equal only to objects with equal names.

    >>> x = Ob('x')
    >>> assert x == Ob('x') and x != 'x' and x != Ob('y')

    Objects are hashable whenever their name is.

    >>> d = {Ob(['x', 'y']): 42}
    Traceback (most recent call last):
    ...
    TypeError: unhashable type: 'list'

    """
    @property
    def name(self):
        """
        The name of an object is immutable, it cannot be empty.

        >>> x = Ob('x')
        >>> x.name
        'x'
        >>> x.name = 'y'  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        """
        return self._name

    def __init__(self, name):
        if not str(name):
            raise ValueError(messages.empty_name(name))
        self._name = name

    def __repr__(self):
        return "Ob({})".format(repr(self.name))

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        if not isinstance(other, Ob):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Arrow:
    """
    Defines an arrow in a free dagger category.

    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
    >>> arrow = Arrow(x, x, [f, g, h])
    >>> print(arrow[::-1])
    h[::-1] >> g[::-1] >> f[::-1]

    Parameters
    ----------
    dom : cat.Ob
        Domain of the arrow.
    cod : cat.Ob
        Codomain of the arrow.
    boxes : list of :class:`Arrow`
        Boxes of the arrow.

    Raises
    ------
    :class:`cat.AxiomError`
        Whenever the boxes do not compose.

    """
    def __init__(self, dom, cod, boxes, _scan=True):
        """
        >>> from discopy.moncat import spiral
        >>> arrow = spiral(3)
        """
        if not isinstance(dom, Ob):
            raise TypeError(messages.type_err(Ob, dom))
        if not isinstance(cod, Ob):
            raise TypeError(messages.type_err(Ob, cod))
        if _scan:
            scan = dom
            for depth, box in enumerate(boxes):
                if not isinstance(box, Arrow):
                    raise TypeError(messages.type_err(Arrow, box))
                if box.dom != scan:
                    raise AxiomError(messages.does_not_compose(
                        boxes[depth - 1] if depth else Id(dom), box))
                scan = box.cod
            if scan != cod:
                raise AxiomError(messages.does_not_compose(
                    boxes[-1] if boxes else Id(dom), Id(cod)))
        self._dom, self._cod, self._boxes = dom, cod, boxes

    @property
    def dom(self):
        """
        The domain of an arrow is immutable.

        >>> arrow = Arrow(Ob('x'), Ob('x'), [])
        >>> assert arrow.dom == Ob('x')
        >>> arrow.dom = Ob('y')  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        """
        return self._dom

    @property
    def cod(self):
        """
        The codomain of an arrow is immutable.

        >>> arrow = Arrow(Ob('x'), Ob('x'), [])
        >>> assert arrow.cod == Ob('x')
        >>> arrow.cod = Ob('y')  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        """
        return self._cod

    @property
    def boxes(self):
        """
        The list of boxes in an arrow is immutable. Use composition instead.

        >>> f = Box('f', Ob('x'), Ob('y'))
        >>> arrow = Arrow(Ob('x'), Ob('x'), [])
        >>> arrow.boxes.append(f)  # This does nothing.
        >>> assert f not in arrow.boxes
        """
        return list(self._boxes)

    def __iter__(self):
        for box in self.boxes:
            yield box

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step == -1:
                boxes = [box[::-1] for box in self.boxes[key]]
                return Arrow(self.cod, self.dom, boxes, _scan=False)
            if (key.step or 1) != 1:
                raise IndexError
            boxes = self.boxes[key]
            if not boxes:
                if (key.start or 0) >= len(self):
                    return Id(self.cod)
                if (key.start or 0) <= -len(self):
                    return Id(self.dom)
                return Id(self.boxes[key.start or 0].dom)
            return Arrow(boxes[0].dom, boxes[-1].cod, boxes, _scan=False)
        return self.boxes[key]

    def __len__(self):
        return len(self.boxes)

    def __repr__(self):
        if not self.boxes:  # i.e. self is identity.
            return repr(Id(self.dom))
        if len(self.boxes) == 1:  # i.e. self is a box.
            return repr(self.boxes[0])
        return "cat.Arrow(dom={}, cod={}, boxes={})".format(
            repr(self.dom), repr(self.cod), repr(self.boxes))

    def __str__(self):
        return ' >> '.join(map(str, self)) or str(self.id(self.dom))

    def __eq__(self, other):
        if not isinstance(other, Arrow):
            return False
        return self.dom == other.dom and self.cod == other.cod\
            and all(x == y for x, y in zip(self.boxes, other.boxes))

    def __hash__(self):
        return hash(repr(self))

    def __rmul__(self, n_times):
        """
        >>> x, y = Ob('x'), Ob('y')
        >>> f = Box('f', x, y)
        >>> print(3 * (f >> f.dagger()))
        f >> f[::-1] >> f >> f[::-1] >> f >> f[::-1]
        """
        return self.id(self.dom).compose(*(n_times * (self, )))

    def then(self, other):
        """
        Returns the composition of `self` with an arrow `other`.

        This method is called using the binary operators `>>` and `<<`:

        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
        >>> assert f.then(g) == f >> g == g << f

        Parameters
        ----------
        other : cat.Arrow
            such that `self.cod == other.dom`.

        Returns
        -------
        arrow : cat.Arrow
            such that :code:`arrow.boxes == self.boxes + other.boxes`.

        Raises
        ------
        :class:`cat.AxiomError`
            whenever `self` and `other` do not compose.

        Notes
        -----

        We can check the axioms of categories
        (i.e. composition is unital and associative):

        >>> assert f >> Id(y) == f == Id(x) >> f
        >>> assert (f >> g) >> h == f >> (g >> h)
        """
        if not isinstance(other, Arrow):
            raise TypeError(messages.type_err(Arrow, other))
        if self.cod != other.dom:
            raise AxiomError(messages.does_not_compose(self, other))
        boxes = self.boxes + other.boxes
        return Arrow(self.dom, other.cod, boxes, _scan=False)

    def __rshift__(self, other):
        return self.then(other)

    def __lshift__(self, other):
        return other.then(self)

    def compose(self, *others, backwards=False):
        """
        Returns the composition of self with a list of other arrows.

        Parameters
        ----------
        others : list
            Other arrows.
        backwards : bool, optional
            Whether to compose in reverse, default is :code`False`.

        Returns
        -------
        arrow : cat.Arrow
            Such that :code:`arrow == self >> others[0] >> ... >> others[-1]`
            if :code:`backwards` else
            :code:`arrow == self << others[0] << ... << others[-1]`.

        Examples
        --------
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
        >>> assert Arrow.compose(f, g, h) == f >> g >> h
        >>> assert f.compose(g, h) == Id(x).compose(f, g, h) == f >> g >> h
        >>> assert h.compose(g, f, backwards=True) == h << g << f
        """
        return fold(lambda f, g: f << g if backwards else f >> g, others, self)

    def dagger(self):
        """
        Returns the dagger of `self`, this method is called using the unary
        operator :code:`[::-1]`, i.e. :code:`self[::-1] == self.dagger()`.

        Returns
        -------
        arrow : cat.Arrow
            Such that
            :code:`arrow.boxes == [box[::-1] for box in self[::-1]]`.

        Notes
        -----
        We can check the axioms of dagger (i.e. a contravariant involutive
        identity-on-objects endofunctor):

        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> assert f[::-1][::-1] == f
        >>> assert Id(x)[::-1] == Id(x)
        >>> assert (f >> g)[::-1] == g[::-1] >> f[::-1]
        """
        return self[::-1]

    @staticmethod
    def id(x):
        """
        Returns the identity arrow on x.

        >>> x = Ob('x')
        >>> assert Arrow.id(x) == Id(x) == Arrow(x, x, [])

        Parameters
        ----------
        x : cat.Ob
            Any object.

        Returns
        -------
        cat.Id
        """
        return Id(x)


class Id(Arrow):
    """
    Defines the identity arrow on x, i.e. with an empty list of boxes.

    >>> x = Ob('x')
    >>> assert Id(x) == Arrow(x, x, [])

    Parameters
    ----------
        x : cat.Ob
            Any object.

    See also
    --------
        cat.Arrow.id
    """
    def __init__(self, x):
        super().__init__(x, x, [], _scan=False)

    def __repr__(self):
        return "Id({})".format(repr(self.dom))

    def __str__(self):
        return "Id({})".format(str(self.dom))


class AxiomError(Exception):
    """
    This is raised whenever we try to build an invalid arrow.
    """


class Box(Arrow):
    """ Defines a box as an arrow with the list of only itself as boxes.

    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y, data=[42])
    >>> assert f == Arrow(x, y, [f])
    >>> assert f.boxes == [f]
    >>> assert f[:0] == Id(f.dom) and f[1:] == Id(f.cod)

    Parameters
    ----------
        name : any
            Name of the box.
        dom : cat.Ob
            Domain.
        cod : cat.Ob
            Codomain.
        data : any
            Extra data in the box, default is `None`.

    """
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        if not str(name):
            raise ValueError(messages.empty_name(name))
        self._name, self._dom, self._cod = name, dom, cod
        self._boxes, self._dagger, self._data = [self], _dagger, data
        Arrow.__init__(self, dom, cod, [self], _scan=False)

    @property
    def name(self):
        """
        The name of a box is immutable.

        >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> assert f.name == 'f'
        >>> f.name = 'g'  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        """
        return self._name

    @property
    def data(self):
        """
        The attribute `data` is immutable, but it can hold a mutable object.

        >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> assert f.data == [42, {0: 1}]
        >>> f.data = [42, {0: 2}]  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        >>> f.data[1][0] = 2
        >>> assert f.data == [42, {0: 2}]
        """
        return self._data

    @property
    def is_dagger(self):
        """
        Whether the box is dagger.
        """
        return self._dagger

    def dagger(self):
        return type(self)(self.name, self.cod, self.dom, data=self.data,
                          _dagger=not self._dagger)

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        return super().__getitem__(key)

    def __repr__(self):
        if self._dagger:
            return repr(self.dagger()) + ".dagger()"
        return "Box({}, {}, {}{})".format(
            *map(repr, [self.name, self.dom, self.cod]),
            ", data=" + repr(self.data) if self.data else '')

    def __str__(self):
        return str(self.name) + ("[::-1]" if self._dagger else '')

    def __hash__(self):
        return hash(super().__repr__())

    def __eq__(self, other):
        if isinstance(other, Box):
            return all(self.__getattribute__(x) == other.__getattribute__(x)
                       for x in ['name', 'dom', 'cod', 'data', '_dagger'])
        if isinstance(other, Arrow):
            return len(other) == 1 and other[0] == self
        return False


class Functor:
    """
    Defines a dagger functor which can be applied to objects and arrows.

    By default, `Functor` defines an endofunctor from the free dagger category
    to itself. The codomain can be changed with the optional parameters
    `ob_cls` and `ar_cls`.

    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> ob, ar = {x: y, y: z, z: y}, {f: g, g: g[::-1]}
    >>> F = Functor(ob, ar)
    >>> assert F(x) == y and F(f) == g

    Parameters
    ----------
    ob : dict_like
        Mapping from :class:`cat.Ob` to `ob_cls`.
    ar : dict_like
        Mapping from :class:`cat.Box` to `ar_cls`.

    Other Parameters
    ----------------
    ob_cls : type, optional
        Class to be used as objects for the codomain of the functor.
        If None, this will be set to :class:`cat.Ob`.
    ar_cls : type, optional
        Class to be used as arrows for the codomain of the functor.
        If None, this will be set to :class:`cat.Arrow`.

    See Also
    --------
    Quiver : For functors from infinitely-generated categories,
             use quivers to create dict-like objects from functions.

    Notes
    -----
    We can check the axioms of dagger functors.

    >>> assert F(Id(x)) == Id(F(x))
    >>> assert F(f >> g) == F(f) >> F(g)
    >>> assert F(f[::-1]) == F(f)[::-1]
    >>> assert F(f.dom) == F(f).dom and F(f.cod) == F(f).cod
    """
    def __init__(self, ob, ar, ob_cls=None, ar_cls=None):
        if ob_cls is None:
            ob_cls = Ob
        if ar_cls is None:
            ar_cls = Arrow
        self.ob_cls, self.ar_cls = ob_cls, ar_cls
        self._ob, self._ar = ob, ar

    @property
    def ob(self):
        """
        >>> F = Functor({Ob('x'): Ob('y')}, {})
        >>> assert F.ob == {Ob('x'): Ob('y')}
        """
        return self._ob

    @property
    def ar(self):
        """
        >>> f, g = Box('f', Ob('x'), Ob('y')), Box('g', Ob('y'), Ob('z'))
        >>> F = Functor({}, {f: g})
        >>> assert F.ar == {f: g}
        """
        return self._ar

    def __eq__(self, other):
        return self.ob == other.ob and self.ar == other.ar

    def __repr__(self):
        return "Functor(ob={}, ar={})".format(repr(self.ob), repr(self.ar))

    def __call__(self, arrow):
        if isinstance(arrow, Ob):
            return self.ob[arrow]
        if isinstance(arrow, Box):
            if arrow.is_dagger:
                return self.ar[arrow.dagger()].dagger()
            return self.ar[arrow]
        if isinstance(arrow, Arrow):
            return self.ar_cls.id(self(arrow.dom)).compose(
                *map(self, arrow.boxes))
        raise TypeError(messages.type_err(Arrow, arrow))


class Quiver:
    """
    Wraps a function into an immutable dict-like object, used as input for a
    :class:`Functor`.

    >>> ob, ar = Quiver(lambda x: x), Quiver(lambda f: f)
    >>> F = Functor(ob, ar)
    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> assert F(x) == x and F(f >> g) == f >> g

    Parameters
    ----------
    func : callable
        Any callable Python object.

    Notes
    -----
    In conjunction with :attr:`Box.data`, this can be used to create a
    :class:`Functor` from a free category with infinitely many generators.

    >>> h = Box('h', x, x, data=42)
    >>> def ar_func(box):
    ...     return Box(box.name, box.dom, box.cod, data=box.data + 1)
    >>> F = Functor(ob, Quiver(ar_func))
    >>> assert F(h).data == 43 and F(F(h)).data == 44

    If :attr:`Box.data` is a mutable object, then so can be the image of a
    :class:`Functor` on it.

    >>> ar = Quiver(lambda f: f if all(f.data) else f[::-1])
    >>> F = Functor(ob, ar)
    >>> m = Box('m', x, x, data=[True])
    >>> assert F(m) == m
    >>> m.data.append(False)
    >>> assert F(m) == m[::-1]
    """
    def __init__(self, func):
        self._func = func

    def __getitem__(self, box):
        return self._func(box)

    def __repr__(self):
        return "Quiver({})".format(repr(self._func))
