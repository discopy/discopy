# -*- coding: utf-8 -*-

"""
Implements free dagger categories and functors.

We can create boxes with objects as domain and codomain:

>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)

We can create arbitrary diagrams with composition:

>>> diagram = Diagram(x, x, [f, g, h])
>>> assert diagram == f >> g >> h == h << g << f

We can create dagger functors from the free category to itself:

>>> ob = {x: z, y: y, z: x}
>>> ar = {f: g.dagger(), g: f.dagger(), h: h.dagger()}
>>> F = Functor(ob, ar)
>>> assert F(diagram) == (h >> f >> g).dagger()
"""

from functools import reduce as fold


class Ob:
    """
    Defines an object in a free category, only distinguished by its name.

    Parameters
    ----------
    name : any
        Name of the object
    """
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """
        The name of an object is immutable.

        >>> x = Ob('x')
        >>> x.name
        'x'
        >>> x.name = 'y'  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        """
        return self._name

    def __repr__(self):
        return "Ob({})".format(repr(self.name))

    def __str__(self):
        """
        When printing an object, we only print its name.

        >>> x = Ob('x')
        >>> print(x)
        x
        """
        return str(self.name)

    def __eq__(self, other):
        """
        Objects are equal only to objects with equal names.

        >>> x = Ob('x')
        >>> assert x == Ob('x') and x != 'x' and x != Ob('y')
        """
        if not isinstance(other, Ob):
            return False
        return self.name == other.name

    def __hash__(self):
        """
        Objects are hashable whenever their name is.

        >>> d = {Ob(['x', 'y']): 42}
        Traceback (most recent call last):
        ...
        TypeError: unhashable type: 'list'
        """
        return hash(self.name)


class Diagram:
    """
    Defines a diagram in a free dagger category.

    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
    >>> diagram = Diagram(x, x, [f, g, h])
    >>> print(diagram.dagger())
    h.dagger() >> g.dagger() >> f.dagger()

    Parameters
    ----------
    dom : discopy.cat.Ob
        Domain of the diagram.
    cod : discopy.cat.Ob
        Codomain of the diagram.
    boxes : list of :class:`discopy.cat.Diagram`
        Boxes of the diagram.

    Raises
    ------
    :class:`discopy.cat.AxiomError`
        Whenever the boxes do not compose.

    """
    def __init__(self, dom, cod, boxes, _fast=False):
        if not isinstance(dom, Ob):
            raise ValueError("Domain of type Ob expected, got {} of type {} "
                             "instead.".format(repr(dom), type(dom)))
        if not isinstance(cod, Ob):
            raise ValueError("Codomain of type Ob expected, got {} of type {} "
                             "instead.".format(repr(cod), type(cod)))
        if not _fast:
            scan = dom
            for gen in boxes:
                if not isinstance(gen, Diagram):
                    raise ValueError(
                        "Box of type Diagram expected, got {} of type {} "
                        "instead.".format(repr(gen), type(gen)))
                if scan != gen.dom:
                    raise AxiomError(
                        "Box with domain {} expected, got {} instead."
                        .format(scan, repr(gen)))
                scan = gen.cod
            if scan != cod:
                raise AxiomError(
                    "Box with codomain {} expected, got {} instead."
                    .format(cod, repr(boxes[-1])))
        self._dom, self._cod, self._boxes = dom, cod, tuple(boxes)

    @property
    def dom(self):
        """
        The domain of a diagram is immutable.

        >>> diagram = Diagram(Ob('x'), Ob('x'), [])
        >>> assert diagram.dom == Ob('x')
        >>> diagram.dom = Ob('y')  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        """
        return self._dom

    @property
    def cod(self):
        """
        The codomain of a diagram is immutable.

        >>> diagram = Diagram(Ob('x'), Ob('x'), [])
        >>> assert diagram.cod == Ob('x')
        >>> diagram.cod = Ob('y')  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        """
        return self._cod

    @property
    def boxes(self):
        """
        The list of boxes in a diagram is immutable. Use composition instead.

        >>> f = Box('f', Ob('x'), Ob('y'))
        >>> diagram = Diagram(Ob('x'), Ob('x'), [])
        >>> diagram.boxes.append(f)  # This does nothing.
        >>> assert f not in diagram.boxes
        """
        return list(self._boxes)

    def __len__(self):
        return len(self.boxes)

    def __repr__(self):
        if not self.boxes:  # i.e. self is identity.
            return repr(Id(self.dom))
        if len(self.boxes) == 1:  # i.e. self is a box.
            return repr(self.boxes[0])
        return "Diagram({}, {}, {})".format(
            repr(self.dom), repr(self.cod), repr(self.boxes))

    def __str__(self):
        return " >> ".join(map(str, self.boxes))

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        return self.dom == other.dom and self.cod == other.cod\
            and all(x == y for x, y in zip(self.boxes, other.boxes))

    def __hash__(self):
        return hash(repr(self))

    def then(self, other):
        """
        Returns the composition of `self` with a diagram `other`.

        This method is called using the binary operators `>>` and `<<`:

        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
        >>> assert f.then(g) == f >> g == g << f

        Parameters
        ----------
        other : discopy.cat.Diagram
            such that `self.cod == other.dom`.

        Returns
        -------
        diagram : discopy.cat.Diagram
            such that `diagram.boxes == self.boxes + other.boxes`.

        Raises
        ------
        :class:`discopy.cat.AxiomError`
            whenever `self` and `other` do not compose.

        Notes
        -----

        We can check the axioms of categories
        (i.e. composition is unital and associative):

        >>> assert f >> Id(y) == f == Id(x) >> f
        >>> assert (f >> g) >> h == f >> (g >> h)
        """
        if not isinstance(other, Diagram):
            raise ValueError("Expected Diagram, got {} of type {} instead."
                             .format(repr(other), type(other)))
        if self.cod != other.dom:
            raise AxiomError("{} does not compose with {}."
                             .format(repr(self), repr(other)))
        return Diagram(
            self.dom, other.cod, self.boxes + other.boxes, _fast=True)

    def __rshift__(self, other):
        return self.then(other)

    def __lshift__(self, other):
        return other.then(self)

    def dagger(self):
        """
        Returns the dagger of `self`.

        Returns
        -------
        diagram : discopy.cat.Diagram
            such that
            `diagram.boxes == [box.dagger() for box in self.boxes[::-1]]`

        Notes
        -----
        We can check the axioms of dagger (i.e. a contravariant involutive
        identity-on-objects endofunctor):

        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> assert f.dagger().dagger() == f
        >>> assert Id(x).dagger() == Id(x)
        >>> assert (f >> g).dagger() == g.dagger() >> f.dagger()
        """
        return Diagram(self.cod, self.dom,
                       [f.dagger() for f in self.boxes[::-1]], _fast=True)

    @staticmethod
    def id(x):  # pylint: disable=invalid-name
        """
        Returns the identity diagram on x.

        >>> x = Ob('x')
        >>> assert Diagram.id(x) == Id(x) == Diagram(x, x, [])

        :param x: Any object
        :type x: :class:`discopy.cat.Ob`
        :returns: :class:`discopy.cat.Id`
        """
        return Id(x)


class Id(Diagram):
    """
    Defines the identity diagram on x, i.e. with an empty list of boxes.

    >>> x = Ob('x')
    >>> assert Id(x) == Diagram(x, x, [])

    Parameters
    ----------
        x : discopy.cat.Ob
            Any object.

    See also
    --------
        discopy.cat.Diagram.id
    """
    def __init__(self, x):
        super().__init__(x, x, [], _fast=True)

    def __repr__(self):
        return "Id({})".format(repr(self.dom))

    def __str__(self):
        return "Id({})".format(str(self.dom))


class AxiomError(Exception):
    """
    This is raised whenever we try to build an invalid diagram.
    """


class Box(Diagram):
    """ Defines a box as a diagram with the list of only itself as boxes.

    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y, data=[42])
    >>> assert f == Diagram(x, y, [f])
    >>> assert f.boxes == [f]

    Parameters
    ----------
        name : any
            Name of the box.
        dom : discopy.cat.Ob
            Domain.
        cod : discopy.cat.Ob
            Codomain.
        data : any
            Extra data in the box, default is `None`.

    """
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        self._name, self._dom, self._cod = name, dom, cod
        self._boxes, self._dagger, self._data = [self], _dagger, data
        Diagram.__init__(self, dom, cod, [self], _fast=True)

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

    def dagger(self):
        return type(self)(self.name, self.cod, self.dom, data=self.data,
                          _dagger=not self._dagger)

    def __repr__(self):
        if self._dagger:
            return repr(self.dagger()) + ".dagger()"
        return "Box({}, {}, {}{})".format(
            *map(repr, [self.name, self.dom, self.cod]),
            ", data=" + repr(self.data) if self.data else '')

    def __str__(self):
        return str(self.name) + (".dagger()" if self._dagger else '')

    def __hash__(self):
        return hash(super().__repr__())

    def __eq__(self, other):
        if not isinstance(other, Diagram):
            return False
        if isinstance(other, Box):
            return repr(self) == repr(other)
        return len(other.boxes) == 1 and other.boxes[0] == self


class Functor:
    """
    Defines a functor given its image on objects and boxes.
    """
    def __init__(self, ob, ar, ob_cls=Ob, ar_cls=Diagram):
        self.ob_cls, self.ar_cls = ob_cls, ar_cls
        self._ob, self._ar = ob, ar

    @property
    def ob(self):  # pylint: disable=invalid-name
        """
        >>> F = Functor({Ob('x'): Ob('y')}, {})
        >>> assert F.ob == {Ob('x'): Ob('y')}
        """
        return self._ob

    @property
    def ar(self):  # pylint: disable=invalid-name
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

    def __call__(self, diagram):
        if isinstance(diagram, Ob):
            return self.ob[diagram]
        if isinstance(diagram, Box):
            if diagram._dagger:
                return self.ar[diagram.dagger()].dagger()
            return self.ar[diagram]
        if isinstance(diagram, Diagram):
            return fold(lambda g, h: g >> self(h),
                        diagram.boxes, self.ar_cls.id(self(diagram.dom)))
        raise ValueError("Expected Ob, Box or Diagram, got {} instead."
                         .format(repr(diagram)))


class Quiver:
    """
    Wraps a Python function into a dict, to be used as input to Functor.
    """
    def __init__(self, func):
        self._func = func

    def __getitem__(self, box):
        return self._func(box)

    def __repr__(self):
        return "Quiver({})".format(repr(self._func))
