# -*- coding: utf-8 -*-

"""
Implements free dagger categories and functors.
"""

from functools import reduce as fold


class Ob:
    """
    Defines an object in a free category, only distinguished by its name.

    :param name: Name of the object
    :type name: any

    >>> x = Ob('x')
    >>> x
    Ob('x')

    The name of an object is immutable.

    >>> x.name = 'y'  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    AttributeError: can't set attribute

    Objects with equal names are equal.

    >>> assert x == Ob('x') and x != 'x' and x != Ob('y')

    When printing an object, we only print the name.

    >>> print(x)
    x

    Objects are hashable whenever their name is.

    >>> assert {x: 42}[x] == 42
    >>> d = {Ob(['x', 'y']): 42}
    Traceback (most recent call last):
    ...
    TypeError: unhashable type: 'list'
    """
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """
        Name of the object, can be of any type.
        """
        return self._name

    def __eq__(self, other):
        if not isinstance(other, Ob):
            return False
        return self.name == other.name

    def __repr__(self):
        return "Ob({})".format(repr(self.name))

    def __str__(self):
        return str(self.name)

    def __hash__(self):
        return hash(self.name)


class Diagram:
    """
    Defines a diagram in a free dagger category.

    :param dom: Domain
    :type dom: discopy.cat.Ob
    :param cod: Codomain
    :type cod: discopy.cat.Ob
    :param boxes: A list of boxes of type :class:`discopy.cat.Diagram`
    :type boxes: list
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
        self._dom, self._cod, self._boxes = dom, cod, boxes

    @property
    def dom(self):
        """
        >>> Diagram(Ob('x'), Ob('x'), []).dom
        Ob('x')
        """
        return self._dom

    @property
    def cod(self):
        """
        >>> Diagram(Ob('x'), Ob('x'), []).cod
        Ob('x')
        """
        return self._cod

    @property
    def boxes(self):
        """
        >>> Diagram(Ob('x'), Ob('y'), [Box('f', Ob('x'), Ob('y'))]).boxes
        [Box('f', Ob('x'), Ob('y'))]
        """
        return self._boxes

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
        Returns the composition of self and other.

        :param other: A diagram with `self.cod == other.dom`
        :type other: :class:`discopy.cat.Diagram`
        :returns: :class:`discopy.cat.Diagram`
        :raises: :class:`discopy.cat.AxiomError`
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
        Returns the dagger of self.
        """
        return Diagram(self.cod, self.dom,
                       [f.dagger() for f in self.boxes[::-1]], _fast=True)

    @staticmethod
    def id(x):  # pylint: disable=invalid-name
        """
        Returns the identity diagram on x.
        """
        return Id(x)


class Id(Diagram):
    """
    Defines an identity diagram, i.e. with an empty list of boxes.

    >>> x = Ob('x')
    >>> assert Id(x) == Diagram.id(x) == Diagram(x, x, [])
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
    """ Defines a box as a diagram with a name, and itself as box.
    Boxes can hold any Python object as data attribute, default is None.

    >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
    """
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        self._name, self._dom, self._cod = name, dom, cod
        self._boxes, self._dagger, self._data = [self], _dagger, data
        Diagram.__init__(self, dom, cod, [self], _fast=True)

    @property
    def name(self):
        """
        >>> Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}]).name
        'f'
        """
        return self._name

    @property
    def data(self):
        """
        The data of a box can be a mutable python object.

        >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> assert f.data == [42, {0: 1}]
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
