# -*- coding: utf-8 -*-

"""
Implements free dagger categories and functors.
We can check the axioms of categories and functors.

>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
>>>
>>> assert Id(x) >> f == f == f >> Id(y)
>>> assert (f >> g).dom == f.dom and (f >> g).cod == g.cod
>>> assert f >> g >> h == f >> (g >> h)
>>>
>>> F = Functor(ob={x: y, y: z, z: x}, ar={f: g, g: h})
>>> assert F(Id(x)) == Id(F(x))
>>> assert F(f >> g) == F(f) >> F(g)
"""

from functools import reduce as fold


class Ob:
    """ Defines an object, only distinguished by its name.

    >>> assert Ob('x') == Ob('x') and Ob('x') != Ob('y')
    """

    def __init__(self, name):
        """
        >>> Ob('x'), Ob(42), Ob('Alice')
        (Ob('x'), Ob(42), Ob('Alice'))
        """
        self._name = name

    @property
    def name(self):
        """ Name of the object, can be of any type.

        >>> Ob('x').name
        'x'
        """
        return self._name

    def __eq__(self, other):
        """
        >>> x, x1, y = Ob('x'), Ob('x'), Ob('y')
        >>> assert x == x1 and x != y and x != 'x'
        >>> assert 'x' != Ob('x')
        """
        if not isinstance(other, Ob):
            return False
        return self.name == other.name

    def __repr__(self):
        """
        >>> Ob('x')
        Ob('x')
        """
        return "Ob({})".format(repr(self.name))

    def __str__(self):
        """
        >>> print(Ob('x'))
        x
        """
        return str(self.name)

    def __hash__(self):
        """
        >>> {Ob('x'): 42}[Ob('x')]
        42
        """
        return hash(repr(self))


class Diagram:
    """ Defines a diagram with domain, codomain and a list of boxes.

    >>> x, y, z, w = Ob('x'), Ob('y'), Ob('z'), Ob('w')
    >>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, w)
    >>> assert f >> g >> h == Diagram(x, w, [f, g, h])
    """

    def __init__(self, dom, cod, boxes, _fast=False):
        """
        >>> Diagram(Ob('x'), Ob('y'), [Box('f', Ob('x'), Ob('y'))])
        Diagram(Ob('x'), Ob('y'), [Box('f', Ob('x'), Ob('y'))])
        >>> Diagram('x', Ob('x'), [])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Domain of type Ob expected, got 'x' ... instead.
        >>> Diagram(Ob('x'), 'x', [])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Codomain of type Ob expected, got 'x' ... instead.
        >>> Diagram(Ob('x'), Ob('x'), [Ob('x')])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Box of type Diagram expected, got Ob('x') ... instead.
        """
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
        >>> Diagram(Ob('x'), Ob('x'), []).boxes
        []
        """
        return self._boxes

    def __len__(self):
        """
        >>> assert len(Diagram(Ob('x'), Ob('x'), [])) == 0
        """
        return len(self.boxes)

    def __repr__(self):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> Diagram(x, x, [])
        Id(Ob('x'))
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> Diagram(x, z, [f, g])  # doctest: +ELLIPSIS
        Diagram(Ob('x'), Ob('z'), [Box(...), Box(...)])
        """
        if not self.boxes:  # i.e. self is identity.
            return repr(Id(self.dom))
        return "Diagram({}, {}, {})".format(
            repr(self.dom), repr(self.cod), repr(self.boxes))

    def __str__(self):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> print(Diagram(x, z, [f, g]))
        f >> g
        """
        return " >> ".join(map(str, self.boxes))

    def __eq__(self, other):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> assert f >> g == Diagram(x, z, [f, g])
        """
        if not isinstance(other, Diagram):
            return False
        return self.dom == other.dom and self.cod == other.cod\
            and all(x == y for x, y in zip(self.boxes, other.boxes))

    def __hash__(self):
        """
        >>> assert {Id(Ob('x')): 42}[Id(Ob('x'))] == 42
        """
        return hash(repr(self))

    def then(self, other):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> assert f.then(g) == f >> g == g << f
        >>> f >> x  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected Diagram, got Ob('x') ... instead.
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
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> assert f.then(g) == f >> g == g << f
        """
        return self.then(other)

    def __lshift__(self, other):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> assert f.then(g) == f >> g == g << f
        """
        return other.then(self)

    def dagger(self):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> h = Diagram(x, z, [f, g])
        >>> assert h.dagger() == g.dagger() >> f.dagger()
        >>> assert h.dagger().dagger() == h
        """
        return Diagram(self.cod, self.dom,
                       [f.dagger() for f in self.boxes[::-1]], _fast=True)

    @staticmethod
    def id(x):  # pylint: disable=invalid-name
        """
        >>> assert Diagram.id(Ob('x')) == Diagram(Ob('x'), Ob('x'), [])
        """
        return Id(x)


class Id(Diagram):
    """ Define an identity diagram, i.e. with an empty list of boxes.

    >>> assert Id(Ob('x')) == Diagram.id(Ob('x'))
    >>> assert Id(Ob('x')) == Diagram(Ob('x'), Ob('x'), [])
    """
    def __init__(self, x):
        """
        >>> idx = Id(Ob('x'))
        >>> assert idx >> idx == idx
        >>> assert idx.dagger() == idx
        """
        super().__init__(x, x, [], _fast=True)

    def __repr__(self):
        """
        >>> Id(Ob('x'))
        Id(Ob('x'))
        """
        return "Id({})".format(repr(self.dom))

    def __str__(self):
        """
        >>> print(Id(Ob('x')))
        Id(x)
        """
        return "Id({})".format(str(self.dom))


class AxiomError(Exception):
    """
    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> Diagram(x, y, [g])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    cat.AxiomError: Box with domain x expected, got Box('g', ...
    >>> Diagram(x, z, [f])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    cat.AxiomError: Box with codomain z expected, got Box('f', ...
    >>> g >> f  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    cat.AxiomError: Box('g',...) does not compose with Box('f', ...).
    """


class Box(Diagram):
    """ Defines a box as a diagram with a name, and itself as box.
    Boxes can hold any Python object as data attribute, default is None.

    Note that when we compose a box with an identity,
    we get a diagram that is defined as equal to the original box.

    >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
    >>> Id(Ob('x')) >> f  # doctest: +ELLIPSIS
    Diagram(Ob('x'), Ob('y'), [Box('f', ...)])
    >>> f >> Id(Ob('y')) == f == Id(Ob('x')) >> f
    True
    """
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        """
        >>> Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        """
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
        >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> f.data
        [42, {0: 1}]
        >>> f.data[1][0] = 2
        >>> f.data
        [42, {0: 2}]
        """
        return self._data

    def dagger(self):
        """
        >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> assert f.dom == f.dagger().cod and f.cod == f.dagger().dom
        >>> assert f == f.dagger().dagger()
        """
        return type(self)(self.name, self.cod, self.dom, data=self.data,
                          _dagger=not self._dagger)

    def __repr__(self):
        """
        >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
        >>> f  # doctest: +ELLIPSIS
        Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}, <function ...])
        >>> f.dagger()  # doctest: +ELLIPSIS
        Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}, <function ...]).dagger()
        """
        if self._dagger:
            return repr(self.dagger()) + ".dagger()"
        return "Box({}, {}, {}{})".format(
            *map(repr, [self.name, self.dom, self.cod]),
            ", data=" + repr(self.data) if self.data else '')

    def __str__(self):
        """
        >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
        >>> print(f)
        f
        >>> print(f.dagger())
        f.dagger()
        """
        return str(self.name) + (".dagger()" if self._dagger else '')

    def __hash__(self):
        """
        >>> {Box('f', Ob('x'), Ob('y')): 42}[Box('f', Ob('x'), Ob('y'))]
        42
        """
        return hash(super().__repr__())

    def __eq__(self, other):
        """
        >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> assert f == Diagram(Ob('x'), Ob('y'), [f])
        """
        if not isinstance(other, Diagram):
            return False
        if isinstance(other, Box):
            return repr(self) == repr(other)
        return len(other.boxes) == 1 and other.boxes[0] == self


class Functor:
    """
    Defines a functor given its image on objects and boxes.

    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> F = Functor({x: y, y: x, z: z}, {f: f.dagger(), g: f >> g})
    >>> assert F((f >> g).dagger()) == F(f >> g).dagger()
    """
    def __init__(self, ob, ar):
        """
        >>> F = Functor({Ob('x'): Ob('y')}, {})
        >>> F(Id(Ob('x')))
        Id(Ob('y'))
        """
        self._ob, self._ar = ob, ar

    @property
    def ob(self):  # pylint: disable=invalid-name
        """
        >>> Functor({}, {}).ob
        {}
        """
        return self._ob

    @property
    def ar(self):  # pylint: disable=invalid-name
        """
        >>> Functor({}, {}).ar
        {}
        """
        return self._ar

    def __eq__(self, other):
        """
        >>> x, y = Ob('x'), Ob('y')
        >>> assert Functor({x: y, y: x}, {}) == Functor({y: x, x: y}, {})
        """
        return self.ob == other.ob and self.ar == other.ar

    def __repr__(self):
        """
        >>> Functor({}, {})
        Functor(ob={}, ar={})
        """
        return "Functor(ob={}, ar={})".format(repr(self.ob), repr(self.ar))

    def __call__(self, diagram):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y), Box('g', y, z)
        >>> F = Functor({x: y, y: x, z: z}, {f: f.dagger(), g: f >> g})
        >>> F(F)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected Ob, Box or Diagram, got Functor... instead.
        >>> print(F(x))
        y
        >>> print(F(f))
        f.dagger()
        >>> print(F(g))
        f >> g
        >>> print(F(f.dagger()))
        f
        >>> print(F(f >> g))
        f.dagger() >> f >> g
        """
        if isinstance(diagram, Ob):
            return self.ob[diagram]
        if isinstance(diagram, Box):
            if diagram._dagger:
                return self.ar[diagram.dagger()].dagger()
            return self.ar[diagram]
        if isinstance(diagram, Diagram):
            return fold(lambda g, h: g >> self(h),
                        diagram.boxes, Id(self(diagram.dom)))
        raise ValueError("Expected Ob, Box or Diagram, got {} instead."
                         .format(repr(diagram)))


class Quiver:
    """ Wraps a Python function into a dict.

    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> F = Functor({x: x, y: y, z: z}, Quiver(lambda x: x))
    >>> f = Box('f', x, y, data=[0, 1])
    >>> F(f)
    Box('f', Ob('x'), Ob('y'), data=[0, 1])
    >>> f.data.append(2)
    >>> F(f)
    Box('f', Ob('x'), Ob('y'), data=[0, 1, 2])
    """
    def __init__(self, func):
        """
        >>> ar = Quiver(lambda x: x ** 2)
        >>> ar[3]
        9
        """
        self._func = func

    def __getitem__(self, box):
        """
        >>> Quiver(lambda x: x * 10)[42]
        420
        >>> Quiver(lambda x: x * 10)[42] = 421  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: 'Quiver' object does not support item assignment
        """
        return self._func(box)

    def __repr__(self):
        """
        >>> Quiver(lambda x: x)  # doctest: +ELLIPSIS
        Quiver(<function <lambda> at ...>)
        """
        return "Quiver({})".format(repr(self._func))
