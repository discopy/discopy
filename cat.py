# -*- coding: utf-8 -*-

"""
Implements free dagger categories and functors.
We can check the axioms of categories and functors.

>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g, h = Gen('f', x, y), Gen('g', y, z), Gen('h', z, x)
>>>
>>> assert Id(x) >> f == f == f >> Id(y)
>>> assert (f >> g).dom == f.dom and (f >> g).cod == g.cod
>>> assert f >> g >> h == f >> (g >> h)
>>>
>>> F = Functor(ob={x: y, y: z, z: x}, ar={f: g, g: h})
>>> assert F(Id(x)) == Id(F(x))
>>> assert F(f >> g) == F(f) >> F(g)
"""

from discopy import config
from functools import reduce as fold


class Ob(object):
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
        """ Name of the object, can be of any hashable type.

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


class Arrow(list):
    """ Defines an arrow with domain, codomain and a list of generators.

    >>> x, y, z, w = Ob('x'), Ob('y'), Ob('z'), Ob('w')
    >>> f, g, h = Gen('f', x, y), Gen('g', y, z), Gen('h', z, w)
    >>> assert f >> g >> h == Arrow(x, w, [f, g, h])
    """

    def __init__(self, dom, cod, gens):
        """
        >>> Arrow(Ob('x'), Ob('y'), [Gen('f', Ob('x'), Ob('y'))])
        Arrow(Ob('x'), Ob('y'), [Gen('f', Ob('x'), Ob('y'))])
        >>> Arrow('x', Ob('x'), [])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Domain of type Ob expected, got 'x' ... instead.
        >>> Arrow(Ob('x'), 'x', [])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Codomain of type Ob expected, got 'x' ... instead.
        >>> Arrow(Ob('x'), Ob('x'), [Ob('x')])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Generator of type Arrow expected, got Ob('x') ... instead.
        """
        if not isinstance(dom, Ob):
            raise ValueError("Domain of type Ob expected, got {} of type {} "
                             "instead.".format(repr(dom), type(dom)))
        if not isinstance(cod, Ob):
            raise ValueError("Codomain of type Ob expected, got {} of type {} "
                             "instead.".format(repr(cod), type(cod)))
        if not config.fast:
            scan = dom
            for f in gens:
                if not isinstance(f, Arrow):
                    raise ValueError(
                        "Generator of type Arrow expected, got {} of type {} "
                        "instead.".format(repr(f), type(f)))
                if scan != f.dom:
                    raise AxiomError(
                        "Generator with domain {} expected, got {} instead."
                        .format(scan, repr(f)))
                scan = f.cod
            if scan != cod:
                raise AxiomError(
                    "Generator with codomain {} expected, got {} instead."
                    .format(cod, repr(gens[-1])))
        self._dom, self._cod, self._gens = dom, cod, gens
        super().__init__(gens)

    @property
    def dom(self):
        """
        >>> Arrow(Ob('x'), Ob('x'), []).dom
        Ob('x')
        """
        return self._dom

    @property
    def cod(self):
        """
        >>> Arrow(Ob('x'), Ob('x'), []).cod
        Ob('x')
        """
        return self._cod

    @property
    def gens(self):
        """
        >>> Arrow(Ob('x'), Ob('x'), []).gens
        []
        """
        return self._gens

    def __repr__(self):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> Arrow(x, x, [])
        Id(Ob('x'))
        >>> f, g = Gen('f', x, y), Gen('g', y, z)
        >>> Arrow(x, z, [f, g])  # doctest: +ELLIPSIS
        Arrow(Ob('x'), Ob('z'), [Gen(...), Gen(...)])
        """
        if not self.gens:  # i.e. self is identity.
            return repr(Id(self.dom))
        return "Arrow({}, {}, {})".format(
            repr(self.dom), repr(self.cod), repr(self.gens))

    def __str__(self):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Gen('f', x, y), Gen('g', y, z)
        >>> print(Arrow(x, z, [f, g]))
        f >> g
        """
        return " >> ".join(map(str, self))

    def __eq__(self, other):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Gen('f', x, y), Gen('g', y, z)
        >>> assert f >> g == Arrow(x, z, [f, g])
        """
        if not isinstance(other, Arrow):
            return False
        return self.dom == other.dom and self.cod == other.cod\
            and all(x == y for x, y in zip(self.gens, other.gens))

    def then(self, other):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Gen('f', x, y), Gen('g', y, z)
        >>> assert f.then(g) == f >> g == g << f
        >>> f >> x  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected Arrow, got Ob('x') ... instead.
        """
        if not isinstance(other, Arrow):
            raise ValueError("Expected Arrow, got {} of type {} instead."
                             .format(repr(other), type(other)))
        if self.cod != other.dom:
            raise AxiomError("{} does not compose with {}."
                             .format(repr(self), repr(other)))
        return Arrow(self.dom, other.cod, self.gens + other.gens)

    def __rshift__(self, other):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Gen('f', x, y), Gen('g', y, z)
        >>> assert f.then(g) == f >> g == g << f
        """
        return self.then(other)

    def __lshift__(self, other):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Gen('f', x, y), Gen('g', y, z)
        >>> assert f.then(g) == f >> g == g << f
        """
        return other.then(self)

    def dagger(self):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Gen('f', x, y), Gen('g', y, z)
        >>> h = Arrow(x, z, [f, g])
        >>> assert h.dagger() == g.dagger() >> f.dagger()
        >>> assert h.dagger().dagger() == h
        """
        return Arrow(self.cod, self.dom, [f.dagger() for f in self.gens[::-1]])

    @staticmethod
    def id(x):
        """
        >>> assert Arrow.id(Ob('x')) == Arrow(Ob('x'), Ob('x'), [])
        """
        return Id(x)


class Id(Arrow):
    """ Define an identity arrow, i.e. with an empty list of generators.

    >>> assert Id(Ob('x')) == Arrow.id(Ob('x')) == Arrow(Ob('x'), Ob('x'), [])
    """
    def __init__(self, x):
        """
        >>> idx = Id(Ob('x'))
        >>> assert idx >> idx == idx
        >>> assert idx.dagger() == idx
        """
        super().__init__(x, x, [])

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
    >>> f, g = Gen('f', x, y), Gen('g', y, z)
    >>> Arrow(x, y, [g])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    cat.AxiomError: Generator with domain x expected, got Gen('g', ...
    >>> Arrow(x, z, [f])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    cat.AxiomError: Generator with codomain z expected, got Gen('f', ...
    >>> g >> f  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    cat.AxiomError: Gen('g',...) does not compose with Gen('f', ...).
    """
    pass


class Gen(Arrow):
    """ Defines a generator as an arrow with a name, and itself as generator.
    Generators can hold any Python object as data attribute, default is None.

    Note that when we compose a generator with an identity,
    we get an arrow that is defined as equal to the original generator.

    >>> f = Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
    >>> Id(Ob('x')) >> f  # doctest: +ELLIPSIS
    Arrow(Ob('x'), Ob('y'), [Gen('f', ...)])
    >>> f >> Id(Ob('y')) == f == Id(Ob('x')) >> f
    True
    """
    def __init__(self, name, dom, cod, data=None, _dagger=False):
        """
        >>> Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        """
        self._name, self._dom, self._cod = name, dom, cod
        self._gens, self._dagger, self._data = [self], _dagger, data
        super().__init__(dom, cod, [self])

    @property
    def name(self):
        """
        >>> Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}]).name
        'f'
        """
        return self._name

    @property
    def data(self):
        """
        >>> f = Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> f.data
        [42, {0: 1}]
        >>> f.data[1][0] = 2
        >>> f.data
        [42, {0: 2}]
        """
        return self._data

    def dagger(self):
        """
        >>> f = Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> assert f.dom == f.dagger().cod and f.cod == f.dagger().dom
        >>> assert f == f.dagger().dagger()
        """
        return Gen(self.name, self.cod, self.dom, data=self.data,
                   _dagger=not self._dagger)

    def __repr__(self):
        """
        >>> f = Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
        >>> f  # doctest: +ELLIPSIS
        Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}, <function ...])
        >>> f.dagger()  # doctest: +ELLIPSIS
        Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}, <function ...]).dagger()
        """
        if self._dagger:
            return repr(self.dagger()) + ".dagger()"
        return "Gen({}, {}, {}{})".format(
            *map(repr, [self.name, self.dom, self.cod]),
            ", data=" + repr(self.data) if self.data else '')

    def __str__(self):
        """
        >>> f = Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
        >>> print(f)
        f
        >>> print(f.dagger())
        f.dagger()
        """
        return str(self.name) + (".dagger()" if self._dagger else '')

    def __hash__(self):
        """
        >>> {Gen('f', Ob('x'), Ob('y')): 42}[Gen('f', Ob('x'), Ob('y'))]
        42
        """
        return hash(repr(self))

    def __eq__(self, other):
        """
        >>> f = Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> assert f == Arrow(Ob('x'), Ob('y'), [f])
        """
        if not isinstance(other, Arrow):
            return False
        if isinstance(other, Gen):
            return repr(self) == repr(other)
        return len(other) == 1 and other.gens[0] == self


class Functor:
    """
    Defines a functor given its image on objects and arrows.

    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Gen('f', x, y), Gen('g', y, z)
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
    def ob(self):
        """
        >>> Functor({}, {}).ob
        {}
        """
        return self._ob

    @property
    def ar(self):
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

    def __call__(self, f):
        """
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Gen('f', x, y), Gen('g', y, z)
        >>> F = Functor({x: y, y: x, z: z}, {f: f.dagger(), g: f >> g})
        >>> F(F)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Expected Ob, Gen or Arrow, got Functor... instead.
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
        if isinstance(f, Ob):
            return self.ob[f]
        elif isinstance(f, Gen):
            if f._dagger:
                return self.ar[f.dagger()].dagger()
            return self.ar[f]
        elif isinstance(f, Arrow):
            return fold(lambda g, h: g >> self(h), f, Id(self(f.dom)))
        raise ValueError("Expected Ob, Gen or Arrow, got {} instead."
                         .format(repr(f)))


class Quiver:
    """ Wraps a Python function into a dict that holds the arrows of a functor.

    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> F = Functor({x: x, y: y, z: z}, Quiver(lambda x: x))
    >>> f = Gen('f', x, y, data=[0, 1])
    >>> F(f)
    Gen('f', Ob('x'), Ob('y'), data=[0, 1])
    >>> f.data.append(2)
    >>> F(f)
    Gen('f', Ob('x'), Ob('y'), data=[0, 1, 2])
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
