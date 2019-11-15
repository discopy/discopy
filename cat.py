"""
Implements free categories and Python-valued functors.
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

from functools import reduce as fold


class _config:
    """ If fast, checking axioms is disabled (approximately twice faster).

    >>> assert _config
    """

    fast = False

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
        >>> Arrow(Ob('x'), Ob('x'), [])
        Arrow(Ob('x'), Ob('x'), [])
        """
        assert isinstance(dom, Ob)
        assert isinstance(cod, Ob)
        if not _config.fast:
            assert all(isinstance(f, Arrow) for f in gens)
            u = dom
            for f in gens:
                assert f.gens  # i.e. f is not the identity arrow
                assert u == f.dom
                u = f.cod
            assert u == cod
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
        >>> f, g = Gen('f', x, y), Gen('g', y, z)
        >>> Arrow(x, z, [f, g])  # doctest: +ELLIPSIS
        Arrow(Ob('x'), Ob('z'), [Gen(...), Gen(...)])
        """
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
        """
        assert isinstance(other, Arrow)
        assert self.cod == other.dom
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

class Gen(Arrow):
    """ Defines a generator as an arrow with a name, and itself as generator.
    Generators can hold any Python object as data attribute, default is None.

    We can check the axioms for free dagger categories. Note that when we
    compose a generator with an identity, we get an arrow that is defined
    as equal to the original generator.

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
        assert isinstance(dom, Ob)
        assert isinstance(cod, Ob)
        self._name, self._dom, self._cod,  = name, dom, cod
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
        >>> Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}]).data
        [42, {0: 1}]
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
        >>> f0 = Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> f1 = Gen('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> assert f0 == f1
        """
        if not isinstance(other, Arrow):
            return False
        if isinstance(other, Gen):
            return repr(self) == repr(other)
        return len(other) == 1 and other.gens[0] == self

class Functor:
    """ Defines a Python-valued functor F given its image on objects and arrows.

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
        if isinstance(f, Gen):
            if f._dagger: return self.ar[f.dagger()].dagger()
            return self.ar[f]
        assert isinstance(f, Arrow)
        return fold(lambda g, h: g >> self(h), f, Id(self(f.dom)))

class Quiver:
    """ Wraps a Python function into a dict that holds the arrows of a functor.

    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> F = Functor({x: x, y: y, z: z}, Quiver(lambda x: x))
    >>> f, g = Gen('f', x, y, data=[0, 1]), Gen('g', y, z, data=[0])
    >>> F(f)
    Gen('f', Ob('x'), Ob('y'), data=[0, 1])
    >>> F(f >> g)  # doctest: +ELLIPSIS
    Arrow(Ob('x'), Ob('z'), ...)
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
        >>> Quiver(lambda x: x)[3] = 42  # doctest: +ELLIPSIS
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
