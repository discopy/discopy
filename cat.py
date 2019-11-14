"""
Implements free categories and Python-valued functors.
We can check the axioms of categories and functors.

>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g, h = Generator('f', x, y), Generator('g', y, z), Generator('h', z, x)
>>>
>>> assert Id(x) >> f == f == f >> Id(y)
>>> assert (f >> g).dom == f.dom and (f >> g).cod == g.cod
>>> assert f >> g >> h == f >> (g >> h)
>>>
>>> ob, ar = {x: y, y: z, z: x}, {f: g, g: h, h: f >> g >> h >> f}
>>> F = Functor(ob, ar)
>>> F(x)
Ob('y')
>>> F(f)
Generator(name='g', dom=Ob('y'), cod=Ob('z'))
>>> F(f >> g >> h)  # doctest: +ELLIPSIS
Arrow(Ob('y'), Ob('y'), ...)
>>> assert F(Id(x)) == Id(F(x))
>>> assert F(f >> g) == F(f) >> F(g)
"""

FAST = False  # If FAST, we do not check axioms (approximately twice faster).

from functools import reduce as fold


class Ob(object):
    """ Defines an object, only distinguished by its name.

    >>> x, x1, y = Ob('x'), Ob('x'), Ob('y')
    >>> assert x == x1 and x != y and x != 'x'
    >>> x
    Ob('x')
    >>> x.name
    'x'
    >>> print(x)
    x
    """
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
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
        return hash(repr(self))

class Arrow(list):
    """ Defines an arrow with domain, codomain and gens: a list of generators.

    >>> f = Generator('f', Ob('x'), Ob('y'))
    >>> g = Generator('g', Ob('y'), Ob('z'))
    >>> h = Arrow(Ob('x'), Ob('z'), [f, g])
    >>> h  # doctest: +ELLIPSIS
    Arrow(Ob('x'), Ob('z'), [...])
    >>> list(h)  # doctest: +ELLIPSIS
    [Generator(name='f', ...), Generator(name='g', ...)]
    >>> print(h)
    f >> g
    >>> h == f.then(g) == f >> g == g << f
    True
    >>> h.dagger()  # doctest: +ELLIPSIS
    Arrow(Ob('z'), Ob('x'), [Generator(name='g', ...).dagger(), ...])
    >>> assert h.dagger() == g.dagger() >> f.dagger()
    """
    def __init__(self, dom, cod, gens):
        if not FAST:
            assert isinstance(dom, Ob)
            assert isinstance(cod, Ob)
            assert isinstance(gens, list)
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
        return self._dom

    @property
    def cod(self):
        return self._cod

    @property
    def gens(self):
        return self._gens

    def __repr__(self):
        return "Arrow({}, {}, {})".format(
            repr(self.dom), repr(self.cod), repr(self.gens))

    def __str__(self):
        return " >> ".join(map(str, self))

    def __eq__(self, other):
        if not isinstance(other, Arrow):
            return False
        return self.dom == other.dom and self.cod == other.cod\
            and all(x == y for x, y in zip(self.gens, other.gens))

    def then(self, other):
        assert isinstance(other, Arrow)
        assert self.cod == other.dom
        return Arrow(self.dom, other.cod, self.gens + other.gens)

    def __rshift__(self, other):
        return self.then(other)

    def __lshift__(self, other):
        return other.then(self)

    def dagger(self):
        return Arrow(self.cod, self.dom, [f.dagger() for f in self.gens[::-1]])

    @staticmethod
    def id(x):
        return Id(x)

class Id(Arrow):
    """ Define an identity arrow, i.e. with an empty list of generators.

    >>> id_x = Id(Ob('x'))
    >>> id_x
    Id(Ob('x'))
    >>> print(id_x)
    Id(x)
    >>> assert id_x == Arrow.id(Ob('x')) == Arrow(Ob('x'), Ob('x'), [])
    >>> assert id_x == id_x.dagger() == id_x.dagger().dagger()
    """
    def __init__(self, x):
        super().__init__(x, x, [])

    def __repr__(self):
        return "Id({})".format(repr(self.dom))

    def __str__(self):
        return "Id({})".format(str(self.dom))

class Generator(Arrow):
    """ Defines a generator as an arrow with a name, and itself as generator.
    Generators can hold any Python object as data attribute, default is None.

    We can check the axioms for free dagger categories. Note that when we
    compose a generator with an identity, we get an arrow that is defined
    as equal to the original generator.

    >>> f = Generator('f', Ob('x'), Ob('y'), data=[42, {0: 1}, lambda x: x])
    >>> f  # doctest: +ELLIPSIS
    Generator(name='f', dom=Ob('x'), cod=Ob('y'), data=[42, ...])
    >>> list(f)  # doctest: +ELLIPSIS
    [Generator(name='f', dom=Ob('x'), cod=Ob('y'), data=[42, ...])]
    >>> print(f)
    f
    >>> Id(Ob('x')) >> f  # doctest: +ELLIPSIS
    Arrow(Ob('x'), Ob('y'), [Generator(name='f', ...)])
    >>> assert Id(Ob('x')) >> f == f == f >> Id(Ob('y'))
    >>> f.dagger()  # doctest: +ELLIPSIS
    Generator(name='f', dom=Ob('x'), cod=Ob('y'), data=[42, ...]).dagger()
    >>> print(f.dagger())
    f.dagger()
    >>> assert f == f.dagger().dagger()
    """
    def __init__(self, name, dom, cod, dagger=False, data=None):
        assert isinstance(dom, Ob)
        assert isinstance(cod, Ob)
        self._name, self._dom, self._cod,  = name, dom, cod
        self._gens, self._dagger, self._data = [self], dagger, data
        super().__init__(dom, cod, [self])

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    def dagger(self):
        return Generator(self.name, self.cod, self.dom,
                         dagger=not self._dagger, data=self.data)

    def __repr__(self):
        if self._dagger:
            return "Generator(name={}, dom={}, cod={}{}).dagger()".format(
                *map(repr, [self.name, self.cod, self.dom]),
                ", data=" + repr(self.data) if self.data else '')
        return "Generator(name={}, dom={}, cod={}{})".format(
            *map(repr, [self.name, self.dom, self.cod]),
            ", data=" + repr(self.data) if self.data else '')

    def __str__(self):
        return str(self.name) + (".dagger()" if self._dagger else '')

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if not isinstance(other, Arrow):
            return False
        if isinstance(other, Generator):
            return repr(self) == repr(other)
        return len(other) == 1 and other.gens[0] == self

class Functor:
    """ Defines a Python-valued functor F given its image on objects and arrows.

    >>> x, y = Ob('x'), Ob('y')
    >>> f = Generator('f', x, y, data=[1, 1, 0])
    >>> ob = {x: y, y: x}
    >>> ar = {f: f.dagger()}
    >>> F = Functor(ob, ar)
    >>> F  # doctest: +ELLIPSIS
    Functor(ob=..., ar=...)
    >>> F(x)
    Ob('y')
    >>> F(f)
    Generator(name='f', dom=Ob('x'), cod=Ob('y'), data=[1, 1, 0]).dagger()
    >>> F(f.dagger())
    Generator(name='f', dom=Ob('x'), cod=Ob('y'), data=[1, 1, 0])
    """
    def __init__(self, ob, ar):
        assert all(isinstance(x, Ob) for x in ob.keys())
        self._ob, self._ar = ob, ar

    @property
    def ob(self):
        return self._ob

    @property
    def ar(self):
        return self._ar

    def __eq__(self, other):
        return self.ob == other.ob and self.ar == other.ar

    def __repr__(self):
        return "Functor(ob={}, ar={})".format(repr(self.ob), repr(self.ar))

    def __call__(self, f):
        if isinstance(f, Ob):
            return self.ob[f]
        if isinstance(f, Generator):
            if f._dagger: return self.ar[f.dagger()].dagger()
            return self.ar[f]
        assert isinstance(f, Arrow)
        return fold(lambda g, h: g >> self(h), f, Id(self(f.dom)))

class Quiver:
    """ Wraps a Python function into a dict that holds the arrows of a functor.

    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> ob, ar = {o: o for o in [x, y, z]}, Quiver(lambda x: x)
    >>> ar[3]
    3
    >>> ar[3] = 4  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
        ar[3] = 4
    TypeError: 'Quiver' object does not support item assignment
    >>> F = Functor(ob, ar)
    >>> F(x)
    Ob('x')
    >>> f, g = Generator('f', x, y, data=[0, 1]), Generator('g', y, z, data=[0])
    >>> F(f)
    Generator(name='f', dom=Ob('x'), cod=Ob('y'), data=[0, 1])
    >>> F(f >> g)  # doctest: +ELLIPSIS
    Arrow(Ob('x'), Ob('z'), ...)
    """
    def __init__(self, func):
        self._func = func

    def __getitem__(self, box):
        return self._func(box)

    def __repr__(self):
        return "Quiver({})".format(repr(self._func))
