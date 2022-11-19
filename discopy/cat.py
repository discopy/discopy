# -*- coding: utf-8 -*-

"""
discopy.cat
===========

Free dagger categories enriched in monoids, with unary operators on homsets.

Classes
-------

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: ../_autosummary

   Ob
   Arrow
   Box
   Sum
   Bubble
   Category
   Functor
   AxiomError

Examples
--------

- free functors
- python functions
- matrix

Axioms
------

We can create boxes with objects as domain and codomain:

>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)

We can create arbitrary arrows with identity and composition:

>>> arrow = Arrow.id(x).then(f, g, h)
>>> assert arrow == f >> g >> h == h << g << f

We can create dagger functors from the free category to itself:

>>> ob = {x: z, y: y, z: x}
>>> ar = {f: g[::-1], g: f[::-1], h: h[::-1]}
>>> F = Functor(ob, ar)
>>> assert F(arrow) == (h >> f >> g)[::-1]
"""

from __future__ import annotations
from functools import total_ordering
from collections.abc import Mapping, Iterable

from discopy import messages
from discopy.utils import factory_name, from_tree, rsubs, rmap


@total_ordering
class Ob:
    """ An object with a :code:`name`. """
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """ Name of the object. """
        return self._name

    def __repr__(self):
        return "{}.{}({})".format(
            type(self).__module__, type(self).__name__, repr(self.name))

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        if not isinstance(other, Ob):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def to_tree(self):
        """ See :func:`discopy.utils.dumps`. """
        return {'factory': factory_name(self), 'name': self.name}

    @classmethod
    def from_tree(cls, tree):
        """ See :func:`discopy.utils.loads`. """
        return cls(tree['name'])


def factory(cls):
    """
    Inheritance mechanism for :meth:`Arrow.id` and :meth:`Arrow.then`.
    """
    cls._factory = cls
    return cls


@factory
class Arrow:
    """
    An arrow is a tuple of composable arrows :code:`inside` with a pair of
    objects :code:`dom` and :code:`cod` as domain and codomain.

    Raises
    ------
    :class:`cat.AxiomError`
        Whenever the arrows inside do not compose.

    Notes
    -----

    The Boolean argument :code:`_scan` allows to avoid checking composition.

    For code clarity, it is recommended not to initialise arrows directly: use
    :meth:`Arrow.id` and :meth`Arrow.then` instead. For example:

    >>> x, y, z = map(Ob, "xyz")
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> arrow = Arrow((f, g), x, z)  # Don't do this...
    >>> arrow_ = Arrow.id(x).then(f, g)  # ...do this instead!
    >>> assert arrow == arrow_
    """
    def __init__(self, inside: tuple[Arrow], dom: Ob, cod: Ob, _scan=True):
        if not isinstance(dom, Ob):
            raise TypeError(messages.type_err(Ob, dom))
        if not isinstance(cod, Ob):
            raise TypeError(messages.type_err(Ob, cod))
        if _scan:
            scan = dom
            for depth, box in enumerate(inside):
                if not isinstance(box, Arrow):
                    raise TypeError(messages.type_err(Arrow, box))
                if box.dom != scan:
                    raise AxiomError(messages.does_not_compose(
                        inside[depth - 1] if depth else Id(dom), box))
                scan = box.cod
            if scan != cod:
                raise AxiomError(messages.does_not_compose(
                    inside[-1] if inside else Id(dom), Id(cod)))
        self._dom, self._cod, self._inside = dom, cod, inside

    @property
    def dom(self):
        """ The domain of an arrow, i.e. its input. """
        return self._dom

    @property
    def cod(self):
        """ The codomain of an arrow, i.e. its output. """
        return self._cod

    @property
    def inside(self):
        """ The list of boxes inside an arrow. """
        return self._inside

    boxes = inside

    def __iter__(self):
        for box in self.inside:
            yield box

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step == -1:
                inside = tuple(box[::-1] for box in self.inside[key])
                return self._factory(inside, self.cod, self.dom, _scan=False)
            if (key.step or 1) != 1:
                raise IndexError
            inside = self.inside[key]
            if not inside:
                if (key.start or 0) >= len(self):
                    return self.id(self.cod)
                if (key.start or 0) <= -len(self):
                    return self.id(self.dom)
                return self.id(self.inside[key.start or 0].dom)
            return self._factory(
                inside, inside[0].dom, inside[-1].cod, _scan=False)
        return self.inside[key]

    def __len__(self):
        return len(self.inside)

    def __repr__(self):
        if not self.inside:  # i.e. self is identity.
            return "{}.{}.id({})".format(
                type(self).__module__, type(self).__name__, repr(self.dom))
        if len(self.inside) == 1:  # i.e. self is a box.
            return repr(self.inside[0])
        return "{}.{}(inside={}, dom={}, cod={})".format(
            type(self).__module__, type(self).__name__,
            repr(self.inside), repr(self.dom), repr(self.cod))

    def __str__(self):
        return ' >> '.join(map(str, self.inside))\
            or str(self.id(self.dom))

    def __eq__(self, other):
        return isinstance(other, Arrow) and all(
            getattr(self, a) == getattr(other, a)
            for a in ["inside", "dom", "cod"])

    def __hash__(self):
        return hash(repr(self))

    def __add__(self, other):
        return self.sum([self]) + other

    def __radd__(self, other):
        return self + other

    def then(self, *others: Arrow) -> Arrow:
        """
        Sequential composition, called with `>>` and `<<`.

        Parameters
        ----------
        others : cat.Arrow
            such that `self.cod == others[0].dom`
            and `all(x.cod == y.dom for x, y in zip(others, others[1:])`.

        Returns
        -------
        arrow : cat.Arrow
            such that :code:`arrow.inside == self.inside
            + sum(other.inside for other in others, ())`.

        Raises
        ------
        :class:`cat.AxiomError`
            whenever `self` and `others` do not compose.
        """
        if not others:
            return self
        if len(others) > 1:
            return self.then(others[0]).then(*others[1:])
        other, = others
        if isinstance(other, Sum):
            return self.sum([self]).then(other)
        if not isinstance(other, Arrow):
            raise TypeError(messages.type_err(Arrow, other))
        if self.cod != other.dom:
            raise AxiomError(messages.does_not_compose(self, other))
        return self._factory(
            self.inside + other.inside, self.dom, other.cod, _scan=False)

    def __rshift__(self, other):
        return self.then(other)

    def __lshift__(self, other):
        return other.then(self)

    def dagger(self):
        """
        Contravariant involution, called with :code:`[::-1]`.

        Returns
        -------
        arrow : cat.Arrow
            Such that
            :code:`arrow.inside == [box[::-1] for box in self.inside[::-1]]`.

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

    @classmethod
    def id(cls, dom: Ob) -> Arrow:
        """
        The identity arrow on `dom`, i.e. with the empty tuple inside.

        >>> x = Ob('x')
        >>> assert Arrow.id(x) == Id(x) == Arrow((), x, x)
        """
        return cls._factory((), dom, dom, _scan=False)

    @property
    def free_symbols(self):
        """
        Free symbols in a :class:`Arrow`.

        >>> from sympy.abc import phi, psi
        >>> x, y = Ob('x'), Ob('y')
        >>> f = Box('f', x, y, data={"Alice": [phi + 1]})
        >>> g = Box('g', y, x, data={"Bob": [psi / 2]})
        >>> assert (f >> g).free_symbols == {phi, psi}
        """
        return {x for box in self.inside for x in box.free_symbols}

    def subs(self, *args):
        """
        Substitute a variable by an expression.

        Parameters
        ----------
        var : sympy.Symbol
            Subtituted variable.
        expr : sympy.Expr
            Substituting expression.

        Returns
        -------
        arrow : Arrow

        Note
        ----
        You can give a list of (var, expr) pairs for multiple substitution.

        Examples
        --------
        >>> from sympy.abc import phi, psi
        >>> x, y = Ob('x'), Ob('y')
        >>> f = Box('f', x, y, data={"Alice": [phi + 1]})
        >>> g = Box('g', y, x, data={"Bob": [psi / 2]})
        >>> assert (f >> g).subs(phi, phi + 1) == f.subs(phi, phi + 1) >> g
        >>> assert (f >> g).subs(phi, 1) == f.subs(phi, 1) >> g
        >>> assert (f >> g).subs(psi, 1) == f >> g.subs(psi, 1)
        """
        return Functor(ob=lambda x: x, ar=lambda f: f.subs(*args))(self)

    def lambdify(self, *symbols, **kwargs):
        """
        Turns a symbolic diagram into a function from parameters to diagram.

        Parameters
        ----------
        symbols : list of sympy.Symbol
            Inputs of the lambda.
        kwargs : any
            Passed to sympy.lambdify

        Returns
        -------
        lambda : callable
            Takes concrete values returns concrete diagrams.

        Examples
        --------
        >>> from sympy.abc import phi, psi
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y, data=phi), Box('g', y, z, data=psi)
        >>> assert f.lambdify(psi)(42) == f
        >>> assert (f >> g).lambdify(phi, psi)(42, 43)\\
        ...     == Box('f', x, y, data=42) >> Box('g', y, z, data=43)
        """
        return lambda *xs: self.id(self.dom).then(*(
            box.lambdify(*symbols, **kwargs)(*xs) for box in self.inside))

    def bubble(self, **params):
        """ Returns a :class:`cat.Bubble` with the diagram inside. """
        return self.bubble_factory(self, **params)

    def fmap(self, func):
        return func(self)

    def to_tree(self):
        """ Encodes an arrow as a tree. """
        return {
            'factory': factory_name(self),
            'dom': self.dom.to_tree(), 'cod': self.cod.to_tree(),
            'inside': [box.to_tree() for box in self.inside]}

    @classmethod
    def from_tree(cls, tree):
        """ Decodes a tree as an arrow. """
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        inside = list(map(from_tree, tree['inside']))
        return cls(dom, cod, inside, _scan=False)


Id = Arrow.id


class AxiomError(Exception):
    """ When arrows do not compose. """


@total_ordering
class Box(Arrow):
    """
    A box is an arrow with a :code:`name` and the tuple of just itself inside.

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
        is_dagger : bool, optional
            Whether the box is dagger.

    Examples
    --------

    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y, data=[42])
    >>> assert f.inside == (f, )

    """
    def __init__(self, name, dom, cod, **params):
        def recursive_free_symbols(data):
            if hasattr(data, 'tolist'):
                data = data.tolist()
            if isinstance(data, Mapping):
                data = data.values()
            if isinstance(data, Iterable):
                # Handles numpy 0-d arrays, which are actually not iterable.
                if not hasattr(data, "shape") or data.shape != ():
                    return set().union(*map(recursive_free_symbols, data))
            return data.free_symbols if hasattr(data, "free_symbols") else {}
        data, _dagger = params.get("data", None), params.get("_dagger", False)
        self._free_symbols = recursive_free_symbols(data)
        self._name, self._dagger, self._data = name, _dagger, data
        Arrow.__init__(self, (self, ), dom, cod, _scan=False)

    @property
    def name(self):
        """
        The name of a box is immutable.

        >>> f = Box('f', Ob('x'), Ob('y'), data=[42, {0: 1}])
        >>> assert f.name == 'f'
        >>> f.name = 'g'  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute...
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
        AttributeError: can't set attribute...
        >>> f.data[1][0] = 2
        >>> assert f.data == [42, {0: 2}]
        """
        return self._data

    @property
    def free_symbols(self):
        return self._free_symbols

    def subs(self, *args):
        if not any(var in self.free_symbols for var in (
                {var for var, _ in args[0]} if len(args) == 1 else {args[0]})):
            return self
        return type(self)(
            self.name, self.dom, self.cod, _dagger=self._dagger,
            data=rsubs(self.data, *args))

    def lambdify(self, *symbols, **kwargs):
        if not any(x in self.free_symbols for x in symbols):
            return lambda *xs: self
        from sympy import lambdify
        return lambda *xs: type(self)(
            self.name, self.dom, self.cod, _dagger=self._dagger,
            data=lambdify(symbols, self.data, **kwargs)(*xs))

    @property
    def is_dagger(self):
        """
        Whether the box is dagger.
        """
        return self._dagger

    def dagger(self):
        return type(self)(
            name=self.name, dom=self.cod, cod=self.dom,
            data=self.data, _dagger=not self._dagger)

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        return super().__getitem__(key)

    def __repr__(self):
        if self._dagger:
            return repr(self.dagger()) + ".dagger()"
        return "Box({}, {}, {}{})".format(
            *map(repr, [self.name, self.dom, self.cod]),
            '' if self.data is None else ", data=" + repr(self.data))

    def __str__(self):
        return str(self.name) + ("[::-1]" if self._dagger else '')

    def __hash__(self):
        return hash(super().__repr__())

    def __eq__(self, other):
        if isinstance(other, Box):
            attributes = ['_name', '_dom', '_cod', '_data', '_dagger']
            return all(
                getattr(self, x) == getattr(other, x) for x in attributes)
        if isinstance(other, Arrow):
            return len(other) == 1 and other[0] == self
        return False

    def __lt__(self, other):
        return self.name < other.name

    def __call__(self, *args, **kwargs):
        if hasattr(self, "_apply"):
            return self._apply(self, *args, **kwargs)
        raise TypeError("Box is not callable, try drawing.diagramize.")

    def to_tree(self):
        tree = {
            'factory': factory_name(self),
            'name': self.name,
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree()}
        if self.is_dagger:
            tree['is_dagger'] = True
        if self.data is not None:
            tree['data'] = self.data
        return tree

    @classmethod
    def from_tree(cls, tree):
        name = tree['name']
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        data, _dagger = tree.get('data', None), 'is_dagger' in tree
        return cls(name=name, dom=dom, cod=cod, data=data, _dagger=_dagger)


class Sum(Box):
    """
    A sum is a tuple of arrows with the same domain and codomain, i.e.
    it implements enrichment in monoids.

    Parameters
    ----------
    terms : list of :class:`Arrow`
        Terms of the formal sum.
    dom : :class:`Ob`, optional
        Domain of the formal sum,
        optional if :code:`diagrams` is non-empty.
    cod : :class:`Ob`, optional
        Codomain of the formal sum,
        optional if :code:`diagrams` is non-empty.

    Examples
    --------
    >>> x, y = Ob('x'), Ob('y')
    >>> f, g = Box('f', x, y), Box('g', x, y)
    >>> unit = Sum([], x, y)
    >>> assert (f + unit) == Sum([f]) == (unit + f)
    >>> print((f + g) >> (f + g)[::-1])
    (f >> f[::-1]) + (f >> g[::-1]) + (g >> f[::-1]) + (g >> g[::-1])

    Note
    ----
    The sum is non-commutative, i.e. :code:`Sum([f, g]) != Sum([g, f])`.

    A diagram is different from the sum of itself, i.e. :code:`Sum([f]) != f`
    """
    @staticmethod
    def upgrade(old):
        return old

    def __init__(self, terms, dom=None, cod=None):
        self.terms = list(terms)
        if not terms:
            if dom is None or cod is None:
                raise ValueError(messages.missing_types_for_empty_sum())
        else:
            dom = terms[0].dom if dom is None else dom
            cod = terms[0].cod if cod is None else cod
            if (dom, cod) != (terms[0].dom, terms[0].cod):
                raise AxiomError(
                    messages.cannot_add(Sum([], dom, cod), terms[0]))
        for arrow in terms:
            if (arrow.dom, arrow.cod) != (dom, cod):
                raise AxiomError(messages.cannot_add(terms[0], arrow))
        name = "Sum({})".format(repr(terms)) if terms\
            else "Sum([], dom={}, cod={})".format(repr(dom), repr(cod))
        super().__init__(name, dom, cod)

    def __eq__(self, other):
        if not isinstance(other, Sum):
            return False
        return (self.dom, self.cod, self.terms)\
            == (other.dom, other.cod, other.terms)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return self.name

    def __str__(self):
        if not self.terms:
            return "Sum([], {}, {})".format(self.dom, self.cod)
        return " + ".join("({})".format(arrow) for arrow in self.terms)

    def __add__(self, other):
        if other == 0:
            return self
        other = other if isinstance(other, Sum) else Sum([other])
        return self.sum(self.terms + other.terms, self.dom, self.cod)

    def __radd__(self, other):
        return self.__add__(other)

    def __iter__(self):
        for arrow in self.terms:
            yield arrow

    def __len__(self):
        return len(self.terms)

    def then(self, *others):
        if len(others) != 1:
            return super().then(*others)
        other = others[0] if isinstance(others[0], Sum) else Sum(list(others))
        unit = Sum([], self.dom, other.cod)
        terms = [f.then(g) for f in self.terms for g in other.terms]
        return self.upgrade(sum(terms, unit))

    def dagger(self):
        unit = Sum([], self.cod, self.dom)
        return self.upgrade(sum([f.dagger() for f in self.terms], unit))

    @property
    def free_symbols(self):
        return {x for box in self.terms for x in box.free_symbols}

    def subs(self, *args):
        unit = Sum([], self.dom, self.cod)
        return self.upgrade(sum([f.subs(*args) for f in self.terms], unit))

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: self.sum(
            [box.lambdify(*symbols, **kwargs)(*xs) for box in self.terms],
            dom=self.dom, cod=self.cod)

    @staticmethod
    def fmap(func):
        def sum_func(diagram):
            return type(diagram)([func(term) for term in diagram.terms])
        return sum_func

    def to_tree(self):
        return {
            'factory': factory_name(self),
            'terms': [t.to_tree() for t in self.terms],
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        terms = list(map(from_tree, tree['terms']))
        return cls(terms=terms, dom=dom, cod=cod)


class Bubble(Box):
    """
    A unary operator on homsets, i.e. a box with an arrow :code:`other`
    inside and an optional pair of objects :code:`dom` and :code:`cod`.
    """
    def __init__(self, other, dom: Ob = None, cod: Ob = None):
        dom = other.dom if dom is None else dom
        cod = other.cod if cod is None else cod
        self._other = other
        Box.__init__(self, "Bubble", dom, cod)

    @property
    def other(self):
        """ The diagram inside a bubble. """
        return self._other

    def __str__(self):
        return "({}).bubble({})".format(
            self.other,
            "" if (self.dom, self.cod) == (self.other.dom, self.other.cod)
            else "dom={}, cod={}".format(self.dom, self.cod))

    def __repr__(self):
        return "Bubble({}{})".format(
            repr(self.other),
            "" if (self.dom, self.cod) == (self.other.dom, self.other.cod)
            else ", dom={}, cod={})".format(repr(self.dom), repr(self.cod)))

    def to_tree(self):
        return {
            'factory': factory_name(self),
            'other': self.other.to_tree(),
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        dom, cod, other = map(from_tree, (
            tree['dom'], tree['cod'], tree['other']))
        return cls(dom=dom, cod=cod, other=other)


Arrow.sum = Sum
Arrow.bubble_factory = Bubble


class Category:
    """
    A category is just a pair of Python types :code:`ob` and :code:`ar` with
    appropriate methods :code:`dom`, :code:`cod`, :code:`id` and :code:`then`.
    """
    def __init__(self, ob: type = None, ar: type = None):
        self.ob, self.ar = (ob or Ob), (ar or Arrow)


class Functor:
    """
    A functor is a pair of maps :code:`ob` and :code:`ar` and a codomain
    category :code:`cod`, :code:`Category(Ob, Arrow)` by default.

    Examples
    --------
    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> ob, ar = {x: y, y: z, z: y}, {f: g, g: g[::-1]}
    >>> F = Functor(ob, ar)
    >>> assert F(x) == y and F(f) == g

    Notes
    -----
    We can check the axioms of dagger functors.

    >>> assert F(Id(x)) == Id(F(x))
    >>> assert F(f >> g) == F(f) >> F(g)
    >>> assert F(f[::-1]) == F(f)[::-1]
    >>> assert F(f.dom) == F(f).dom and F(f.cod) == F(f).cod

    Functors are bubble-preserving.

    >>> assert F(f.bubble()) == F(f).bubble()

    See Also
    --------
    Quiver : For functors from infinitely-generated categories,
             use quivers to create dict-like objects from functions.
    """
    cod = Category(Ob, Arrow)

    def __init__(self, ob: dict | Callable, ar: dict | Callable, cod=None):
        self.cod = cod or type(self).cod
        self._ob, self._ar = ob, ar

    @property
    def ob(self):
        """
        Mapping on objects.

        >>> F = Functor({Ob('x'): Ob('y')}, {})
        >>> assert F.ob == {Ob('x'): Ob('y')}
        """
        return self._ob if isinstance(self._ob, Mapping) else Quiver(self._ob)

    @property
    def ar(self):
        """
        Mapping on arrows.

        >>> f, g = Box('f', Ob('x'), Ob('y')), Box('g', Ob('y'), Ob('z'))
        >>> F = Functor({}, {f: g})
        >>> assert F.ar == {f: g}
        """
        return self._ar\
            if hasattr(self._ar, "__getitem__") else Quiver(self._ar)

    def __eq__(self, other):
        return self.ob == other.ob and self.ar == other.ar

    def __repr__(self):
        return "Functor(ob={}, ar={})".format(repr(self.ob), repr(self.ar))

    def __call__(self, arrow):
        if isinstance(arrow, Sum):
            return self.cod.ar.sum(
                list(map(self, arrow)), self(arrow.dom), self(arrow.cod))
        if isinstance(arrow, Bubble):
            return self(arrow.other).bubble(
                dom=self(arrow.dom), cod=self(arrow.cod))
        if isinstance(arrow, Ob):
            return self.ob[arrow]
        if isinstance(arrow, Box):
            if arrow.is_dagger:
                return self.ar[arrow.dagger()].dagger()
            return self.ar[arrow]
        if isinstance(arrow, Arrow):
            return self.cod.ar.id(self(arrow.dom)).then(*map(self, arrow))
        raise TypeError(messages.type_err(Arrow, arrow))


class Quiver(Mapping):
    """
    Wraps a function into an immutable dict-like object, used as input for a
    :class:`Functor`.

    Parameters
    ----------
    func : callable
        Any callable Python object.

    Examples
    --------

    >>> ob, ar = Quiver(lambda x: x), Quiver(lambda f: f)
    >>> F = Functor(ob, ar)
    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> assert F(x) == x and F(f >> g) == f >> g

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

    def __len__(self):
        """
        >>> dict(Quiver(lambda x: x))   # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: Quivers have no length, you can't iterate them.
        """
        raise TypeError("Quivers have no length, you can't iterate them.")

    __iter__ = __len__
