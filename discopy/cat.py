# -*- coding: utf-8 -*-

"""
The free (dagger) category
with formal sums, unary operators and symbolic variables.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    Arrow
    Box
    Id
    Sum
    Bubble
    Category
    Functor
    Composable

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        factory
        dumps
        loads

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

We can check the axioms of dagger (i.e. a contravariant involutive
identity-on-objects endofunctor):

>>> x, y, z = Ob('x'), Ob('y'), Ob('z')
>>> f, g = Box('f', x, y), Box('g', y, z)
>>> assert f[::-1][::-1] == f
>>> assert Id(x)[::-1] == Id(x)
>>> assert (f >> g)[::-1] == g[::-1] >> f[::-1]

We can check the axioms of dagger functors.

>>> assert F(Id(x)) == Id(F(x))
>>> assert F(f >> g) == F(f) >> F(g)
>>> assert F(f[::-1]) == F(f)[::-1]
>>> assert F(f.dom) == F(f).dom and F(f.cod) == F(f).cod

Functors are bubble-preserving.

>>> assert F(f.bubble()) == F(f).bubble()
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering, cached_property
from typing import (
    Callable, Mapping, Iterable, Optional, Type, TYPE_CHECKING)

from discopy import messages, utils
from discopy.utils import (
    factory,
    factory_name,
    from_tree,
    rsubs,
    unbiased,
    MappingOrCallable,
    Composable,
    assert_isinstance,
    assert_iscomposable,
    assert_isparallel,
)

if TYPE_CHECKING:
    import sympy

dumps, loads = utils.dumps, utils.loads


@total_ordering
class Ob:
    """
    An object with a string as :code:`name`.

    Parameters:
        name : The name of the object.

    Example
    -------
    >>> x, x_, y = Ob('x'), Ob('x'), Ob('y')
    >>> assert x == x_ and x != y
    """
    def __setstate__(self, state):
        if "name" not in state and "_name" in state:
            state["name"] = state["_name"]
            del state["_name"]
        self.__dict__.update(state)

    def __init__(self, name: str = ""):
        assert_isinstance(name, str)
        self.name = name

    def __repr__(self):
        return f"{factory_name(type(self))}({repr(self.name)})"

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.name == other.name

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return self.name < other.name

    def to_tree(self) -> dict:
        """
        Serialise a DisCoPy object, see :func:`dumps`.

        Example
        -------
        >>> Ob('x').to_tree()
        {'factory': 'cat.Ob', 'name': 'x'}
        """
        return {'factory': factory_name(type(self)), 'name': self.name}

    @classmethod
    def from_tree(cls, tree: dict) -> Ob:
        """
        Decode a serialised DisCoPy object, see :func:`loads`.

        Parameters:
            tree : DisCoPy serialisation.

        Example
        -------
        >>> x = Ob('x')
        >>> assert Ob.from_tree(x.to_tree()) == x
        """
        return cls(tree['name'])


@factory
class Arrow(Composable[Ob]):
    """
    An arrow is a tuple of composable boxes :code:`inside` with a pair of
    objects :code:`dom` and :code:`cod` as domain and codomain.

    Parameters:
        inside: The tuple of boxes inside an arrow.
        dom: The domain of an arrow, i.e. its input.
        cod: The codomain of an arrow, i.e. output
        _scan: Whether to check composition.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            dagger
            bubble

    Tip
    ---
    For code clarity, it is recommended not to initialise arrows directly but
    to use :meth:`Arrow.id` and :meth:`Arrow.then` instead. For example:

    >>> x, y, z = map(Ob, "xyz")
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> arrow = Arrow.id(x).then(f, g)  # Do this...
    >>> arrow_ = Arrow((f, g), x, z)    # ...rather than that!
    >>> assert arrow == arrow_

    Note
    ----
    Arrows can be indexed and sliced using square brackets. Indexing behaves
    like that of strings, i.e. when we index an arrow we get an arrow back.

    >>> assert (f >> g)[0] == f and (f >> g)[1] == g
    >>> assert f[:0] == Arrow.id(f.dom)
    >>> assert f[1:] == Arrow.id(f.cod)

    Note
    ----
    If ``dom`` or ``cod`` are not instances of ``ty_factory``, they are
    automatically cast. This means one can use e.g. ``int`` instead of ``Ob``,
    see :class:`monoidal.PRO`.
    """
    ty_factory = Ob

    def __setstate__(self, state):
        if 'inside' not in state:  # Backward compatibility
            self.dom, self.cod, self.inside = (
                state['_dom'], state['_cod'], tuple(state['_boxes']))
            del state['_dom'], state['_cod'], state['_boxes']
        self.__dict__.update(state)

    def __init__(self, inside: tuple[Box, ...], dom: Ob | str, cod: Ob | str,
                 _scan: bool = True) -> None:
        ty_factory = type(self).ty_factory
        dom = dom if isinstance(dom, ty_factory) else ty_factory(dom)
        cod = cod if isinstance(cod, ty_factory) else ty_factory(cod)
        self.dom, self.cod, self.inside = dom, cod, inside
        if _scan:
            for box in inside:
                assert_isinstance(box, Box)
            for f, g in zip((Id(dom), ) + inside, inside + (Id(cod), )):
                assert_iscomposable(f, g)

    def __iter__(self):
        for box in self.inside:
            yield box

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step == -1:
                inside = tuple(box.dagger() for box in self.inside[key])
                return self.factory(inside, self.cod, self.dom, _scan=False)
            if (key.step or 1) != 1:
                raise IndexError
            inside = self.inside[key]
            if not inside:
                if (key.start or 0) >= len(self):
                    return self.id(self.cod)
                if (key.start or 0) <= -len(self):
                    return self.id(self.dom)
                return self.id(self.inside[key.start or 0].dom)
            return self.factory(
                inside, inside[0].dom, inside[-1].cod, _scan=False)
        if isinstance(key, int):
            if key >= len(self) or key < -len(self):
                raise IndexError
            if key < 0:
                return self[len(self) + key]
            return self[key:key + 1]
        raise TypeError

    def __len__(self):
        return len(self.inside)

    def __repr__(self):
        if not self.inside:  # i.e. self is identity.
            return f"{factory_name(type(self))}.id({repr(self.dom)})"
        return f"{factory_name(self.factory)}(inside={repr(self.inside)}, " \
               f"dom={repr(self.dom)}, cod={repr(self.cod)})"

    def __str__(self):
        return ' >> '.join(map(str, self.inside)) or f"Id({self.dom})"

    def __eq__(self, other):
        return isinstance(other, self.factory)\
            and self.is_parallel(other) and self.inside == other.inside

    def __hash__(self):
        return hash(repr(self))

    def __add__(self, other):
        return self.sum_factory((self, )) + other

    def __radd__(self, other):
        return self if other == 0 else NotImplemented

    @classmethod
    def id(cls: Type[Arrow], dom: Optional[Ob] = None) -> Arrow:
        """
        The identity arrow with the empty tuple inside, called with ``Id``.

        Parameters:
            dom : The domain (and codomain) of the identity.

        Note
        ----
        If ``dom`` is not provided, we use the default value of ``ty_factory``.

        Example
        -------
        >>> assert Arrow.id() == Id() == Id(Ob())
        >>> assert Arrow.id('x') == Id('x') == Id(Ob('x'))
        """
        dom = cls.ty_factory() if dom is None else dom
        return cls.factory((), dom, dom, _scan=False)

    def then(self, *others: Arrow) -> Arrow:
        """
        Sequential composition, called with :code:`>>` and :code:`<<`.

        Parameters:
            others : The other arrows to compose.

        Raises:
            AxiomError : Whenever `self` and `others` do not compose.
        """
        if any(isinstance(other, Sum) for other in others):
            return self.sum_factory((self, )).then(*others)
        inside, dom, cod = self.inside, self.dom, self.cod
        for other in others:
            assert_isinstance(other, self.factory)
            assert_isinstance(self, other.factory)
            inside, cod = inside + other.inside, other.cod
        return self.factory(inside, dom, cod)

    def dagger(self) -> Arrow:
        """ Contravariant involution, called with :code:`[::-1]`. """
        return self[::-1]

    @classmethod
    def zero(cls, dom, cod):
        """
        Return the empty sum with a given domain and codomain.

        Parameters:
            dom : The domain of the empty sum.
            cod : The codomain of the empty sum.
        """
        return cls.sum_factory((), dom, cod)

    def bubble(self, **params) -> Bubble:
        """ Unary operator on homsets. """
        return self.bubble_factory(self, **params)

    @property
    def free_symbols(self) -> "set[sympy.Symbol]":
        """
        The free :code:`sympy` symbols in an arrow.

        Example
        -------

        >>> from sympy.abc import phi, psi
        >>> x, y = Ob('x'), Ob('y')
        >>> f = Box('f', x, y, data={"Alice": [phi + 1]})
        >>> g = Box('g', y, x, data={"Bob": [psi / 2]})
        >>> diagram = (f >> g).bubble() + Id(x)
        >>> assert diagram.free_symbols == {phi, psi}
        """
        return {x for box in self.inside for x in box.free_symbols}

    def subs(self, *args) -> Arrow:
        """
        Substitute a variable by an expression.

        Parameters:
            var (sympy.Symbol) : The subtituted variable.
            expr (sympy.Expr) : The substituting expression.

        Tip
        ---
        You can give a list of :code:`(var, expr)` for multiple substitution.

        Example
        -------
        >>> from sympy.abc import phi, psi
        >>> x, y = Ob('x'), Ob('y')
        >>> f = Box('f', x, y, data={"Alice": [phi + 1]})
        >>> g = Box('g', y, x, data={"Bob": [psi / 2]})
        >>> assert (f >> g).subs(phi, phi + 1) == f.subs(phi, phi + 1) >> g
        >>> assert (f >> g).subs(phi, 1) == f.subs(phi, 1) >> g
        >>> assert (f >> g).subs(psi, 1) == f >> g.subs(psi, 1)
        """
        inside = tuple(box.subs(*args) for box in self.inside)
        return self.factory(inside, self.dom, self.cod, _scan=False)

    def lambdify(self, *symbols: "sympy.Symbol", **kwargs) -> Callable:
        """
        Turn a symbolic diagram into a function from parameters to diagram.

        Parameters:
            symbols : The inputs of the function.
            kwargs : Passed to :code:`sympy.lambdify`.

        Example
        -------
        >>> from sympy.abc import phi, psi
        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g = Box('f', x, y, data=phi), Box('g', y, z, data=psi)
        >>> assert f.lambdify(psi)(42) == f
        >>> assert (f >> g).lambdify(phi, psi)(42, 43)\\
        ...     == Box('f', x, y, data=42) >> Box('g', y, z, data=43)
        """
        return lambda *xs: self.factory(
            dom=self.dom, cod=self.cod, inside=tuple(
                box.lambdify(*symbols, **kwargs)(*xs) for box in self.inside))

    def to_tree(self) -> dict:
        """
        Serialise a DisCoPy arrow, see :func:`discopy.utils.dumps`.

        Example
        -------
        >>> from pprint import PrettyPrinter
        >>> pprint = PrettyPrinter(indent=4, width=70, sort_dicts=False).pprint
        >>> f = Box('f', 'x', 'y', data=42)
        >>> pprint((f >> f[::-1]).to_tree())
        {   'factory': 'cat.Arrow',
            'inside': [   {   'factory': 'cat.Box',
                              'name': 'f',
                              'dom': {'factory': 'cat.Ob', 'name': 'x'},
                              'cod': {'factory': 'cat.Ob', 'name': 'y'},
                              'data': 42},
                          {   'factory': 'cat.Box',
                              'name': 'f',
                              'dom': {'factory': 'cat.Ob', 'name': 'y'},
                              'cod': {'factory': 'cat.Ob', 'name': 'x'},
                              'is_dagger': True,
                              'data': 42}],
            'dom': {'factory': 'cat.Ob', 'name': 'x'},
            'cod': {'factory': 'cat.Ob', 'name': 'x'}}
        """
        return {
            'factory': factory_name(type(self)),
            'inside': [box.to_tree() for box in self.inside],
            'dom': self.dom.to_tree(), 'cod': self.cod.to_tree()}

    @classmethod
    def from_tree(cls, tree: dict) -> Arrow:
        """
        Decode a serialised DisCoPy arrow, see :func:`discopy.utils.loads`.

        Parameters:
            tree : DisCoPy serialisation.

        Example
        -------
        >>> f = Box('f', 'x', 'y', data=42)
        >>> assert Arrow.from_tree((f >> f[::-1]).to_tree()) == f >> f[::-1]
        """
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        inside = tuple(map(from_tree, tree['inside']))
        return cls(inside, dom, cod, _scan=False)


@total_ordering
class Box(Arrow):
    """
    A box is an arrow with a :code:`name` and the tuple of just itself inside.

    Parameters:
        name : The name of the box.
        dom : The domain of the box, i.e. the input.
        cod : The codomain of the box, i.e. the output.
        data (any) : Extra data in the box, default is :code:`None`.
        is_dagger : Whether the box is dagger.

    Example
    -------

    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y, data=[42])
    >>> assert f.inside == (f, )
    """
    def __setstate__(self, state):
        if 'inside' not in state:  # Backward compatibility
            self.name, self.data, self.is_dagger = (
                state['_name'], state['_data'], state['_dagger'])
            del state['_name'], state['_data'], state['_dagger']
        super().__setstate__(state)

    def __init__(
            self, name: str, dom: Ob, cod: Ob, data=None, is_dagger=False):
        assert_isinstance(name, str)
        self.name, self.data, self.is_dagger = name, data, is_dagger
        Arrow.__init__(self, (self, ), dom, cod, _scan=False)

    @cached_property
    def free_symbols(self) -> "set[sympy.Symbol]":
        def recursive_free_symbols(data):
            if isinstance(data, Mapping):
                data = data.values()
            if isinstance(data, Iterable):
                # Handles numpy 0-d arrays, which are actually not iterable.
                if not hasattr(data, "shape") or data.shape != ():
                    return set().union(*map(recursive_free_symbols, data))
            return getattr(data, "free_symbols", set())
        return recursive_free_symbols(self.data)

    def subs(self, *args) -> Box:
        if not any(var in self.free_symbols for var in (
                {var for var, _ in args[0]} if len(args) == 1 else {args[0]})):
            return self
        return type(self)(
            self.name, self.dom, self.cod, is_dagger=self.is_dagger,
            data=rsubs(self.data, *args))

    def lambdify(self, *symbols: "sympy.Symbol", **kwargs) -> Callable:
        if not any(x in self.free_symbols for x in symbols):
            return lambda *xs: self
        from sympy import lambdify
        return lambda *xs: type(self)(
            self.name, self.dom, self.cod, is_dagger=self.is_dagger,
            data=lambdify(symbols, self.data, **kwargs)(*xs))

    def dagger(self) -> Box:
        return type(self)(
            self.name, self.cod, self.dom,
            data=self.data, is_dagger=not self.is_dagger)

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        return super().__getitem__(key)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        str_data = '' if self.data is None else ", data=" + repr(self.data)
        return factory_name(type(self))\
            + f"({repr(self.name)}, {repr(self.dom)}, " \
              f"{repr(self.cod)}{str_data})"

    def __str__(self):
        return str(self.name) + ("[::-1]" if self.is_dagger else '')

    def __hash__(self):
        return hash(Arrow.__repr__(self))

    def __eq__(self, other):
        if isinstance(other, Box):
            return type(self) is type(other)\
                and self.name == other.name\
                and self.is_parallel(other)\
                and self.is_dagger == other.is_dagger\
                and bool(self.data == other.data)
        return isinstance(other, Arrow)\
            and self >> self.id(self.cod) == other  # cast box as diagram

    def __lt__(self, other):
        return self.name < other.name

    def to_tree(self) -> dict:
        tree = {
            'factory': factory_name(type(self)),
            'name': self.name,
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree()}
        if self.is_dagger:
            tree['is_dagger'] = True
        if self.data is not None:
            tree['data'] = self.data
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Box:
        name = tree['name']
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        data, is_dagger = tree.get('data', None), 'is_dagger' in tree
        return cls(name=name, dom=dom, cod=cod, data=data, is_dagger=is_dagger)


class Sum(Box):
    """
    A sum is a tuple of arrows :code:`terms` with the same domain and codomain.

    Parameters:
        terms : The terms of the formal sum.
        dom : The domain of the formal sum.
        cod : The codomain of the formal sum.

    Example
    -------
    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> unit = Sum((), x, y)
    >>> assert f + unit == f == unit + f
    >>> assert f >> (g + g) == (f >> g) + (f >> g) == (f + f) >> g

    Important
    ---------
    Domain and codomain are optional only if the terms are non-empty.

    Note
    ----
    The sum is non-commutative, i.e. :code:`Sum([f, g]) != Sum([g, f])`.
    """
    def __init__(
            self, terms: tuple[Arrow, ...], dom: Ob = None, cod: Ob = None):
        if not terms and (dom is None or cod is None):
            raise ValueError(messages.MISSING_TYPES_FOR_EMPTY_SUM)
        dom = terms[0].dom if dom is None else dom
        cod = terms[0].cod if cod is None else cod
        for arrow in terms:
            assert_isparallel(Sum((), dom, cod), arrow)
        str_args = f", dom={repr(dom)}, cod={repr(cod)}" if not terms else ""
        name = f"{factory_name(type(self))}(terms={repr(terms)}{str_args})"
        self.terms = terms
        super().__init__(name, dom, cod)

    def __eq__(self, other):
        if isinstance(other, Sum):
            return (self.dom, self.cod, self.terms)\
                == (other.dom, other.cod, other.terms)
        return len(self.terms) == 1 and self.terms[0] == other

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return self.name

    def __str__(self):
        return " + ".join(f"({arrow})" for arrow in self.terms)\
            if self.terms else\
            f"{factory_name(type(self))}((), {self.dom}, {self.cod})"

    def __add__(self, other):
        assert_isparallel(self, other)
        other = other if isinstance(other, Sum)\
            else self.sum_factory((other, ))
        return self.sum_factory(self.terms + other.terms, self.dom, self.cod)

    def __iter__(self):
        for arrow in self.terms:
            yield arrow

    def __len__(self):
        return len(self.terms)

    @unbiased
    def then(self, other):
        other = other if isinstance(other, Sum)\
            else self.sum_factory((other, ))
        terms = tuple(f.then(g) for f in self.terms for g in other.terms)
        return self.sum_factory(terms, self.dom, other.cod)

    def dagger(self):
        terms = tuple(f.dagger() for f in self.terms)
        return self.sum_factory(terms, self.cod, self.dom)

    @property
    def free_symbols(self):
        return {x for box in self.terms for x in box.free_symbols}

    def subs(self, *args):
        terms = tuple(f.subs(*args) for f in self.terms)
        return self.sum_factory(terms, self.dom, self.cod)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: self.sum_factory(
            tuple(box.lambdify(*symbols, **kwargs)(*xs) for box in self.terms),
            dom=self.dom, cod=self.cod)

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'terms': [t.to_tree() for t in self.terms],
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        terms = tuple(map(from_tree, tree['terms']))
        return cls(terms=terms, dom=dom, cod=cod)


class Bubble(Box):
    """
    A bubble is a box with an arrow :code:`arg` inside and an optional pair of
    objects :code:`dom` and :code:`cod`.

    Parameters:
        arg : The arrow inside the bubble.
        dom : The domain of the bubble, default is that of :code:`other`.
        cod : The codomain of the bubble, default is that of :code:`other`.
    """
    def __init__(self, arg: Arrow, dom: Ob = None, cod: Ob = None):
        dom = arg.dom if dom is None else dom
        cod = arg.cod if cod is None else cod
        self.arg = arg
        Box.__init__(self, "Bubble", dom, cod)

    @property
    def is_id_on_objects(self):
        """ Whether the bubble is identity on objects. """
        return (self.dom, self.cod) == (self.arg.dom, self.arg.cod)

    def __str__(self):
        str_args = '' if self.is_id_on_objects\
            else f'dom={self.dom}, cod={self.cod}'
        return f"({self.arg}).bubble({str_args})"

    def __repr__(self):
        str_args = repr(self.arg) if self.is_id_on_objects else\
            f"{repr(self.arg)}, dom={repr(self.dom)}, cod={repr(self.cod)}"
        return f"{factory_name(type(self))}({str_args})"

    @property
    def free_symbols(self):
        return super().free_symbols.union(self.arg.free_symbols)

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'arg': self.arg.to_tree(),
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        dom, cod, arg = map(from_tree, (
            tree['dom'], tree['cod'], tree['arg']))
        return cls(arg=arg, dom=dom, cod=cod)


@dataclass
class Category:
    """
    A category is just a pair of Python types :code:`ob` and :code:`ar` with
    appropriate methods :code:`dom`, :code:`cod`, :code:`id` and :code:`then`.

    Parameters:
        ob : The objects of the category, default is :class:`Ob`.
        ar : The arrows of the category, default is :class:`Arrow`.

    Example
    -------
    >>> Category()
    Category(cat.Ob, cat.Arrow)
    >>> CAT
    Category(cat.Category, cat.Functor)
    """
    ob, ar = Ob, Arrow

    def __init__(self, ob: type = None, ar: type = None):
        self.ob, self.ar = (ob or type(self).ob), (ar or type(self).ar)

    def __repr__(self):
        return f"Category({factory_name(self.ob)}, {factory_name(self.ar)})"

    def __eq__(self, other):
        return isinstance(other, Category)\
            and (self.ob, self.ar) == (other.ob, other.ar)

    def __hash__(self):
        return hash((self.ob, self.ar))


class Functor(Composable[Category]):
    """
    A functor is a pair of maps :code:`ob` and :code:`ar` and an optional
    codomain category :code:`cod`.

    Parameters:
        ob : Mapping from :class:`Ob` to :code:`cod.ob`.
        ar : Mapping from :class:`Box` to :code:`cod.ar`.
        cod : The codomain, :code:`Category(Ob, Arrow)` by default.

    Example
    -------
    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> ob, ar = {x: y, y: z, z: y}, {f: g, g: g[::-1]}
    >>> F = Functor(ob, ar)
    >>> assert F(x) == y and F(f) == g

    Tip
    ---
    Both :code:`ob` and :code:`ar` can be a function rather than a dictionary.
    In conjunction with :attr:`Box.data`, this can be used to create a
    :class:`Functor` from a free category with infinitely many generators.

    >>> ob = lambda x: x
    >>> ar = lambda f: Box(f.name, f.dom, f.cod, data=f.data + 1)
    >>> F = Functor(ob, ar)
    >>> h = Box('h', x, x, data=42)
    >>> assert F(h).data == 43 and F(F(h)).data == 44

    If :attr:`Box.data` is a mutable object, then so can be the image of a
    :class:`Functor` on it.

    >>> ar = lambda f: f if all(f.data) else f[::-1]
    >>> F = Functor(ob, ar)
    >>> m = Box('m', x, x, data=[True])
    >>> assert F(m) == m
    >>> m.data.append(False)
    >>> assert F(m) == m[::-1]
    """
    dom = cod = Category(Ob, Arrow)

    @classmethod
    def id(cls, dom: Category = None) -> Functor:
        """
        The identity functor on a given category ``dom``.

        Parameters:
            dom : The domain of the functor.
        """
        return cls(lambda x: x, lambda f: f, dom=dom, cod=dom)

    def then(self, other: Functor) -> Functor:
        """
        The composition of functor with another.

        Parameters:
            other : The other functor with which to compose.

        Note
        ----
        Functor composition is unital only on the left. Indeed, we cannot check
        equality of functors defined with functions instead of dictionaries.

        Example
        -------
        >>> x, y = Ob('x'), Ob('y')
        >>> F, G = Functor({x: y}, {}), Functor({y: x}, {})
        >>> print(F >> G)
        cat.Functor(ob={cat.Ob('x'): cat.Ob('x')}, ar={})
        >>> assert F >> Functor.id() == F != Functor.id() >> F
        >>> print(Functor.id() >> F)  # doctest: +ELLIPSIS
        cat.Functor(ob=<function ...>, ar=...)
        """
        assert_isinstance(other, Functor)
        assert_iscomposable(self, other)
        ob, ar = self.ob.then(other), self.ar.then(other)
        return type(self)(ob, ar, dom=self.dom, cod=other.cod)

    def __init__(
            self,
            ob: Mapping[Ob, Ob] | Callable[[Ob], Ob] | None = None,
            ar: Mapping[Box, Arrow] | Callable[[Box], Arrow] | None = None,
            dom: Category = None, cod: Category = None):
        self.dom, self.cod = dom or type(self).dom, cod or type(self).cod
        self.ob: MappingOrCallable[Ob, Ob] = MappingOrCallable(ob or {})
        self.ar: MappingOrCallable[Box, Arrow] = MappingOrCallable(ar or {})

    def __eq__(self, other):
        return type(self) is type(other)\
            and (self.ob, self.ar, self.cod) == (other.ob, other.ar, other.cod)

    def __repr__(self):
        cod_repr = "" if self.cod == type(self).cod else f", cod={self.cod}"
        return factory_name(type(self))\
            + f"(ob={self.ob}, ar={self.ar}{cod_repr})"

    def __call__(self, other):
        if isinstance(other, Ob):
            return self.ob[other]
        if isinstance(other, Sum):
            return sum(map(self, other.terms),
                       self.cod.ar.zero(self(other.dom), self(other.cod)))
        if isinstance(other, Bubble):
            return self(other.arg).bubble(
                dom=self(other.dom), cod=self(other.cod))
        if isinstance(other, Box) and other.is_dagger:
            return self(other.dagger()).dagger()
        if isinstance(other, Box):
            result = self.ar[other]
            # This allows some nice syntactic sugar for the ar mapping.
            return result if isinstance(result, self.cod.ar)\
                else self.cod.ar(result, self(other.dom), self(other.cod))
        assert_isinstance(other, Arrow)
        result = self.cod.ar.id(self(other.dom))
        for box in other.inside:
            result = result >> self(box)
        return result


Arrow.sum_factory = Sum
Arrow.bubble_factory = Bubble
CAT = Category(Category, Functor)
Id = Arrow.id
