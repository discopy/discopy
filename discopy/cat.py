# -*- coding: utf-8 -*-

"""
The free category
(enriched in monoids, unary operators and :code:`sympy` symbols).

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
    AxiomError

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        factory

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

from functools import total_ordering, cached_property
from collections.abc import Callable, Mapping, Iterable

from discopy import messages
from discopy.utils import (
    factory_name,
    from_tree,
    rsubs,
    rmap,
    assert_isinstance,
)


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
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return "{}({})".format(factory_name(type(self)), repr(self.name))

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

    def to_tree(self) -> dict:
        """ See :func:`discopy.utils.dumps`. """
        return {'factory': factory_name(type(self)), 'name': self.name}

    @classmethod
    def from_tree(cls, tree: dict) -> Ob:
        """
        See :func:`discopy.utils.loads`.

        Parameters:
            tree : DisCoPy serialisation.
        """
        return cls(tree['name'])


def factory(cls: type) -> type:
    """
    Allows the composition of a subclass to remain in the subclass.

    Parameters:
        cls : Some subclass of :class:`Arrow`.

    Example
    -------
    Let's create :code:`Circuit` as a subclass of :class:`Arrow`.

    >>> @factory
    ... class Circuit(Arrow):
    ...     pass

    The :code:`Circuit` subclass itself has a subclass :code:`Gate` as boxes.

    >>> class Gate(Box, Circuit):
    ...     pass

    The identity and composition of :code:`Circuit` is again a :code:`Circuit`.

    >>> X = Gate('X', Ob('qubit'), Ob('qubit'))
    >>> assert isinstance(X >> X, Circuit)
    >>> assert isinstance(Circuit.id(Ob('qubit')), Circuit)
    """
    cls.factory = cls
    return cls


class Composable:
    """
    Abstract class implementing the syntactic sugar :code:`>>` and :code:`<<`
    for forward and backward composition with some method :code:`then`.

    Example
    -------
    >>> class List(list, Composable):
    ...     def then(self, other):
    ...         return self + other
    >>> assert List([1, 2]) >> List([3]) == List([1, 2, 3])
    >>> assert List([3]) << List([1, 2]) == List([1, 2, 3])
    """
    def then(self, other: Composable) -> Composable:
        """
        Sequential composition, to be instantiated.

        Parameters:
            other : The other arrow to compose sequentially.
        """
        raise NotImplementedError

    __rshift__ = __llshift__ = lambda self, other: self.then(other)
    __lshift__ = __lrshift__ = lambda self, other: other.then(self)


@factory
class Arrow(Composable):
    """
    An arrow is a tuple of composable arrows :code:`inside` with a pair of
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
            sum
            bubble

    Tip
    ---
    For code clarity, it is recommended not to initialise arrows directly and
    use :meth:`Arrow.id` and :meth:`Arrow.then` instead. For example:

    >>> x, y, z = map(Ob, "xyz")
    >>> f, g = Box('f', x, y), Box('g', y, z)
    >>> arrow = Arrow.id(x).then(f, g)  # Do this...
    >>> arrow_ = Arrow((f, g), x, z)    # ...rather than that!
    >>> assert arrow == arrow_

    Tip
    ---
    Arrows can be indexed and sliced like ordinary Python lists. For example:

    >>> assert (f >> g)[0] == f and (f >> g)[1] == g
    >>> assert f[:0] == Arrow.id(f.dom)
    >>> assert f[1:] == Arrow.id(f.cod)
    """
    def __init__(
            self, inside: tuple[Arrow, ...], dom: Ob, cod: Ob, _scan=True):
        assert_isinstance(dom, Ob)
        assert_isinstance(cod, Ob)
        if _scan:
            scan = dom
            for depth, box in enumerate(inside):
                assert_isinstance(box, Arrow)
                if box.dom != scan:
                    raise AxiomError(messages.does_not_compose(
                        inside[depth - 1] if depth else Id(dom), box))
                scan = box.cod
            if scan != cod:
                raise AxiomError(messages.does_not_compose(
                    inside[-1] if inside else Id(dom), Id(cod)))
        self.dom, self.cod, self.inside = dom, cod, inside

    def __iter__(self):
        for box in self.inside:
            yield box

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step == -1:
                inside = tuple(box[::-1] for box in self.inside[key])
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
            if key < 0:
                return self[len(self) + key]
            if key >= len(self):
                raise IndexError
            return self[key:key + 1]
        raise TypeError

    def __len__(self):
        return len(self.inside)

    def __repr__(self):
        if not self.inside:  # i.e. self is identity.
            return "{}.id({})".format(factory_name(type(self)), repr(self.dom))
        return "{}(inside={}, dom={}, cod={})".format(
            factory_name(type(self)),
            repr(self.inside), repr(self.dom), repr(self.cod))

    def __str__(self):
        return ' >> '.join(map(str, self.inside)) or "Id({})".format(self.dom)

    def __eq__(self, other):
        return isinstance(other, Arrow) and all(
            getattr(self, a) == getattr(other, a)
            for a in ["inside", "dom", "cod"])

    def __hash__(self):
        return hash(repr(self))

    def __add__(self, other):
        return self.sum((self, )) + other

    def __radd__(self, other):
        return self if other == 0 else NotImplemented

    @classmethod
    def id(cls, dom: Ob) -> Arrow:
        """
        The identity arrow, i.e. with the empty tuple inside.

        Parameters:
            dom : The domain (and codomain) of the identity.
        """
        return cls.factory((), dom, dom, _scan=False)

    def then(self, *others: Arrow) -> Arrow:
        """
        Sequential composition, called with :code:`>>` and :code:`<<`.

        Parameters:
            others : The other arrows to compose.

        Raises:
            cat.AxiomError : Whenever `self` and `others` do not compose.
        """
        if not others:
            return self
        if len(others) > 1:
            return self.then(others[0]).then(*others[1:])
        other, = others
        if isinstance(other, self.sum):
            return self.sum((self, )).then(other)
        assert_isinstance(other, self.factory)
        assert_isinstance(self, other.factory)
        if self.cod != other.dom:
            raise AxiomError(messages.does_not_compose(self, other))
        return self.factory(
            self.inside + other.inside, self.dom, other.cod, _scan=False)

    def dagger(self) -> Arrow:
        """ Contravariant involution, called with :code:`[::-1]`. """
        return self[::-1]

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
        >>> assert (f >> g).free_symbols == {phi, psi}
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
        return Functor(ob=lambda x: x, ar=lambda f: f.subs(*args))(self)

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
        return lambda *xs: self.id(self.dom).then(*(
            box.lambdify(*symbols, **kwargs)(*xs) for box in self.inside))

    def bubble(self, **params) -> Bubble:
        """ Unary operator on homsets. """
        return self.bubble_factory(self, **params)

    def to_tree(self) -> dict:
        """ See :func:`discopy.utils.dumps`. """
        return {
            'factory': factory_name(type(self)),
            'dom': self.dom.to_tree(), 'cod': self.cod.to_tree(),
            'inside': [box.to_tree() for box in self.inside]}

    @classmethod
    def from_tree(cls, tree: dict) -> Arrow:
        """
        See :func:`discopy.utils.loads`.

        Parameters:
            tree : DisCoPy serialisation.
        """
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        inside = tuple(map(from_tree, tree['inside']))
        return cls(inside, dom, cod, _scan=False)


Id = Arrow.id


class AxiomError(Exception):
    """ When arrows do not compose. """


@total_ordering
class Box(Arrow):
    """
    A box is an arrow with a :code:`name` and the tuple of just itself inside.

    Parameters:
        name : The name of the box.
        dom : The domain of the box, i.e. the input.
        cod : The codomain of the box, i.e. the output.
        data (any) : Extra data in the box, default is :code:`None`.
        is_dagger (bool, optional) : Whether the box is dagger.

    Example
    -------

    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y, data=[42])
    >>> assert f.inside == (f, )
    """
    def __init__(self, name: str, dom: Ob, cod: Ob, **params):
        self.name = name
        self.is_dagger = params.get("is_dagger", False)
        self.data = params.get("data", None)
        Arrow.__init__(self, (self, ), dom, cod, _scan=False)

    @cached_property
    def free_symbols(self) -> "set[sympy.Symbol]":
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
            name=self.name, dom=self.cod, cod=self.dom,
            data=self.data, is_dagger=not self.is_dagger)

    def __getitem__(self, key):
        if key == slice(None, None, -1):
            return self.dagger()
        return super().__getitem__(key)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        return "{}({}, {}, {}{})".format(
            factory_name(type(self)),
            *map(repr, [self.name, self.dom, self.cod]),
            '' if self.data is None else ", data=" + repr(self.data))

    def __str__(self):
        return str(self.name) + ("[::-1]" if self.is_dagger else '')

    def __hash__(self):
        return hash(super().__repr__())

    def __eq__(self, other):
        if isinstance(other, Box):
            attributes = ['name', 'dom', 'cod', 'data', 'is_dagger']
            return all(
                getattr(self, x) == getattr(other, x) for x in attributes)
        return isinstance(other, Arrow) and other.inside == (self, )

    def __lt__(self, other):
        return self.name < other.name

    def __call__(self, *args, **kwargs):
        if hasattr(self, "_apply"):
            return self._apply(self, *args, **kwargs)
        raise TypeError("Box is not callable, try drawing.diagramize.")

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
        if not terms:
            if dom is None or cod is None:
                raise ValueError(messages.missing_types_for_empty_sum())
        else:
            dom = terms[0].dom if dom is None else dom
            cod = terms[0].cod if cod is None else cod
            if (dom, cod) != (terms[0].dom, terms[0].cod):
                raise AxiomError(
                    messages.cannot_add(Sum((), dom, cod), terms[0]))
        for arrow in terms:
            if (arrow.dom, arrow.cod) != (dom, cod):
                raise AxiomError(messages.cannot_add(terms[0], arrow))
        name = "{}(terms={}{})".format(
            factory_name(type(self)), repr(terms), ", dom={}, cod={}".format(
                repr(dom), repr(cod)) if not terms else "")
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
        return " + ".join("({})".format(arrow) for arrow in self.terms)\
            if self.terms else "{}((), {}, {})".format(
                factory_name(type(self)), self.dom, self.cod)

    def __add__(self, other):
        if (self.dom, self.cod) != (other.dom, other.cod):
            raise AxiomError(messages.cannot_add(self, other))
        other = other if isinstance(other, Sum) else self.sum((other, ))
        return self.sum(self.terms + other.terms, self.dom, self.cod)

    def __iter__(self):
        for arrow in self.terms:
            yield arrow

    def __len__(self):
        return len(self.terms)

    def then(self, *others):
        if len(others) != 1:
            return super().then(*others)
        other, = others
        other = other if isinstance(other, Sum) else self.sum((other, ))
        terms = tuple(f.then(g) for f in self.terms for g in other.terms)
        return self.sum(terms, self.dom, other.cod)

    def dagger(self):
        terms = tuple(f.dagger() for f in self.terms)
        return self.sum(terms, self.cod, self.dom)

    @property
    def free_symbols(self):
        return {x for box in self.terms for x in box.free_symbols}

    def subs(self, *args):
        terms = tuple(f.subs(*args) for f in self.terms)
        return self.sum(terms, self.dom, self.cod)

    def lambdify(self, *symbols, **kwargs):
        return lambda *xs: self.sum(
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
        return "({}).bubble({})".format(
            self.arg, "" if self.is_id_on_objects
            else "dom={}, cod={}".format(self.dom, self.cod))

    def __repr__(self):
        return factory_name(type(self)) + "({})".format(
            repr(self.arg) if self.is_id_on_objects
            else "{}, dom={}, cod={})".format(*map(repr, [
                self.arg, self.dom, self.cod])))

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


Arrow.sum = Sum
Arrow.bubble_factory = Bubble


class Category:
    """
    A category is just a pair of Python types :code:`ob` and :code:`ar` with
    appropriate methods :code:`dom`, :code:`cod`, :code:`id` and :code:`then`.

    Parameters:
        ob : The objects of the category, default is :class:`Ob`.
        ar : The arrows of the category, default is :class:`Arrow`.
    """
    ob, ar = Ob, Arrow

    def __init__(self, ob: type = None, ar: type = None):
        self.ob, self.ar = (ob or type(self).ob), (ar or type(self).ar)


class Functor:
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

    def __init__(self, ob: Mapping[Ob, Ob], ar: Mapping[Box, Arrow],
                 cod: Category = None):
        class Dict:
            """ dict-like object from callable. """
            def __init__(self, func: Callable):
                self.func = func

            def __getitem__(self, key):
                return self.func(key)
        self.cod = cod or type(self).cod
        self.ob = ob if isinstance(ob, Mapping) else Dict(ob)
        self.ar = ar if isinstance(ar, Mapping) else Dict(ar)

    def __eq__(self, other):
        return self.ob == other.ob and self.ar == other.ar

    def __repr__(self):
        return factory_name(type(self)) + "(ob={}, ar={})".format(
            repr(self.ob), repr(self.ar))

    def __call__(self, other):
        if isinstance(other, Sum):
            return self.cod.ar.sum(
                tuple(map(self, other)), self(other.dom), self(other.cod))
        if isinstance(other, Bubble):
            return self(other.arg).bubble(
                dom=self(other.dom), cod=self(other.cod))
        if isinstance(other, Ob):
            return self.ob[other]
        if isinstance(other, Box) and other.is_dagger:
            return self.ar[other.dagger()].dagger()
        if isinstance(other, Box):
            result = self.ar[other]
            if isinstance(result, self.cod.ar): return result
            # This allows some nice syntactic sugar for the ar mapping.
            return self.cod.ar(result, self(other.dom), self(other.cod))
            return self.ar[other]
        if isinstance(other, Arrow):
            result = self.cod.ar.id(self(other.dom))
            for box in other.inside:
                result = result >> self(box)
            return result
        raise TypeError
