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

from functools import total_ordering
from collections.abc import Mapping, Iterable

from discopy import messages
from discopy.utils import factory_name, from_tree, rsubs, rmap


@total_ordering
class Ob:
    """
    Defines an object in a free category, only distinguished by its name.

    Parameters
    ----------
    name : any
        Name of the object.

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
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """
        The name of an object is immutable, it cannot be empty.

        >>> x = Ob('x')
        >>> x.name
        'x'
        >>> x.name = 'y'
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute
        """
        return self._name

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

    def __lt__(self, other):
        return self.name < other.name

    def to_tree(self):
        return {'factory': factory_name(self), 'name': self.name}

    @classmethod
    def from_tree(cls, tree):
        return cls(tree['name'])


class Arrow:
    """
    Defines an arrow in a free dagger category.

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

    Examples
    --------

    >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
    >>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
    >>> arrow = Arrow(x, x, [f, g, h])
    >>> print(arrow[::-1])
    h[::-1] >> g[::-1] >> f[::-1]
    """
    def __init__(self, dom, cod, boxes, _scan=True):
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

    @staticmethod
    def upgrade(old):
        """ Allows class inheritance. """
        return old

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
                return self.upgrade(
                    Arrow(self.cod, self.dom, boxes, _scan=False))
            if (key.step or 1) != 1:
                raise IndexError
            boxes = self.boxes[key]
            if not boxes:
                if (key.start or 0) >= len(self):
                    return Id(self.cod)
                if (key.start or 0) <= -len(self):
                    return Id(self.dom)
                return Id(self.boxes[key.start or 0].dom)
            return self.upgrade(
                Arrow(boxes[0].dom, boxes[-1].cod, boxes, _scan=False))
        return self.boxes[key]

    def __len__(self):
        return len(self.boxes)

    def __repr__(self):
        if not self.boxes:  # i.e. self is identity.
            return repr(Id(self.dom))
        if len(self.boxes) == 1:  # i.e. self is a box.
            return repr(self.boxes[0])
        return "Arrow(dom={}, cod={}, boxes={})".format(
            repr(self.dom), repr(self.cod), repr(self.boxes))

    def __str__(self):
        return ' >> '.join(map(str, self)) or str(self.id(self.dom))

    def __eq__(self, other):
        if not isinstance(other, Arrow):
            return False
        return all(getattr(self, a) == getattr(other, a)
                   for a in ["dom", "cod", "boxes"])

    def __hash__(self):
        return hash(repr(self))

    def __add__(self, other):
        return self.sum([self]) + other

    def __radd__(self, other):
        return self + other

    def then(self, *others):
        """
        Returns the composition of `self` with arrows `others`.

        This method is called using the binary operators `>>` and `<<`:

        >>> x, y, z = Ob('x'), Ob('y'), Ob('z')
        >>> f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)
        >>> assert f.then(g) == f >> g == g << f

        Parameters
        ----------
        others : cat.Arrow
            such that `self.cod == others[0].dom`
            and `all(x.cod == y.dom for x, y in zip(others, others[1:])`.

        Returns
        -------
        arrow : cat.Arrow
            such that :code:`arrow.boxes == self.boxes
            + sum(other.boxes for other in others, [])`.

        Raises
        ------
        :class:`cat.AxiomError`
            whenever `self` and `others` do not compose.

        Notes
        -----

        We can check the axioms of categories
        (i.e. composition is unital and associative):

        >>> assert f >> Id(y) == f == Id(x) >> f
        >>> assert (f >> g) >> h == f >> (g >> h)
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
        return self.upgrade(Arrow(
            self.dom, other.cod, self.boxes + other.boxes, _scan=False))

    def __rshift__(self, other):
        return self.then(other)

    def __lshift__(self, other):
        return other.then(self)

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
    def id(dom):
        """
        Returns the identity arrow on `dom`.

        >>> x = Ob('x')
        >>> assert Arrow.id(x) == Id(x) == Arrow(x, x, [])

        Parameters
        ----------
        dom : cat.Ob
            Any object.

        Returns
        -------
        cat.Id
        """
        return Id(dom)

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
        return {x for box in self.boxes for x in box.free_symbols}

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
        return self.upgrade(
            Functor(ob=lambda x: x, ar=lambda f: f.subs(*args))(self))

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
            box.lambdify(*symbols, **kwargs)(*xs) for box in self.boxes))

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
            'boxes': [box.to_tree() for box in self.boxes]}

    @classmethod
    def from_tree(cls, tree):
        """ Decodes a tree as an arrow. """
        dom, cod = map(from_tree, (tree['dom'], tree['cod']))
        boxes = list(map(from_tree, tree['boxes']))
        return cls(dom, cod, boxes, _scan=False)


class Id(Arrow):
    """
    Defines the identity arrow on `dom`, i.e. with an empty list of boxes.

    Parameters
    ----------
    dom : cat.Ob
        Any object.

    Examples
    --------

    >>> x = Ob('x')
    >>> assert Id(x) == Arrow(x, x, [])

    See also
    --------
        cat.Arrow.id
    """
    def __init__(self, dom):
        Arrow.__init__(self, dom, dom, [], _scan=False)

    def __repr__(self):
        return "Id({})".format(repr(self.dom))

    def __str__(self):
        return "Id({})".format(str(self.dom))

    from_tree = Arrow.from_tree


class AxiomError(Exception):
    """
    This is raised whenever we try to build an invalid arrow.
    """


@total_ordering
class Box(Arrow):
    """ Defines a box as an arrow with the list of only itself as boxes.

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

    Examples
    --------

    >>> x, y = Ob('x'), Ob('y')
    >>> f = Box('f', x, y, data=[42])
    >>> assert f == Arrow(x, y, [f])
    >>> assert f.boxes == [f]
    >>> assert f[:0] == Id(f.dom) and f[1:] == Id(f.cod)

    """
    def __init__(self, name, dom, cod, **params):
        def recursive_free_symbols(data):
            if isinstance(data, Mapping):
                data = data.values()
            if isinstance(data, Iterable):
                # Handles numpy 0-d arrays, which are actually not iterable.
                if not hasattr(data, "shape") or data.shape != ():
                    return set().union(*map(recursive_free_symbols, data))
            return data.free_symbols if hasattr(data, "free_symbols") else {}
        data, _dagger = params.get("data", None), params.get("_dagger", False)
        self._free_symbols = recursive_free_symbols(data)
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
    Implements enrichment over monoids, i.e. formal sums of diagrams.

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
    >>> f + g
    Sum([Box('f', Ob('x'), Ob('y')), Box('g', Ob('x'), Ob('y'))])
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
            [box.lambdify(*symbols, **kwargs)(*xs) for box in self.terms])

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
    """ A unary operator on homsets. """
    def __init__(self, inside, dom=None, cod=None):
        dom = inside.dom if dom is None else dom
        cod = inside.cod if cod is None else cod
        self._inside = inside
        Box.__init__(self, "Bubble", dom, cod)

    @property
    def inside(self):
        """ The diagram inside a bubble. """
        return self._inside

    def __str__(self):
        return "({}).bubble({})".format(
            self.inside,
            "" if (self.dom, self.cod) == (self.inside.dom, self.inside.cod)
            else "dom={}, cod={}".format(self.dom, self.cod))

    def __repr__(self):
        return "Bubble({}{})".format(
            repr(self.inside),
            "" if (self.dom, self.cod) == (self.inside.dom, self.inside.cod)
            else ", dom={}, cod={})".format(repr(self.dom), repr(self.cod)))

    def to_tree(self):
        return {
            'factory': factory_name(self),
            'inside': self.inside.to_tree(),
            'dom': self.dom.to_tree(),
            'cod': self.cod.to_tree()}

    @classmethod
    def from_tree(cls, tree):
        dom, cod, inside = map(from_tree, (
            tree['dom'], tree['cod'], tree['inside']))
        return cls(dom=dom, cod=cod, inside=inside)


Arrow.sum = Sum
Arrow.bubble_factory = Bubble


class Functor:
    """
    Defines a dagger functor which can be applied to objects and arrows.

    By default, `Functor` defines an endofunctor from the free dagger category
    to itself. The codomain can be changed with the optional parameters
    `ob_factory` and `ar_factory`.

    Parameters
    ----------
    ob : dict_like
        Mapping from :class:`cat.Ob` to `ob_factory`.
    ar : dict_like
        Mapping from :class:`cat.Box` to `ar_factory`.

    Other Parameters
    ----------------
    ob_factory : type, optional
        Class to be used as objects for the codomain of the functor.
        If None, this will be set to :class:`cat.Ob`.
    ar_factory : type, optional
        Class to be used as arrows for the codomain of the functor.
        If None, this will be set to :class:`cat.Arrow`.

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
    def __init__(self, ob, ar, ob_factory=None, ar_factory=None):
        if ob_factory is None:
            ob_factory = Ob
        if ar_factory is None:
            ar_factory = Arrow
        self.ob_factory, self.ar_factory = ob_factory, ar_factory
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
            return self.ar_factory.sum(
                list(map(self, arrow)), self(arrow.dom), self(arrow.cod))
        if isinstance(arrow, Bubble):
            return self(arrow.inside).bubble(
                dom=self(arrow.dom), cod=self(arrow.cod))
        if isinstance(arrow, Ob):
            return self.ob[arrow]
        if isinstance(arrow, Box):
            if arrow.is_dagger:
                return self.ar[arrow.dagger()].dagger()
            return self.ar[arrow]
        if isinstance(arrow, Arrow):
            return self.ar_factory.id(self(arrow.dom)).then(*map(self, arrow))
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
