"""
The abstract base classes for categories.

These mirror the concrete hierarchy of :mod:`discopy` modules: each class adds
the characteristic generator of its categorical structure as an
:func:`abc.abstractmethod`, e.g. :class:`BraidedCategory` is a
:class:`MonoidalCategory` with an abstract :meth:`BraidedCategory.braid`.

.. raw:: html
    :file: api/architecture.html

Software dependencies between modules go top-to-bottom, left-to-right and
forgetful functors between categories go the other way.

Each class also declares its :func:`discopy.axioms.axiom` equations, which
every free category inherits along with the structure they axiomatise:
:class:`Category` states the unitality and associativity of composition,
the typing of its identities and composites, and the involution and
contravariance of its dagger; a :class:`ColouredMonoid` inherits them as
the unitality and associativity of its product, its composition.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Serialisable
    Category
    ColouredMonoid
    Monoid
    Nat
    MonoidalCategory
    PRO
    TracedCategory
    ResiduatedMonoid
    BiclosedCategory
    Pregroup
    RigidCategory
    PivotalCategory
    BraidedCategory
    PROB
    SymmetricCategory
    PROP
    MarkovCategory
    ClosedCategory
    FeedbackCategory
    BalancedCategory
    RibbonCategory
    CompactCategory
    HypergraphCategory
    NamedGeneric
"""

from __future__ import annotations

import pickle
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Self

from discopy.axioms import (  # noqa: F401
    Axiom, ComposablePair, ComposableTriple, Equation, Theory, axiom,
    no_strategy)
from discopy.utils import (  # noqa: F401
    NamedGeneric, classproperty, factory_name)


class Serialisable(Theory):
    """
    The serialisation interface of DisCoPy, one hook driving all three
    mechanisms: the class attribute ``serialised_attrs`` names attributes that
    are also keyword arguments of ``__init__``, from which follow

    - a generic pair of inverse methods :meth:`to_tree` and
      :meth:`from_tree`, the JSON serialisation behind
      :func:`discopy.utils.dumps` and :func:`discopy.utils.loads`,
    - a generic :meth:`__repr__` such that ``eval(repr(x)) == x``,
    - :meth:`__setstate__`, the terminal of every pickle migration
      chain; the class parameters of
      :class:`discopy.utils.NamedGeneric` are pickled by their own
      machinery.

    A subclass with a different constructor declares its keys once
    instead of reimplementing each method.

    Each mechanism comes with the law that it is a roundtrip, i.e. that
    a term reads back from what it was written to: :meth:`transparency`
    for its representation, :meth:`pickling` and :meth:`copying` for the
    pickle protocol and :meth:`serialisation` for its tree. They are
    axioms like any other, so a class that also generates its instances
    — a :class:`discopy.axioms.Testable` — has them checked against
    generated terms, and one that violates a law declares it
    ``.failing`` rather than leaving it untested.

    Example
    -------
    >>> from discopy.cat import Box
    >>> assert Box.serialised_attrs\\
    ...     == ('name', 'dom', 'cod', 'is_dagger', 'data')
    >>> f = Box('f', 'x', 'y', data=42)
    >>> assert Box.from_tree(f.to_tree()) == f
    """
    serialised_attrs: tuple[str, ...] = ()

    axioms = no_strategy

    def is_default(self, key: str) -> bool:
        """
        Whether the value of an attribute equals its class default,
        in which case :meth:`to_tree` and :meth:`__repr__` drop it.

        Parameters:
            key : The name of the attribute.
        """
        if not hasattr(type(self), key):
            return False
        value, default = getattr(self, key), getattr(type(self), key)
        return value is default or (
            type(value) is type(default) and value == default)

    def __repr__(self):
        """
        The transparent representation of a DisCoPy object: an attribute
        without a class default is positional, one that differs from its
        default is a keyword argument and one equal to it is dropped.

        Example
        -------
        >>> import discopy
        >>> from discopy.cat import Box
        >>> f = Box('f', 'x', 'y', data=42)
        >>> f
        cat.Box('f', cat.Ob('x'), cat.Ob('y'), data=42)
        >>> assert eval(repr(f), vars(discopy)) == f
        """
        return factory_name(type(self)) + "(" + ", ".join(
            f"{key}={repr(getattr(self, key))}" if hasattr(type(self), key)
            else repr(getattr(self, key))
            for key in self.serialised_attrs if not self.is_default(key)) + ")"

    def __setstate__(self, state):
        """
        Restore a pickled state, the terminal that every pickle
        migration shim chains into with ``super().__setstate__``.

        Parameters:
            state : The pickled state of the object.
        """
        self.__dict__.update(state)

    def to_tree(self) -> dict:
        """
        Serialise a DisCoPy object, see :func:`dumps`.

        The tree records the :func:`factory_name` and then each of the
        ``serialised_attrs``, dropping a key when its value equals the
        class attribute of the same name, e.g. a box that is not a
        dagger. An attribute with a ``to_tree`` method is serialised, a
        non-empty list or tuple of such attributes becomes the list of
        their trees, raw JSON data passes through unchanged.

        Example
        -------
        >>> from pprint import PrettyPrinter
        >>> pprint = PrettyPrinter(indent=4, width=70, sort_dicts=False).pprint
        >>> from discopy.cat import Box
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
        tree = {'factory': factory_name(type(self))}
        for key in self.serialised_attrs:
            if self.is_default(key):
                continue
            value = getattr(self, key)
            if hasattr(value, 'to_tree'):
                value = value.to_tree()
            elif isinstance(value, (list, tuple)) and value and all(
                    hasattr(v, 'to_tree') for v in value):
                value = [v.to_tree() for v in value]
            tree[key] = value
        return tree

    @classmethod
    def from_tree(cls, tree: dict) -> Serialisable:
        """
        Decode a serialised DisCoPy object, see :func:`loads`.

        A key missing from the tree falls back to the default value of
        the corresponding keyword argument of ``__init__``. A value with
        a ``'factory'`` key decodes recursively, a non-empty list of
        such values to the tuple of decoded objects, raw JSON data
        passes through unchanged.

        Parameters:
            tree : DisCoPy serialisation.

        Example
        -------
        >>> from discopy.cat import Ob
        >>> assert Ob.from_tree({'factory': 'cat.Ob', 'name': 'x'}) == Ob('x')
        """
        from discopy.utils import from_tree

        kwargs = {}
        for key in cls.serialised_attrs:
            if key not in tree:
                continue
            value = tree[key]
            if isinstance(value, dict) and 'factory' in value:
                value = from_tree(value)
            elif isinstance(value, list) and value and all(
                    isinstance(v, dict) and 'factory' in v for v in value):
                value = tuple(map(from_tree, value))
            kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def environment(cls) -> dict:
        """
        The namespace the representation of a term reads back in: the
        public names of the package, as ``from discopy import *`` binds
        them, so that a representation qualified by module such as
        ``cat.Box('f', cat.Ob('x'), cat.Ob('y'))`` evaluates, and then
        those of the module the class is defined in, so that one
        printing bare names such as ``Tensor[int]([0], dom=Dim(1),
        cod=Dim(1))`` evaluates too. The module comes second because a
        term prints the names its own module binds: ``Dim`` in
        ``discopy.tensor`` is the one a tensor is built from.

        The import is local because the package imports this module.
        """
        import discopy

        public = lambda namespace: {
            name: value for name, value in namespace.items()
            if not name.startswith("_")}
        module = sys.modules[cls.__module__]
        return dict(public(vars(discopy)), **public(vars(module)))

    @axiom
    def transparency(cls, term: Self) -> Equation:
        """
        The representation of a term evaluates back to it, in the
        :meth:`environment` of its type.
        """
        return Equation(eval(repr(term), cls.environment()), term)

    @axiom
    def pickling(cls, term: Self) -> Equation:
        """
        A term loads back from its pickle, of the same class: the equation
        is between the pairs of a class and a term, since a subscript of a
        :class:`discopy.utils.NamedGeneric` is part of what a pickle keeps.
        """
        loaded = pickle.loads(pickle.dumps(term))
        return Equation((type(loaded), loaded), (type(term), term))

    @axiom
    def copying(cls, term: Self) -> Equation:
        """
        A term is equal to its deep copy, of the same class. Copying goes
        through the same protocol as :meth:`pickling` without the bytes,
        so a class whose reduction drops what its state needs — the
        parameters of a :class:`discopy.utils.NamedGeneric`, say — breaks
        one law with the other.
        """
        copied = deepcopy(term)
        return Equation((type(copied), copied), (type(term), term))

    @axiom
    def serialisation(cls, term: Self) -> Equation:
        """
        A term decodes back from its tree and from the JSON of its tree.
        A type without a tree declares the law inapplicable.
        """
        from discopy.utils import dumps, from_tree, loads

        return Equation(from_tree(term.to_tree()), loads(dumps(term)), term)


class Category[C0, C1: Category](Theory, ABC):
    """
    A category is a class with two class variables ``ob, ar``, two attributes
    ``dom, cod`` and two methods ``id, then``.

    This base class also implements syntactic sugar :code:`>>` and :code:`<<`
    for forward and backward composition with the method :code:`then`.

    Example
    -------
    >>> class List(list, Category):
    ...     ob, dom, cod = type(None), None, None
    ...     def then(self, other):
    ...         return self + other
    >>> assert List([1, 2]) >> List([3]) == List([1, 2, 3])
    >>> assert List([3]) << List([1, 2]) == List([1, 2, 3])
    """
    ob: ClassVar[type[C0]]
    factory: ClassVar[type[C1]]
    dom: C0
    cod: C0

    axioms = no_strategy

    #: Backward-compatible alias for :attr:`factory`, since types are
    #: themselves the objects of diagrams.
    ar = classproperty(lambda cls: getattr(cls, "factory", cls))

    @classmethod
    def equation_factory(cls, *terms) -> Equation:
        """
        Construct an equation, using strict equality by default.

        A class that quotients its equations overrides this, e.g. by
        hypergraph isomorphism from symmetric categories on, so an axiom
        built with it is checked up to whatever quotient the category
        defines — and :meth:`discopy.axioms.Axiom.modulo` weakens it
        further.
        """
        return Equation(*terms)

    @classmethod
    @abstractmethod
    def id(cls, dom: C0) -> C1:
        """
        Identity morphism on an object :code:`dom: C0`, to be instantiated.

        Parameters:
            dom (C0) : The domain of an identity is also its codomain.
        """

    @abstractmethod
    def then(self, *others: C1) -> C1:
        """
        Sequential composition of `n >= 1` morphisms, to be instantiated.

        Parameters:
            other : The other morphism to compose sequentially.
        """

    def is_composable(self, other: C1) -> bool:
        """
        Whether two morphisms are composable, i.e. the codomain of the first is
        the domain of the second.

        Parameters:
            other : The other morphism.
        """
        return self.cod == other.dom

    def is_parallel(self, other: Category) -> bool:
        """
        Whether two morphisms are parallel, i.e. they have the same
        domain and codomain.

        Parameters:
            other : The other morphism.
        """
        return (self.dom, self.cod) == (other.dom, other.cod)

    @axiom
    def unitality(
            cls, f: C1) -> Equation[C1]:
        """ Left and right unitality of composition. """
        return cls.equation_factory(
            cls.id(f.dom).then(f), f, f.then(cls.id(f.cod)))

    @axiom
    def associativity(
            cls, triple: ComposableTriple[C1]) -> Equation[C1]:
        """ Associativity of composition. """
        f, g, h = triple
        return cls.equation_factory(
            f.then(g).then(h), f.then(g.then(h)))

    @axiom
    def identity_typing(
            cls, x: C0) -> Equation[C0]:
        """ Typing of identity morphisms. """
        identity = cls.id(x)
        return cls.ob.equation_factory(identity.dom, x, identity.cod)

    @axiom
    def composition_dom_typing(
            cls, pair: ComposablePair[C1]) -> Equation[C0]:
        """ Domain typing of composition. """
        f, g = pair
        return cls.ob.equation_factory(f.then(g).dom, f.dom)

    @axiom
    def composition_cod_typing(
            cls, pair: ComposablePair[C1]) -> Equation[C0]:
        """ Codomain typing of composition. """
        f, g = pair
        return cls.ob.equation_factory(f.then(g).cod, g.cod)

    @axiom
    def dagger_involution(
            cls, f: C1) -> Equation[C1]:
        """ The dagger is involutive. """
        return cls.equation_factory(f.dagger().dagger(), f)

    @axiom
    def dagger_contravariance(
            cls, pair: ComposablePair[C1]) -> Equation[C1]:
        """ The dagger reverses composition. """
        f, g = pair
        return cls.equation_factory(
            f.then(g).dagger(), g.dagger().then(f.dagger()))

    __rshift__ = __llshift__ = lambda self, other: self.then(other)
    __lshift__ = __lrshift__ = lambda self, other: other.then(self)


class ColouredMonoid[C0, C1: ColouredMonoid](Category[C0, C1]):
    """
    A coloured monoid is a category whose sequential composition ``then`` is
    given by a monoidal ``tensor``, with the objects ``C0`` (its colours) as
    the boundaries of its morphisms.

    An ordinary :obj:`Monoid` is the special case with a single, trivial
    colour, i.e. :class:`type(None)`. We do not enforce this so
    that e.g. :class:`monoidal.Ty` can take colours as objects.
    """
    @classmethod
    def id(cls, dom: C0 = None) -> C1:
        """The monoidal unit, i.e. the empty tensor ``cls()``."""
        return cls()

    @classmethod
    def unit(cls, colour: C0 = None) -> C0 | C1:
        """
        The unit at a colour, i.e. the identity on it.

        It need not be an element of the monoid, which is why it may land in
        ``C0``: the layers of :class:`monoidal.Layer` are closed under
        ``tensor`` but the empty one is a type rather than a layer.
        """
        return cls.id(colour)

    @abstractmethod
    def tensor(self, *objects: C1) -> C1:
        """ The n-ary product of a monoid for ``n > 0``. """

    def then(self, *others: C1) -> C1:
        """Sequential composition, given by the monoid product."""
        return self.tensor(*others)

    @classmethod
    def whisker(cls, other: C0 | C1) -> C1:
        """
        Do nothing if ``other`` is already a morphism else apply :meth:`id`.

        Parameters:
            other : The object or morphism to be tensored on the left or right.
        """
        return other if isinstance(other, cls) else cls.id(other)

    def __matmul__(self, other):
        return self.tensor(other)

    def __rmatmul__(self, other):
        return self.whisker(other).tensor(self)


class Monoid[C1: Monoid](ColouredMonoid[type(None), C1]):
    """ A monoid is a coloured monoid with a single, trivial colour. """


@dataclass
class Nat(Monoid["Nat"]):
    """
    ``Nat`` is the free monoid on one generator, i.e. the natural numbers
    with addition as tensor. It is also a sequence over its unary encoding:
    :meth:`__len__` gives back the natural number itself and slicing reads
    it off as a sequence of ``1``'s, e.g. ``Nat(3)[:1] == Nat(1)``.

    Parameters:
        n : The natural number.
    """
    n: int = 0

    def tensor(self, *others: Nat) -> Nat:
        if any(not isinstance(other, Nat) for other in others):
            return NotImplemented  # This allows whiskering on the left.
        return type(self)(self.n + sum(other.n for other in others))

    def __len__(self) -> int:
        return self.n

    def __index__(self) -> int:
        return self.n

    def __getitem__(self, key: int | slice) -> Nat:
        """
        Slicing a natural number reads it off as a sequence of ``1``'s.

        Parameters:
            key : An integer or a slice.
        """
        if isinstance(key, slice):
            return type(self)(len(range(self.n)[key]))
        if key >= self.n or key < -self.n:
            raise IndexError
        return type(self)(1)


class MonoidalCategory[C0: ColouredMonoid, C1: MonoidalCategory](
        Category[C0, C1]):
    """
    A monoidal category is a :class:`Category` with a method :code:`tensor` for
    both its objects and its morphisms.

    This base class also implements syntactic sugar :code:`@` for whiskering.
    """

    @classmethod
    @abstractmethod
    def tensor(cls, *morphisms: C1) -> C1:
        """
        Parallel composition of ``n >= 0`` morphisms, to be instantiated.

        Parameters:
            other : The other morphism to compose in parallel.
        """

    @classmethod
    def whisker(cls, other: C0 | C1) -> C1:
        """
        Do nothing if ``other`` is already a morphism else apply :meth:`id`.

        Parameters:
            other : The object or morphism to be tensored on the left or right.
        """
        return other if isinstance(other, MonoidalCategory) else cls.id(other)

    def __matmul__(self, other):
        return self.tensor(self.whisker(other))

    def __rmatmul__(self, other):
        return self.whisker(other).tensor(self)


class PRO[C1: PRO](MonoidalCategory[Nat, C1]):
    """
    A PRO is a :class:`MonoidalCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free monoidal category on one generator.
    """


class TracedCategory[C0, C1](MonoidalCategory[C0, C1]):
    """
    A traced category is a :class:`MonoidalCategory` with a method
    :code:`trace` for the partial trace of a morphism over some objects.
    """
    @abstractmethod
    def trace(self, n: int = 1, left: bool = False) -> C1:
        """
        The trace of a morphism, to be instantiated.

        Tracing no object at all is the identity, i.e. the vanishing axiom
        ``f.trace(0) == f``, see `nLab
        <https://ncatlab.org/nlab/show/traced+monoidal+category>`_.

        Parameters:
            n : The number of objects to trace over.
            left : Whether to trace the wires on the left or right.
        """


class ResiduatedMonoid[C0, C1: ResiduatedMonoid](ColouredMonoid[C0, C1]):
    """
    A monoid is residuated when it comes with methods ``over`` and ``under``
    with syntactic sugar ``<<`` and ``>>``.
    """
    @abstractmethod
    def over(self, other: C1) -> C1:
        """ The right-to-left exponential object ``self`` to the ``other``. """

    @abstractmethod
    def under(self, other: C1) -> C1:
        """ The left-to-right exponential object ``self`` to the ``other``. """

    def __lshift__(self, other):
        return self.over(other)

    def __rshift__(self, other):
        return other.under(self)


class BiclosedCategory[
        C0: ResiduatedMonoid, C1: BiclosedCategory](MonoidalCategory[C0, C1]):
    """
    A biclosed category is a :class:`MonoidalCategory` with methods :code:`ev`
    and :code:`curry` for the evaluation and currying of morphisms.

    We also assume the type for objects comes with methods for left and right
    exponentials :code`x << y` and :code`x >> y`.
    """
    @classmethod
    @abstractmethod
    def ev(cls, base: C0, exponent: C0, left: bool = True) -> C1:
        """
        The evaluation of an exponential type, to be instantiated.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """

    @abstractmethod
    def curry(self, n: int = 1, left: bool = True) -> C1:
        """
        The currying of a morphism, to be instantiated.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.
        """

    def base_and_exponent(self, n: int, left: bool) -> tuple[C0, C0]:
        """
        The base and exponent that :meth:`uncurry` evaluates, read off the
        exponential object in the codomain.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        if not self.cod.is_exp:
            raise ValueError
        base, exponent = self.cod.base, self.cod.exponent
        if n < len(exponent):
            raise ValueError
        return base, exponent

    def uncurry(self, n: int = 1, left: bool = True) -> C1:
        """
        Uncurry a morphism by composing it with :meth:`ev`, assuming its
        codomain is an exponential object. If the exponent has less than
        ``n`` objects, we uncurry the remaining ones in turn.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        if n < 0:
            raise ValueError
        if not n:
            return self
        base, exponent = self.base_and_exponent(n, left)
        result = self @ exponent >> self.ev(base, exponent, True) if left\
            else exponent @ self >> self.ev(base, exponent, False)
        return result.uncurry(n - len(exponent), left)


class Pregroup[C0, C1: Pregroup](ResiduatedMonoid[C0, C1]):
    """
    A pregroup is a residuated monoid where the left and right exponentials are
    given by tensoring with the chosen left and right duals for each object.
    """
    l: C1
    r: C1

    def over(self, other: C1) -> C1:
        return self @ other.l

    def under(self, other: C1) -> C1:
        return other.r @ self


class RigidCategory[C0: Pregroup, C1: RigidCategory](BiclosedCategory[C0, C1]):
    """
    A rigid category is a :class:`BiclosedCategory` with a :class:`Pregroup` as
    object type and methods for :code:`cups` and :code:`caps`.
    """
    @classmethod
    @abstractmethod
    def cups(cls, left: C0, right: C0) -> C1:
        """
        The cups witnessing :code:`right` as the adjoint of :code:`left`.

        Parameters:
            left : The left-hand side of the cups.
            right : Its adjoint, i.e. the right-hand side of the cups.
        """

    @classmethod
    @abstractmethod
    def caps(cls, left: C0, right: C0) -> C1:
        """
        The caps witnessing :code:`right` as the adjoint of :code:`left`.

        Parameters:
            left : The left-hand side of the caps.
            right : Its adjoint, i.e. the right-hand side of the caps.
        """

    @classmethod
    def ev(cls, base: C0, exponent: C0, left: bool = True) -> C1:
        """
        The evaluation of a rigid morphism is obtained using cups.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """
        return base @ cls.cups(exponent.l, exponent) if left\
            else cls.cups(exponent, exponent.r) @ base

    def curry(self, n: int = 1, left: bool = True) -> C1:
        """
        The curry of a rigid morphism is obtained using caps.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.
        """
        if n < 0 or n > len(self.dom):
            raise ValueError
        if not n:
            return self
        if left:
            base, exponent = self.dom[:-n], self.dom[-n:]
            return base @ self.caps(exponent, exponent.l) >> self @ exponent.l
        base, exponent = self.dom[n:], self.dom[:n]
        return self.caps(exponent.r, exponent) @ base >> exponent.r @ self

    def base_and_exponent(self, n: int, left: bool) -> tuple[C0, C0]:
        """
        Contrary to :meth:`BiclosedCategory.base_and_exponent`, a pregroup has
        no exponential object to read the exponent off the codomain: it is the
        ``n`` objects at the end resp. the start of the codomain, dualised.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        if n > len(self.cod):
            raise ValueError
        return (self.cod[:-n], self.cod[-n:].r) if left\
            else (self.cod[n:], self.cod[:n].l)

    def transpose(self, left: bool = False) -> C1:
        """
        The transpose of a morphism, i.e. its composition with cups and caps.

        Parameters:
            left : Whether to transpose left or right.

        Example
        -------
        >>> from discopy.monoidal import Equation
        >>> from discopy.rigid import Ty, Box
        >>> x, y = map(Ty, "xy")
        >>> f = Box('f', x, y)
        >>> Equation(f.transpose(left=True), f, f.transpose(),
        ...     symbols=("$\\\\mapsfrom$", "$\\\\mapsto$")).draw(
        ...         figsize=(8, 3), doctest="docs/_static/rigid/transpose.svg")

        .. image:: /_static/rigid/transpose.svg
        """
        if left:
            return self.cod.l @ self.caps(self.dom, self.dom.l)\
                >> self.cod.l @ self @ self.dom.l\
                >> self.cups(self.cod.l, self.cod) @ self.dom.l
        return self.caps(self.dom.r, self.dom) @ self.cod.r\
            >> self.dom.r @ self @ self.cod.r\
            >> self.dom.r @ self.cups(self.cod, self.cod.r)


class PivotalCategory[C0, C1](RigidCategory[C0, C1], TracedCategory[C0, C1]):
    """
    A pivotal category is a :class:`RigidCategory` where the left and right
    adjoints coincide, hence it is also a :class:`TracedCategory`.
    """


class BraidedCategory[C0, C1](MonoidalCategory[C0, C1]):
    """
    A braided category is a :class:`MonoidalCategory` with a method
    :code:`braid` for the natural isomorphism :code:`x @ y -> y @ x`.
    """
    @classmethod
    @abstractmethod
    def braid(cls, left: C0, right: C0) -> C1:
        """
        The braid of two objects, to be instantiated.

        Parameters:
            left : The object on the left of the braid.
            right : The object on the right of the braid.
        """


class PROB[C1: PROB](PRO[C1], BraidedCategory[Nat, C1]):
    """
    A PROB is a :class:`BraidedCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free braided category on one generator.
    """


class SymmetricCategory[C0, C1](BraidedCategory[C0, C1]):
    """
    A symmetric category is a :class:`BraidedCategory` where the braid is its
    own inverse called :code:`swap` for the symmetry :code:`x @ y -> y @ x`.
    """
    @classmethod
    @abstractmethod
    def swap(cls, left: C0, right: C0) -> C1:
        """
        The swap of two objects, to be instantiated.

        Parameters:
            left : The object on the left of the swap.
            right : The object on the right of the swap.
        """

    @classmethod
    def permutation(cls, xs: Sequence[int], doms: Sequence[C0]) -> C1:
        """ Compose swaps to permute the atomic objects in ``dom``. """
        xs, doms = list(xs), list(doms)
        if list(range(len(doms))) != sorted(xs):
            raise ValueError
        tensor = lambda objects: sum(objects, start=cls.ob())
        result, done = cls.id(tensor(doms)), cls.ob()
        while xs != list(range(len(xs))):
            i = xs[0]
            left, head = tensor(doms[:i]), tensor(doms[i:i + 1])
            result >>= done @ cls.swap(left, head) @ tensor(doms[i + 1:])
            done, doms = done @ head, doms[:i] + doms[i + 1:]
            xs = [x - 1 if x > i else x for x in xs[1:]]
        return result

    @classmethod
    def braid(cls, left: C0, right: C0) -> C1:
        return cls.swap(left, right)


class PROP[C1: PROP](PROB[C1], SymmetricCategory[Nat, C1]):
    """
    A PROP is a :class:`SymmetricCategory` whose objects are the natural
    numbers :class:`Nat`, i.e. the free symmetric category on one generator.
    """


class MarkovCategory[C0, C1](SymmetricCategory[C0, C1]):
    """
    A Markov category is a :class:`SymmetricCategory` with methods
    :code:`copy` and :code:`merge` for the supply of commutative comonoids.
    """
    @classmethod
    @abstractmethod
    def copy(cls, x: C0, n: int = 2) -> C1:
        """
        Make :code:`n` copies of a given object :code:`x`.

        Parameters:
            x : The object to copy.
            n : The number of copies.
        """


class ClosedCategory[C0, C1](BiclosedCategory[C0, C1], MarkovCategory[C0, C1]):
    """
    A closed category is a symmetric :class:`BiclosedCategory`. We also assume
    it comes with copy and discard so it is also a :class:`MarkovCategory`.
    """


class FeedbackCategory[C0, C1](MarkovCategory[C0, C1]):
    """
    A feedback category is a :class:`MarkovCategory` with a :code:`delay`
    endofunctor and a :code:`feedback` operator.
    """
    @abstractmethod
    def delay(self, n_steps: int = 1) -> C1:
        """
        The delay endofunctor applied to a morphism.

        Parameters:
            n_steps : The number of time steps to delay.
        """

    @abstractmethod
    def feedback(self, dom: C0, cod: C0, mem: C0) -> C1:
        """
        The feedback operator on a morphism.

        Parameters:
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory type to trace over.
        """


class BalancedCategory[C0, C1](
        BraidedCategory[C0, C1], TracedCategory[C0, C1]):
    """
    A balanced category is a :class:`BraidedCategory` and a
    :class:`TracedCategory` with a method :code:`twist` for the natural
    automorphism :code:`x -> x`.
    """
    @classmethod
    @abstractmethod
    def twist(cls, dom: C0) -> C1:
        """
        The twist on an object, to be instantiated.

        Parameters:
            dom : The object on which to take the twist.
        """


class RibbonCategory[C0, C1](
        PivotalCategory[C0, C1], BalancedCategory[C0, C1]):
    """
    A ribbon category is a :class:`PivotalCategory` which is also a
    :class:`BalancedCategory`, i.e. where diagrams can draw knots and links.
    """


class CompactCategory[C0, C1](
        RibbonCategory[C0, C1], SymmetricCategory[C0, C1]):
    """
    A compact category is a :class:`RibbonCategory` which is also a
    :class:`SymmetricCategory`, i.e. with cups, caps and swaps and where
    the twist is the identity.
    """
    @classmethod
    def twist(cls, dom: C0) -> C1:
        return cls.id(dom)


class HypergraphCategory[C0, C1](
        CompactCategory[C0, C1], MarkovCategory[C0, C1]):
    """
    A hypergraph category is a symmetric category with a supply of spiders,
    i.e. special commutative Frobenius algebras on each objects.

    This makes it both a :class:`CompactCategory` and a :class:`MarkovCategory`
    """
    @classmethod
    @abstractmethod
    def spiders(cls, n_legs_in: int, n_legs_out: int, typ: C0) -> C1:
        """
        The spiders on a given type with ``n_legs_in`` and ``n_legs_out``.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
        """
