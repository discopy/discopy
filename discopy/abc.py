# -*- coding: utf-8 -*-

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

Each class also declares its :func:`discopy.testing.axiom` equations, which
every free category inherits along with the structure they axiomatise.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Category
    MonoidalCategory
    BraidedCategory
    TracedCategory
    BalancedCategory
    SymmetricCategory
    MarkovCategory
    FeedbackCategory
    ClosedCategory
    RigidCategory
    PivotalCategory
    RibbonCategory
    NamedGeneric
    Equation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Callable, ClassVar, Generic, TypeVar

from discopy.testing import (
    Atomic, Axiom, Bifunctor, ComposablePair, ComposableTriple,
    FeedbackJoining, FeedbackVanishing, HorizontalPair,
    LeftCurrying, Natural, NonEmpty, RightCurrying, TraceDinaturalityLeft,
    TraceDinaturalityRight, TraceNaturalityLeft, TraceNaturalityRight,
    TraceSuperposing, axiom)
from discopy.utils import classproperty, factory_name, get_origin


class Category[C0, C1: Category](ABC):
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

    #: Backward-compatible alias for :attr:`factory`, since types are
    #: themselves the objects of diagrams.
    ar = classproperty(lambda cls: getattr(cls, "factory", cls))

    @classmethod
    def equation_factory(cls, *terms) -> Equation:
        """
        Construct an equation, using strict equality by default.

        A class that quotients its equations overrides this, e.g. by
        hypergraph isomorphism from symmetric categories on, so an axiom
        built with it is checked up to whatever quotient the carrier
        defines — and :meth:`discopy.testing.Axiom.modulo` weakens it
        further.
        """
        return Equation(*terms)

    @classproperty
    def axioms(cls) -> dict[str, Axiom]:
        """
        The axioms inherited by ``cls``, by name, subclasses overriding
        bases.

        Names are collected before they are filtered, so that assigning
        anything that is not an axiom over an inherited one drops it
        altogether, rather than restating it.
        """
        visible = {
            name: value
            for base in reversed(cls.__mro__)
            for name, value in base.__dict__.items()}
        return {name: value.bind(cls) for name, value in visible.items()
                if isinstance(value, Axiom)}

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

    @axiom
    def monoid_unitality(
            cls, x: C1) -> Equation[C1]:
        """ Unitality of a monoid. """
        return cls.equation_factory(cls.unit() @ x, x, x @ cls.unit())

    @axiom
    def monoid_associativity(
            cls, triple: ComposableTriple[C1]) -> Equation[C1]:
        """ Associativity of a monoid. """
        x, y, z = triple
        return cls.equation_factory(x @ (y @ z), (x @ y) @ z)

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


# A monoid is a coloured monoid with a single, trivial colour.
type Monoid[C1: ColouredMonoid] = ColouredMonoid[type(None), C1]


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

    @axiom
    def bifunctoriality(
            cls, square: Bifunctor[C1]) -> Equation[C1]:
        """ Bifunctoriality of the tensor. """
        f, g, h, k = square
        return cls.equation_factory(
            f @ g >> h @ k, (f >> h) @ (g >> k))

    @axiom
    def tensor_unitality(
            cls, pair: HorizontalPair[C1]) -> Equation[C1]:
        """ Preservation of identities by tensor. """
        x, y = (cell.dom for cell in pair)
        return cls.equation_factory(
            cls.id(x) @ cls.id(y), cls.id(x @ y))

    @axiom
    def tensor_dom_typing(
            cls, pair: HorizontalPair[C1]) -> Equation[C0]:
        """ Domain typing of tensor. """
        f, g = pair
        return cls.ob.equation_factory((f @ g).dom, f.dom @ g.dom)

    @axiom
    def tensor_cod_typing(
            cls, pair: HorizontalPair[C1]) -> Equation[C0]:
        """ Codomain typing of tensor. """
        f, g = pair
        return cls.ob.equation_factory((f @ g).cod, f.cod @ g.cod)

    @axiom
    def dagger_monoidality(
            cls, pair: HorizontalPair[C1]) -> Equation[C1]:
        """ The dagger distributes over the tensor. """
        f, g = pair
        return cls.equation_factory(
            (f @ g).dagger(), f.dagger() @ g.dagger())


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

    @axiom
    def trace_vanishing(
            cls, f: C1) -> Equation[C1]:
        """ Vanishing of a trace over the unit. """
        return cls.equation_factory(
            f.trace(0), f, f.trace(0, left=True))

    @axiom
    def trace_superposing_left(
            cls, pair: TraceSuperposing[C0, C1]) -> Equation[C1]:
        """ Left-oriented superposing. """
        f, obj = pair
        return cls.equation_factory(
            (f @ obj).trace(left=True), f.trace(left=True) @ obj)

    @axiom
    def trace_superposing_right(
            cls, pair: TraceSuperposing[C0, C1]) -> Equation[C1]:
        """ Right-oriented superposing. """
        f, obj = pair
        return cls.equation_factory(
            (obj @ f).trace(), obj @ f.trace())

    @axiom
    def trace_naturality_left(
            cls, sliding: TraceNaturalityLeft[C0, C1]) -> Equation[C1]:
        """ Left-oriented trace naturality. """
        f, x, g = sliding
        return cls.equation_factory(
            (x @ g).then(f).then(x @ g).trace(len(x), left=True),
            g.then(f.trace(len(x), left=True)).then(g))

    @axiom
    def trace_naturality_right(
            cls, sliding: TraceNaturalityRight[C0, C1]) -> Equation[C1]:
        """ Right-oriented trace naturality. """
        f, x, g = sliding
        return cls.equation_factory(
            (g @ x).then(f).then(g @ x).trace(len(x)),
            g.then(f.trace(len(x))).then(g))

    @axiom
    def trace_dinaturality_left(
            cls, sliding: TraceDinaturalityLeft[C0, C1]) -> Equation[C1]:
        """ Left-oriented trace dinaturality. """
        f, g = sliding
        source, target = g.cod, g.dom
        base, cobase = f.dom[len(source):], f.cod[len(target):]
        return cls.equation_factory(
            f.then(g @ cobase).trace(len(source), left=True),
            (g @ base).then(f).trace(len(target), left=True))

    @axiom
    def trace_dinaturality_right(
            cls, sliding: TraceDinaturalityRight[C0, C1]) -> Equation[C1]:
        """ Right-oriented trace dinaturality. """
        f, g = sliding
        source, target = g.cod, g.dom
        base = f.dom[:-len(source)] if len(source) else f.dom
        cobase = f.cod[:-len(target)] if len(target) else f.cod
        return cls.equation_factory(
            f.then(cobase @ g).trace(len(source)),
            (base @ g).then(f).trace(len(target)))


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

    @axiom
    def currying_left(
            cls, arguments: LeftCurrying[C0, C1]) -> Equation[C1]:
        """ Left currying followed by evaluation. """
        f, base, exponent = arguments
        return cls.equation_factory(
            cls._uncurry(f, base, exponent, left=True), f)

    @axiom
    def currying_right(
            cls, arguments: RightCurrying[C0, C1]) -> Equation[C1]:
        """ Right currying followed by evaluation. """
        f, base, exponent = arguments
        return cls.equation_factory(
            cls._uncurry(f, base, exponent, left=False), f)

    @classmethod
    def _uncurry(
            cls, f: C1, base: C0, exponent: C0, left: bool):
        curried = f.curry(left=left)
        ev = cls.ev(base, exponent, left)
        return (curried @ exponent).then(ev) if left\
            else (exponent @ curried).then(ev)

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

    @axiom
    def adjunction(
            cls, x: C1) -> Equation[C1]:
        """ The left and right adjoints are mutually inverse. """
        return cls.equation_factory(x.l.r, x, x.r.l)


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

    @axiom
    def snake_equations(
            cls, x: C0) -> Equation[C1]:
        """ The two snake equations. """
        snake_r = (cls.id(x) @ cls.caps(x.r, x)).then(
            cls.cups(x, x.r) @ cls.id(x))
        snake_l = (cls.caps(x, x.l) @ cls.id(x)).then(
            cls.id(x) @ cls.cups(x.l, x))
        return cls.equation_factory(snake_r, cls.id(x), snake_l)

    @axiom
    def caps_coherence(
            cls, x: NonEmpty[C0],
            y: NonEmpty[C0]) -> Equation[C1]:
        """ Monoidal coherence of caps. """
        x, y = x.value, y.value
        return cls.equation_factory(
            cls.caps(x @ y, (x @ y).l),
            cls.caps(x, x.l).then(x @ cls.caps(y, y.l) @ x.l))

    @axiom
    def rotate_contravariance(
            cls, pair: ComposablePair[C1]) -> Equation[C1]:
        """ Rotation reverses composition. """
        f, g = pair
        return cls.equation_factory(
            f.then(g).rotate(), g.rotate().then(f.rotate()))

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
    @axiom
    def self_dual(
            cls, x: C0) -> Equation[C0]:
        """ Equality of left and right adjoints. """
        return cls.ob.equation_factory(x.r, x.l)

    @axiom
    def pivotality(
            cls, f: C1) -> Equation[C1]:
        """ Equality of left and right transposes. """
        dom, cod = f.dom, f.cod
        left_transpose = (cod.l @ cls.caps(dom, dom.l)).then(
            cod.l @ f @ dom.l).then(cls.cups(cod.l, cod) @ dom.l)
        right_transpose = (cls.caps(dom.r, dom) @ cod.r).then(
            dom.r @ f @ cod.r).then(dom.r @ cls.cups(cod, cod.r))
        return cls.equation_factory(left_transpose, right_transpose)


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

    @axiom
    def hexagon_left(
            cls, x: Atomic[C0],
            y: Atomic[C0],
            z: Atomic[C0]) -> Equation[C1]:
        """ The left hexagon equation. """
        x, y, z = x.value, y.value, z.value
        return cls.equation_factory(
            cls.braid(x, y @ z),
            (cls.braid(x, y) @ z).then(y @ cls.braid(x, z)))

    @axiom
    def hexagon_right(
            cls, x: Atomic[C0],
            y: Atomic[C0],
            z: Atomic[C0]) -> Equation[C1]:
        """ The right hexagon equation. """
        x, y, z = x.value, y.value, z.value
        return cls.equation_factory(
            cls.braid(x @ y, z),
            (x @ cls.braid(y, z)).then(cls.braid(x, z) @ y))

    @axiom
    def braid_naturality(
            cls, f: C1, g: C1) -> Equation[C1]:
        """ Naturality of the braid. """
        return cls.equation_factory(
            f @ g >> cls.braid(f.cod, g.cod),
            cls.braid(f.dom, g.dom) >> g @ f,
        )


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

    @axiom
    def swap_inverse(
            cls, x: C0, y: C0) -> Equation[C1]:
        """ Involutivity of the swap. """
        return cls.equation_factory(
            cls.swap(x, y).then(cls.swap(y, x)), cls.id(x @ y))


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

    @axiom
    def copy_counitality(
            cls, x: C0) -> Equation[C1]:
        """ Counitality of copying. """
        copy, discard = cls.copy(x), cls.copy(x, n=0)
        return cls.equation_factory(
            copy.then(discard @ x), cls.id(x),
            copy.then(x @ discard))

    @axiom
    def copy_coassociativity(
            cls, x: C0) -> Equation[C1]:
        """ Coassociativity of copying. """
        copy = cls.copy(x)
        return cls.equation_factory(
            copy.then(copy @ x), copy.then(x @ copy))

    @axiom
    def copy_cocommutativity(
            cls, x: C0) -> Equation[C1]:
        """ Cocommutativity of copying. """
        copy = cls.copy(x)
        return cls.equation_factory(copy.then(cls.swap(x, x)), copy)

    @axiom
    def discard_coherence(
            cls, x: C0) -> Equation[C1]:
        """ Monoidal coherence of discarding. """
        return cls.equation_factory(
            cls.copy(x @ x, n=0),
            cls.copy(x, n=0) @ cls.copy(x, n=0))

    @axiom
    def copy_monoidal_coherence(
            cls, x: C0) -> Equation[C1]:
        """ Monoidal coherence of copying. """
        return cls.equation_factory(
            cls.copy(x @ x),
            (cls.copy(x) @ cls.copy(x)).then(
                x @ cls.swap(x, x) @ x))


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
    def feedback(
            self, dom: C0 = None, cod: C0 = None, mem: C0 = None) -> C1:
        """
        The feedback operator on a morphism.

        Parameters:
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory type to trace over.
        """

    @axiom
    def feedback_vanishing(
            cls, arguments: FeedbackVanishing[C0, C1]) -> Equation[C1]:
        """ Vanishing of feedback over the unit. """
        f, unit = arguments
        return cls.equation_factory(f.feedback(mem=unit), f)

    dagger_involution = Category.dagger_involution.inapplicable(
        "The delay of a feedback category is not reversible.")

    dagger_contravariance = Category.dagger_contravariance.inapplicable(
        "The delay of a feedback category is not reversible.")

    dagger_monoidality = MonoidalCategory.dagger_monoidality.inapplicable(
        "The delay of a feedback category is not reversible.")

    @axiom
    def feedback_joining(
            cls, arguments: FeedbackJoining[C0, C1]) -> Equation[C1]:
        """ Joining nested feedback loops. """
        f, mem = arguments
        return cls.equation_factory(
            f.feedback(mem=mem), f.feedback().feedback())


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

    @axiom
    def balanced_twist(
            cls, x: Atomic[C0],
            y: Atomic[C0]) -> Equation[C1]:
        """ Compatibility of the twist and braid. """
        x, y = x.value, y.value
        return cls.equation_factory(
            cls.twist(x @ y),
            cls.braid(x, y).then(
                cls.twist(y) @ cls.twist(x)).then(
                    cls.braid(y, x)))


class RibbonCategory[C0, C1](
        PivotalCategory[C0, C1], BalancedCategory[C0, C1]):
    """
    A ribbon category is a :class:`PivotalCategory` which is also a
    :class:`BalancedCategory`, i.e. where diagrams can draw knots and links.
    """
    @axiom
    def twist_as_trace(
            cls, x: Atomic[C0]) -> Equation[C1]:
        """ The twist as both orientations of a traced braid. """
        x = x.value
        braid = cls.braid(x, x)
        return cls.equation_factory(
            braid.trace(left=True), cls.twist(x), braid.trace())


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

    @axiom
    def reidemeister_1_cap(
            cls, x: C0) -> Equation[C1]:
        """ Reidemeister move 1 for caps. """
        return cls.equation_factory(
            cls.caps(x, x.r).then(cls.swap(x, x.r)),
            cls.caps(x.r, x))

    @axiom
    def reidemeister_1_cup(
            cls, x: C0) -> Equation[C1]:
        """ Reidemeister move 1 for cups. """
        return cls.equation_factory(
            cls.swap(x, x.r).then(cls.cups(x.r, x)),
            cls.cups(x, x.r))


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

    @axiom
    def frobenius(
            cls, x: C0) -> Equation[C1]:
        """ The Frobenius equation. """
        split, merge = cls.spiders(1, 2, x), cls.spiders(2, 1, x)
        return cls.equation_factory(
            split @ x >> x @ merge,
            merge >> split,
            x @ split >> merge @ x)

    @axiom
    def speciality(
            cls, x: C0) -> Equation[C1]:
        """ Speciality of the Frobenius structure. """
        split, merge = cls.spiders(1, 2, x), cls.spiders(2, 1, x)
        return cls.equation_factory(
            split.then(merge), cls.spiders(1, 1, x), cls.id(x))

    @axiom
    def spider_fusion(
            cls, x: C0, m: Natural, n: Natural) -> Equation[C1]:
        """ Fusion of two spiders connected by one leg. """
        return cls.equation_factory(
            cls.spiders(m, 1, x).then(cls.spiders(1, n, x)),
            cls.spiders(m, n, x))


class NamedGeneric(Generic[TypeVar('T')]):
    """
    A ``NamedGeneric`` is a ``Generic`` where the type parameter has a name.

    Parameters:
        attr : The name of the type parameter.

    Note
    ----
    In a standard ``Generic`` class, the type parameter disappears when the
    member of the class is instantiated, e.g.

    >>> assert list[int]([1, 2, 3])\\
    ...     == list[float]([1, 2, 3])\\
    ...     == [1, 2, 3]

    In a ``NamedGeneric``, the type parameter is attached to the members of the
    class so that we have access to it.

    Example
    -------

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class L(NamedGeneric["dtype"]):
    ...     inside: list
    >>> assert L[int]([1, 2, 3]).dtype == int
    >>> assert L[int]([1, 2, 3]) != L[float]([1, 2, 3])
    """
    _cache = dict()

    def __class_getitem__(_, attributes):
        if not isinstance(attributes, tuple):
            attributes = (attributes,)

        G = Generic.__class_getitem__(tuple(map(TypeVar, attributes)))

        class Result(G):
            def __class_getitem__(cls, values):
                if hasattr(cls, "__is_named_generic__"):
                    cls = cls.__bases__[0]
                values = values if isinstance(values, tuple) else (values,)
                cls_values = tuple(
                    getattr(cls, attr, None) for attr in attributes)
                if cls not in NamedGeneric._cache:
                    NamedGeneric._cache[cls] = {cls_values: cls}
                if values not in NamedGeneric._cache[cls]:
                    origin = get_origin(cls)

                    class C(origin):
                        __is_named_generic__ = True

                        # We need this to fix pickling of nested classes
                        # https://stackoverflow.com/questions/1947904/how-can-i-pickle-a-dynamically-created-nested-class-in-python
                        def __reduce__(self):
                            func, args, data = super().__reduce__()
                            # Check if class name is of the form:
                            # *ClassName*[*type*]
                            if '[' in args[0].__name__:
                                args = (origin, ) + args[1:]
                                data |= {"__class_getitem__values__": values}
                            return func, args, data

                    C.__module__ = origin.__module__
                    names = [
                        factory_name(v)
                        if isinstance(v, type)
                        and v.__module__.startswith("discopy")
                        else getattr(v, "__name__", str(v)) for v in values]
                    C.__name__ = C.__qualname__ = origin.__name__\
                        + f"[{', '.join(names)}]"
                    C.__origin__ = cls
                    for attr, value in zip(attributes, values):
                        setattr(C, attr, value)
                    NamedGeneric._cache[cls][values] = C
                return NamedGeneric._cache[cls][values]

            def __setstate__(self, state):
                if "__class_getitem__values__" in state:
                    values = state.pop("__class_getitem__values__")
                    self.__class__ = self.__class__[values]
                setstate = getattr(super(), "__setstate__", None)
                if setstate is None:
                    self.__dict__.update(state)
                else:
                    setstate(state)

            __name__ = __qualname__\
                = f"NamedGeneric[{', '.join(map(repr, attributes))}]"

        for attr in attributes:
            setattr(Result, attr, getattr(Result, attr, None))
        return Result


class Equation(NamedGeneric["ar"]):
    """
    An equation is a list of ``terms`` to be compared up to a function
    ``up_to``, the identity by default.  Casting it to ``bool`` checks
    whether its terms are all equal up to that function.

    Parameters:
        terms : The terms of the equation.
        symbol : The symbol between each pair of terms, ``"="`` by default.
        symbols : The symbols between each pair of terms, overriding
            ``symbol``; ``len(terms) * (symbol, )`` by default.
        up_to : The function up to which ``bool(equation)`` compares its
            terms, overriding the subclass' :attr:`up_to` if given.

    Example
    -------
    The number of boxes inside an arrow is left unchanged by associativity,
    so we can compare arrows up to the function that counts them modulo 2:

    >>> from discopy.cat import Ob, Box, Equation
    >>> x = Ob('x')
    >>> f, g = Box('f', x, x), Box('g', x, x)
    >>> parity = lambda term: len(term.inside) % 2
    >>> assert not Equation(f, f >> g >> g)
    >>> assert Equation(f, f >> g >> g, up_to=parity)
    """
    up_to = None

    def __init__(self, *terms, symbol="=", symbols=None, up_to=None):
        self.terms = terms
        self.symbols = tuple(symbols) if symbols is not None\
            else len(terms) * (symbol, )
        if up_to is not None:
            self.up_to = up_to

    def modulo(self, up_to: Callable) -> Equation:
        """
        The same equation compared up to the given function, rebinding
        :attr:`up_to`, whose name the attribute already takes.

        >>> from discopy.cat import Ob, Box, Equation
        >>> x = Ob('x')
        >>> f, g = Box('f', x, x), Box('g', x, x)
        >>> assert Equation(f >> g, g >> f).modulo(lambda _: True)
        """
        return type(self)(*self.terms, symbols=self.symbols, up_to=up_to)

    def __repr__(self):
        """
        >>> from discopy.cat import Ob, Box, Equation
        >>> Equation(Box('f', Ob('x'), Ob('x')))
        cat.Equation(cat.Box('f', cat.Ob('x'), cat.Ob('x')))
        """
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.terms))})"

    def __str__(self):
        return f"Equation({', '.join(map(str, self.terms))})"

    def __bool__(self):
        terms = self.terms if self.up_to is None\
            else list(map(self.up_to, self.terms))
        return all(term == terms[0] for term in terms)
