# -*- coding: utf-8 -*-

"""
The abstract base classes for categories.

These mirror the concrete hierarchy of :mod:`discopy` modules: each class adds
the characteristic generator of its categorical structure as an
:func:`abc.abstractmethod`, e.g. :class:`BraidedCategory` is a
:class:`MonoidalCategory` with an abstract :meth:`BraidedCategory.braid`.

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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Generic, Type, TypeVar, ClassVar

from discopy.utils import get_origin


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
    ob: ClassVar[Type[C0]]
    ar: ClassVar[Type[C1]]
    dom: C0
    cod: C0

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

    def check_unitality(self) -> bool:
        """
        Check that :code:`self` is unchanged by composing with identities.
        """
        return self.id(self.dom).then(self) == self\
            and self.then(self.id(self.cod)) == self

    def check_associativity(self, other: C1, another: C1) -> bool:
        """
        Check associativity of composition with two other morphisms.

        Parameters:
            other : A morphism composable after :code:`self`.
            another : A morphism composable after :code:`other`.
        """
        return self.then(other).then(another)\
            == self.then(other.then(another))

    @classmethod
    def check_identity_typing(cls, dom: C0) -> bool:
        """
        Check that the identity on :code:`dom` has it as domain and codomain.

        Parameters:
            dom : The object on which to take the identity.
        """
        return cls.id(dom).dom == dom and cls.id(dom).cod == dom

    def check_composition_typing(self, other: C1) -> bool:
        """
        Check that the composition :code:`self.then(other)` has the domain of
        :code:`self` and the codomain of :code:`other`.

        Parameters:
            other : A morphism composable after :code:`self`.
        """
        composite = self.then(other)
        return composite.dom == self.dom and composite.cod == other.cod

    __rshift__ = __llshift__ = lambda self, other: self.then(other)
    __lshift__ = __lrshift__ = lambda self, other: other.then(self)


class Monoid[T]:
    """
    A monoid is a class with class variable ``ob`` and class method ``tensor``.
    """
    ob: ClassVar[Type[T]]

    @classmethod
    @abstractmethod
    def tensor(cls) -> T:
        """ The unit of a monoid. """

    @abstractmethod
    def tensor(self, *objects: T) -> T:
        """ The n-ary product of a monoid for ``n > 0``. """

    def check_monoid_unitality(self, unit: T) -> bool:
        """
        Check that :code:`self` is unchanged by tensoring with the unit.

        Parameters:
            unit : The unit of the monoid, e.g. the empty type.
        """
        return unit @ self == self and self @ unit == self

    def check_monoid_associativity(self, other: T, another: T) -> bool:
        """
        Check associativity of the tensor with two other objects.

        Parameters:
            other : An object to tensor after :code:`self`.
            another : An object to tensor after :code:`other`.
        """
        return self @ (other @ another) == (self @ other) @ another

    def __matmul__(self, other):
        return self.tensor(other)


class MonoidalCategory[C0: Monoid, C1: MonoidalCategory](Category[C0, C1]):
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

    def check_bifunctoriality(self, other: C0 | C1, g: C1, h: C0 | C1) -> bool:
        """
        Check the interchange law between :code:`self.then(g)` and
        :code:`other.then(h)`.

        Parameters:
            other : A morphism or object to tensor with :code:`self`.
            g : A morphism composable after :code:`self`.
            h : A morphism or object composable after :code:`other`.
        """
        return (self @ other).then(g @ h)\
            == self.then(g) @ self.whisker(other).then(h)

    @classmethod
    def check_tensor_unitality(cls, x: C0, y: C0) -> bool:
        """
        Check that tensoring preserves identities.

        Parameters:
            x : The domain of the first identity.
            y : The domain of the second identity.
        """
        return cls.id(x) @ cls.id(y) == cls.id(x @ y)

    def check_tensor_typing(self, other: C1) -> bool:
        """
        Check that :code:`self @ other` has the tensor of domains as domain
        and the tensor of codomains as codomain.

        Parameters:
            other : A morphism to tensor with :code:`self`.
        """
        return (self @ other).dom == self.dom @ other.dom\
            and (self @ other).cod == self.cod @ other.cod


class TracedCategory[C0, C1](MonoidalCategory[C0, C1]):
    """
    A traced category is a :class:`MonoidalCategory` with a method
    :code:`trace` for the partial trace of a morphism over some objects.
    """
    @abstractmethod
    def trace(self, n: int = 1, left: bool = False) -> C1:
        """
        The trace of a morphism, to be instantiated.

        Parameters:
            n : The number of objects to trace over.
            left : Whether to trace the wires on the left or right.
        """

    def check_trace_vanishing(self) -> bool:
        """
        Check that tracing over the unit object does nothing.
        """
        return self.trace(0) == self and self.trace(0, left=True) == self

    def check_trace_superposing(self, obj: C0, left: bool = False) -> bool:
        """
        Check that tracing :code:`self` superposed with :code:`obj` is the
        same as superposing :code:`obj` with the trace of :code:`self`.

        Parameters:
            obj : The object to superpose :code:`self` with.
            left : Whether to trace the wires on the left or right.
        """
        if left:
            return (self @ obj).trace(left=True) == self.trace(left=True) @ obj
        return (obj @ self).trace() == obj @ self.trace()

    def check_trace_naturality(
            self, x: C0, g: C1, left: bool = False) -> bool:
        """
        Check the tightening axiom, i.e. that an endomorphism :code:`g` on
        the untraced wires can be slid through the trace over :code:`x`.

        Parameters:
            x : The object to trace over.
            g : An endomorphism on the untraced wires of :code:`self`.
            left : Whether to trace the wires on the left or right.
        """
        if left:
            lhs = (x @ g).then(self).then(x @ g).trace(left=True)
            rhs = g.then(self.trace(left=True)).then(g)
        else:
            lhs = (g @ x).then(self).then(g @ x).trace()
            rhs = g.then(self.trace()).then(g)
        return lhs == rhs

    def check_trace_dinaturality(
            self, x: C0, g: C1, left: bool = False) -> bool:
        """
        Check the sliding axiom, i.e. that a morphism :code:`g` can be slid
        from before to after the trace over the untraced wires :code:`x`.

        Parameters:
            x : The object left untraced.
            g : The morphism to slide across the trace.
            left : Whether to trace the wires on the left or right.
        """
        if left:
            lhs = self.then(g @ x).trace(left=True)
            rhs = (g @ x).then(self).trace(left=True)
        else:
            lhs = self.then(x @ g).trace()
            rhs = (x @ g).then(self).trace()
        return lhs == rhs


class ResiduatedMonoid[T](Monoid[T]):
    """
    A monoid is residuated when it comes with methods ``over`` and ``under``
    with syntactic sugar ``<<`` and ``>>``.
    """
    @abstractmethod
    def over(self, other: T) -> T:
        """ The right-to-left exponential object ``self`` to the ``other``. """

    @abstractmethod
    def under(self, other: T) -> T:
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

    def check_currying(
            self, base: C0, exponent: C0, left: bool = True) -> bool:
        """
        Check that uncurrying :code:`self.curry(left=left)` recovers
        :code:`self`, i.e. that currying and evaluation are inverse to each
        other.

        Note
        ----
        This is in general a semantic axiom, e.g. it need not hold for free
        biclosed diagrams under syntactic equality.

        Parameters:
            base : The base of the exponential type, i.e. :code:`self.cod`.
            exponent : The object to curry :code:`self` over.
            left : Whether to curry on the left or right.
        """
        curried = self.curry(left=left)
        ev = type(self).ev(base, exponent, left)
        uncurried = (curried @ exponent).then(ev) if left\
            else (exponent @ curried).then(ev)
        return uncurried == self


class Pregroup[T](ResiduatedMonoid[T]):
    """
    A pregroup is a residuated monoid where the left and right exponentials are
    given by tensoring with the chosen left and right duals for each object.
    """
    l: T
    r: T

    def tensor(self, *others: T) -> T:
        return super(Monoid, self).tensor(*others)

    def __matmul__(self, other: T) -> T:
        return self.tensor(other)

    def over(self, other: T) -> T:
        return self @ other.l

    def under(self, other: T) -> T:
        return other.r @ self

    def check_adjunction(self) -> bool:
        """
        Check that the left and right adjoints of :code:`self` are inverse
        to each other.
        """
        return self.l.r == self and self.r.l == self


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
    def check_snake_equations(
            cls, x: C0, eq: Callable[[C1, C1], bool] = None) -> bool:
        """
        Check the snake equations, i.e. that the cups and caps for :code:`x`
        compose to the identity.

        Note
        ----
        This is in general only true up to some normalisation, e.g. for free
        rigid diagrams it requires :code:`eq` to compare normal forms. This is
        the one axiom that does not hold on the nose for any concrete category
        in :mod:`discopy`, hence the :code:`eq` parameter.

        Parameters:
            x : The object to check the snake equations for.
            eq : The notion of equality to use, ``==`` by default.
        """
        eq = eq or (lambda f, g: f == g)
        snake_r = (cls.id(x) @ cls.caps(x.r, x)).then(
            cls.cups(x, x.r) @ cls.id(x))
        snake_l = (cls.caps(x, x.l) @ cls.id(x)).then(
            cls.id(x) @ cls.cups(x.l, x))
        return eq(snake_r, cls.id(x)) and eq(snake_l, cls.id(x))

    @classmethod
    def check_caps_coherence(cls, x: C0, y: C0) -> bool:
        """
        Check that the caps for :code:`x @ y` decompose into the caps for
        :code:`x` and :code:`y`, using that the adjoint is a monoid
        anti-homomorphism, i.e. :code:`(x @ y).l == y.l @ x.l`.

        Parameters:
            x, y : The two objects to tensor before taking caps.
        """
        return cls.caps(x @ y, (x @ y).l)\
            == cls.caps(x, x.l).then(x @ cls.caps(y, y.l) @ x.l)


class PivotalCategory[C0, C1](RigidCategory[C0, C1], TracedCategory[C0, C1]):
    """
    A pivotal category is a :class:`RigidCategory` where the left and right
    adjoints coincide, hence it is also a :class:`TracedCategory`.
    """
    @classmethod
    def check_self_dual(cls, x: C0) -> bool:
        """
        Check that the left and right adjoints of :code:`x` coincide.

        Parameters:
            x : The object to check self-duality for.
        """
        return x.r == x.l

    def check_transpose(self) -> bool:
        """
        Check that the left and right transpose of :code:`self` coincide.
        This is the defining axiom of a pivotal category on top of a rigid one,
        i.e. that a morphism can be rotated by a full turn in either direction.

        The transpose of :code:`self: x -> y` is built by bending the input and
        output wires around using :meth:`cups` and :meth:`caps`, on the left to
        get a morphism :code:`y.l -> x.l` and on the right :code:`y.r -> x.r`.
        In a pivotal category :code:`x.l == x.r`, so they have the same type.

        Note
        ----
        This is in general a semantic axiom, e.g. it need not hold for free
        pivotal diagrams under syntactic equality.
        """
        dom, cod = self.dom, self.cod
        left_transpose = (cod.l @ self.caps(dom, dom.l)).then(
            cod.l @ self @ dom.l).then(self.cups(cod.l, cod) @ dom.l)
        right_transpose = (self.caps(dom.r, dom) @ cod.r).then(
            dom.r @ self @ cod.r).then(dom.r @ self.cups(cod, cod.r))
        return left_transpose == right_transpose


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

    @classmethod
    def check_hexagon(cls, x: C0, y: C0, z: C0) -> bool:
        """
        Check the two hexagon equations relating :code:`braid` and
        :code:`tensor`.

        Parameters:
            x, y, z : The three objects of the hexagon equations.
        """
        left = cls.braid(x, y @ z)\
            == (cls.braid(x, y) @ z).then(y @ cls.braid(x, z))
        right = cls.braid(x @ y, z)\
            == (x @ cls.braid(y, z)).then(cls.braid(x, z) @ y)
        return left and right

    def check_braid_naturality(self, other: C1) -> bool:
        """
        Check the naturality of the braid with respect to :code:`self` and
        :code:`other`.

        Parameters:
            other : The other morphism to braid with :code:`self`.
        """
        return (self @ other).then(type(self).braid(self.cod, other.cod))\
            == type(self).braid(self.dom, other.dom).then(other @ self)


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

    @classmethod
    def check_balanced_twist(cls, x: C0, y: C0) -> bool:
        """
        Check that the twist on :code:`x @ y` decomposes via braids and the
        twists on :code:`x` and :code:`y`.

        Parameters:
            x, y : The two objects to tensor before twisting.
        """
        return cls.twist(x @ y) == cls.braid(x, y).then(
            cls.twist(y) @ cls.twist(x)).then(cls.braid(y, x))


class SymmetricCategory[C0, C1](BalancedCategory[C0, C1]):
    """
    A symmetric category is a :class:`BalancedCategory` where the braid is its
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
    def twist(cls, dom: C0) -> C1:
        return cls.id(dom)

    @classmethod
    def braid(cls, left: C0, right: C0) -> C1:
        return cls.swap(left, right)

    @classmethod
    def check_swap_inverse(cls, x: C0, y: C0) -> bool:
        """
        Check that the swap is its own inverse, i.e. Reidemeister move 2.

        Parameters:
            x, y : The two objects to swap.
        """
        return cls.swap(x, y).then(cls.swap(y, x)) == cls.id(x @ y)


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

    @classmethod
    def check_copy_counitality(cls, x: C0) -> bool:
        """
        Check the counitality of :code:`copy`, i.e. that discarding one of
        the two copies of :code:`x` does nothing.

        Parameters:
            x : The object to copy and discard.
        """
        copy, discard = cls.copy(x), cls.copy(x, n=0)
        return copy.then(discard @ x) == cls.id(x)\
            and copy.then(x @ discard) == cls.id(x)

    @classmethod
    def check_copy_coassociativity(cls, x: C0) -> bool:
        """
        Check the coassociativity of :code:`copy`.

        Parameters:
            x : The object to copy three times.
        """
        copy = cls.copy(x)
        return copy.then(copy @ x) == copy.then(x @ copy)

    @classmethod
    def check_copy_cocommutativity(cls, x: C0) -> bool:
        """
        Check the cocommutativity of :code:`copy`.

        Parameters:
            x : The object to copy.
        """
        copy = cls.copy(x)
        return copy.then(cls.swap(x, x)) == copy

    @classmethod
    def check_copy_coherence(cls, x: C0) -> bool:
        """
        Check that copying :code:`x @ x` decomposes into copying each
        factor and swapping the middle wires.

        Parameters:
            x : The object to tensor with itself before copying.
        """
        return cls.copy(x @ x, n=0) == cls.copy(x, n=0) @ cls.copy(x, n=0)\
            and cls.copy(x @ x)\
            == (cls.copy(x) @ cls.copy(x)).then(x @ cls.swap(x, x) @ x)


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

    def check_feedback_vanishing(self, unit: C0) -> bool:
        """
        Check that feeding back over the unit object does nothing.

        Parameters:
            unit : The unit of the monoid of objects, e.g. the empty type.
        """
        return self.feedback(mem=unit) == self

    def check_feedback_joining(self, mem: C0) -> bool:
        """
        Check that feeding back over :code:`mem` is the same as feeding
        back twice in a row.

        Parameters:
            mem : The memory type to feedback over, of length two.
        """
        return self.feedback(mem=mem) == self.feedback().feedback()


class RibbonCategory[C0, C1](
        PivotalCategory[C0, C1], BalancedCategory[C0, C1]):
    """
    A ribbon category is a :class:`PivotalCategory` which is also a
    :class:`BalancedCategory`, i.e. where diagrams can draw knots and links.
    """
    @classmethod
    def check_twist_as_trace(cls, x: C0) -> bool:
        """
        Check that the twist on :code:`x` is the (left and right) trace of
        the braid of :code:`x` with itself.

        Note
        ----
        This is in general a semantic axiom, e.g. it need not hold for free
        ribbon diagrams under syntactic equality.

        Parameters:
            x : The object to twist and braid.
        """
        braid = cls.braid(x, x)
        return cls.twist(x) == braid.trace(left=True)\
            and cls.twist(x) == braid.trace()


class CompactCategory[C0, C1](
        RibbonCategory[C0, C1], SymmetricCategory[C0, C1]):
    """
    A compact category is a :class:`RibbonCategory` which is also a
    :class:`SymmetricCategory`, i.e. with cups, caps and swaps.
    """
    @classmethod
    def check_reidemeister_1(cls, x: C0) -> bool:
        """
        Check Reidemeister move 1, i.e. that a cap or cup can be slid past
        a swap.

        Parameters:
            x : The object to check the equation for.
        """
        return cls.caps(x, x.r).then(cls.swap(x, x.r)) == cls.caps(x.r, x)\
            and cls.swap(x, x.r).then(cls.cups(x.r, x)) == cls.cups(x, x.r)


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

    @classmethod
    def check_frobenius(cls, x: C0) -> bool:
        """
        Check the Frobenius law relating the splitting and merging spiders
        on :code:`x`.

        Parameters:
            x : The object to take spiders on.
        """
        split, merge = cls.spiders(1, 2, x), cls.spiders(2, 1, x)
        left = (split @ x).then(x @ merge)
        middle = merge.then(split)
        right = (x @ split).then(merge @ x)
        return left == middle and middle == right

    @classmethod
    def check_speciality(cls, x: C0) -> bool:
        """
        Check that splitting then merging the spiders on :code:`x` is the
        identity.

        Parameters:
            x : The object to take spiders on.
        """
        split, merge = cls.spiders(1, 2, x), cls.spiders(2, 1, x)
        return split.then(merge) == cls.spiders(1, 1, x)\
            and cls.spiders(1, 1, x) == cls.id(x)


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
                    names = [getattr(v, "__name__", str(v)) for v in values]
                    C.__name__ = C.__qualname__ = origin.__name__\
                        + f"[{', '.join(names)}]"
                    C.__origin__ = cls
                    for attr, value in zip(attributes, values):
                        setattr(C, attr, value)
                    NamedGeneric._cache[cls][values] = C
                return NamedGeneric._cache[cls][values]

            __name__ = __qualname__\
                = f"NamedGeneric[{', '.join(map(repr, attributes))}]"

        for attr in attributes:
            setattr(Result, attr, getattr(Result, attr, None))
        return Result

    def __setstate__(self, state):
        if "__class_getitem__values__" in state:
            new_cls = self.__class__[state["__class_getitem__values__"]]
            self.__class__ = new_cls
