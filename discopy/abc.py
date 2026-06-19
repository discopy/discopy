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
from typing import Generic, Type, TypeVar, ClassVar

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

    __rshift__ = __llshift__ = lambda self, other: self.then(other)
    __lshift__ = __lrshift__ = lambda self, other: other.then(self)


class Monoid[C0, C1: Monoid](Category[C0, C1]):
    """
    A monoid is a category with ``then`` given by ``tensor``.

    Note
    ----
    Usually, a monoid is expected to have :class:`type(None)` as its object
    type. We do not enforce this constraint so that :class:`monoidal.Ty` can
    instead take colours as objects.
    """
    ar: ClassVar[Type[C1]]

    @classmethod
    @abstractmethod
    def tensor(cls) -> C1:
        """The empty tensor, i.e. the monoidal unit."""

    @classmethod
    def id(cls, dom: C0 = None) -> C1:
        """The monoidal unit, seen as an identity morphism."""
        return cls.ar()

    @abstractmethod
    def tensor(self, *objects: C1) -> C1:
        """ The n-ary product of a monoid for ``n > 0``. """

    def then(self, *others: C1) -> C1:
        """Sequential composition, given by the monoid product."""
        return self.tensor(*others)

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


class ResiduatedMonoid[C0, C1: ResiduatedMonoid](Monoid[C0, C1]):
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
    :class:`SymmetricCategory`, i.e. with cups, caps and swaps.
    """


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
