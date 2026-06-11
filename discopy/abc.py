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
from typing import Generic, Optional, Type, TypeVar

from discopy.utils import get_origin

T = TypeVar('T')


class Category(ABC, Generic[T]):
    """
    A category is an arrow type, i.e. a class with the appropriate methods
    :code:`dom`, :code:`cod`, :code:`id` and :code:`then`, together with a
    :attr:`ty_factory` for its objects.

    As such, it also implements the syntactic sugar :code:`>>` and :code:`<<`
    for forward and backward composition with the method :code:`then`.

    Example
    -------
    >>> class List(list, Category):
    ...     def then(self, other):
    ...         return self + other
    >>> assert List([1, 2]) >> List([3]) == List([1, 2, 3])
    >>> assert List([3]) << List([1, 2]) == List([1, 2, 3])
    """
    factory: Type[Category]
    sum_factory: Type[Category]
    ty_factory: Type[T]
    dom: T
    cod: T

    @abstractmethod
    def then(self, other: Optional[Category[T]], *others: Category[T]
             ) -> Category[T]:
        """
        Sequential composition, to be instantiated.

        Parameters:
            other : The other arrow to compose sequentially.
        """

    def is_composable(self, other: Category) -> bool:
        """
        Whether two arrows are composable, i.e. the codomain of the first is
        the domain of the second.

        Parameters:
            other : The other arrow.
        """
        return self.cod == other.dom

    def is_parallel(self, other: Category) -> bool:
        """
        Whether two arrows are parallel, i.e. they have the same
        domain and codomain.

        Parameters:
            other : The other arrow.
        """
        return (self.dom, self.cod) == (other.dom, other.cod)

    __rshift__ = __llshift__ = lambda self, other: self.then(other)
    __lshift__ = __lrshift__ = lambda self, other: other.then(self)


class MonoidalCategory(Category[T]):
    """
    A monoidal category is a :class:`Category` with a method :code:`tensor`,
    implementing the syntactic sugar :code:`@` for whiskering and parallel
    composition.
    """
    @classmethod
    @abstractmethod
    def id(cls, dom: any) -> MonoidalCategory:
        """
        Identity on a given domain, to be instantiated.

        Parameters:
            dom : The object on which to take the identity.
        """

    @abstractmethod
    def tensor(self, other: MonoidalCategory) -> MonoidalCategory:
        """
        Parallel composition, to be instantiated.

        Parameters:
            other : The other arrow to compose in parallel.
        """

    @classmethod
    def whisker(cls, other: any) -> MonoidalCategory:
        """
        Apply :meth:`MonoidalCategory.id` if :code:`other` is not tensorable
        else do nothing.

        Parameters:
            other : The whiskering object.
        """
        return other if isinstance(other, MonoidalCategory) else cls.id(other)

    def __matmul__(self, other):
        return self.tensor(self.whisker(other))

    def __rmatmul__(self, other):
        return self.whisker(other).tensor(self)


class BraidedCategory(MonoidalCategory[T]):
    """
    A braided category is a :class:`MonoidalCategory` with a method
    :code:`braid` for the natural isomorphism :code:`x @ y -> y @ x`.
    """
    @classmethod
    @abstractmethod
    def braid(cls, left: T, right: T) -> BraidedCategory:
        """
        The braid of two objects, to be instantiated.

        Parameters:
            left : The object on the left of the braid.
            right : The object on the right of the braid.
        """


class TracedCategory(MonoidalCategory[T]):
    """
    A traced category is a :class:`MonoidalCategory` with a method
    :code:`trace` for the partial trace of a morphism over some objects.
    """
    @abstractmethod
    def trace(self, n: int = 1, left: bool = False) -> TracedCategory:
        """
        The trace of a morphism, to be instantiated.

        Parameters:
            n : The number of objects to trace over.
            left : Whether to trace the wires on the left or right.
        """


class BalancedCategory(BraidedCategory[T], TracedCategory[T]):
    """
    A balanced category is a :class:`BraidedCategory` and a
    :class:`TracedCategory` with a method :code:`twist` for the natural
    automorphism :code:`x -> x`.
    """
    @classmethod
    @abstractmethod
    def twist(cls, dom: T) -> BalancedCategory:
        """
        The twist on an object, to be instantiated.

        Parameters:
            dom : The object on which to take the twist.
        """


class SymmetricCategory(BalancedCategory[T]):
    """
    A symmetric category is a :class:`BalancedCategory` with a method
    :code:`swap` for the symmetry :code:`x @ y -> y @ x`.
    """
    @classmethod
    @abstractmethod
    def swap(cls, left: T, right: T) -> SymmetricCategory:
        """
        The swap of two objects, to be instantiated.

        Parameters:
            left : The object on the left of the swap.
            right : The object on the right of the swap.
        """


class MarkovCategory(SymmetricCategory[T]):
    """
    A Markov category is a :class:`SymmetricCategory` with methods
    :code:`copy` and :code:`merge` for the supply of commutative comonoids.
    """
    @classmethod
    @abstractmethod
    def copy(cls, x: T, n: int = 2) -> MarkovCategory:
        """
        Make :code:`n` copies of a given object :code:`x`.

        Parameters:
            x : The object to copy.
            n : The number of copies.
        """

    @classmethod
    @abstractmethod
    def merge(cls, x: T, n: int = 2) -> MarkovCategory:
        """
        Merge :code:`n` copies of a given object :code:`x`.

        Parameters:
            x : The object to merge.
            n : The number of copies.
        """


class FeedbackCategory(MarkovCategory[T]):
    """
    A feedback category is a :class:`MarkovCategory` with a :code:`delay`
    endofunctor and a :code:`feedback` operator.
    """
    @abstractmethod
    def delay(self, n_steps: int = 1) -> FeedbackCategory:
        """
        The delay endofunctor applied to a morphism.

        Parameters:
            n_steps : The number of time steps to delay.
        """

    @abstractmethod
    def feedback(self, dom=None, cod=None, mem=None) -> FeedbackCategory:
        """
        The feedback operator on a morphism.

        Parameters:
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory type to trace over.
        """


class ClosedCategory(MonoidalCategory[T]):
    """
    A closed category is a :class:`MonoidalCategory` with methods :code:`ev`
    and :code:`curry` for the evaluation and currying of morphisms.
    """
    @classmethod
    @abstractmethod
    def ev(cls, base: T, exponent: T, left: bool = True) -> ClosedCategory:
        """
        The evaluation of an exponential type, to be instantiated.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """

    @abstractmethod
    def curry(self, n: int = 1, left: bool = True) -> ClosedCategory:
        """
        The currying of a morphism, to be instantiated.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.
        """


class RigidCategory(ClosedCategory[T]):
    """
    A rigid category is a :class:`ClosedCategory` where every object has a
    left and right adjoint, witnessed by methods :code:`cups` and :code:`caps`.
    """
    @classmethod
    @abstractmethod
    def cups(cls, left: T, right: T) -> RigidCategory:
        """
        The cups witnessing :code:`right` as the adjoint of :code:`left`.

        Parameters:
            left : The left-hand side of the cups.
            right : Its adjoint, i.e. the right-hand side of the cups.
        """

    @classmethod
    @abstractmethod
    def caps(cls, left: T, right: T) -> RigidCategory:
        """
        The caps witnessing :code:`right` as the adjoint of :code:`left`.

        Parameters:
            left : The left-hand side of the caps.
            right : Its adjoint, i.e. the right-hand side of the caps.
        """


class PivotalCategory(RigidCategory[T], TracedCategory[T]):
    """
    A pivotal category is a :class:`RigidCategory` where the left and right
    adjoints coincide, hence it is also a :class:`TracedCategory`, with a
    method :code:`conjugate` for the horizontal reflection of a morphism.
    """
    @abstractmethod
    def conjugate(self) -> PivotalCategory:
        """ The horizontal reflection of a morphism, to be instantiated. """


class RibbonCategory(PivotalCategory[T], BalancedCategory[T]):
    """
    A ribbon category is a :class:`PivotalCategory` and a
    :class:`BalancedCategory`, i.e. a balanced category where every object has
    an adjoint.
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
