# -*- coding: utf-8 -*-

"""
The abstract base classes for categories.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Category
    MonoidalCategory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, Type, TypeVar

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
