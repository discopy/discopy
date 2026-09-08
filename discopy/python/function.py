# -*- coding: utf-8 -*-

"""
The category of Python functions with sequential composition.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Function
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from contextlib import contextmanager

from discopy.abc import Category
from discopy.utils import (
    assert_iscomposable, assert_isinstance,
    tuplify, untuplify, classproperty, factory)


def _type_monoid():
    """ The free monoid ``monoidal.List[type]`` used as ``Function.ob``. """
    from discopy.monoidal import List
    return List[type]


@factory
@dataclass
class Function(Category):
    """
    Python function with sequential composition.

    Parameters:
        inside : The callable Python object inside the function.
        dom : The domain of the function, i.e. its input type.
        cod : The codomain of the function, i.e. its output type.

    .. admonition:: Summary

        .. autosummary::

            id
            then

    Note
    ----
    The objects of the category are :code:`List[type]`, the free monoid on
    Python's :code:`type`, so that a :class:`monoidal.Functor` into
    :class:`Function` folds the image of a type with the monoid product
    :code:`@` rather than tuple concatenation.
    """
    inside: Callable
    dom: type
    cod: type

    ob = classproperty(lambda cls: _type_monoid())
    type_checking = True

    def __init__(self, inside: Callable, dom: type, cod: type):
        self.inside, self.dom, self.cod = (
            inside, self.cast(dom), self.cast(cod))

    @classmethod
    def cast(cls, dom):
        """ Cast a type or tuple of types into ``cls.ob``, the free monoid. """
        return dom if isinstance(dom, cls.ob)\
            else cls.ob(*dom) if isinstance(dom, tuple) else cls.ob(dom)

    @classmethod
    def id(cls, dom: type) -> Function:
        """
        The identity function on a given list of types :code:`dom`.

        Parameters:
            dom (type) : The list of types on which to take the identity.
        """
        return cls(lambda *xs: untuplify(xs), dom, dom)

    def then(self, other: Function) -> Function:
        """
        The sequential composition of two functions, called with :code:`>>`.

        Parameters:
            other : The other function to compose in sequence.
        """
        assert_isinstance(other, type(self))
        assert_iscomposable(self, other)
        return type(self)(
            lambda *args: other(*tuplify(self(*args))), self.dom, other.cod)

    @classproperty
    @contextmanager
    def no_type_checking(cls):
        tmp, cls.type_checking = cls.type_checking, False
        try:
            yield
        finally:
            cls.type_checking = tmp

    def __call__(self, arg):
        if self.type_checking:
            assert_isinstance(arg, tuple(self.dom))
        result = self.inside(arg)
        if self.type_checking:
            assert_isinstance(result, tuple(self.cod))
        return result
