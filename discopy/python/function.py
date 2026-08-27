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
from discopy.testing import Strategy
from discopy.utils import (
    assert_iscomposable, assert_isinstance,
    tuplify, untuplify, classproperty, factory)


class Types(tuple, Strategy["Types"]):
    """
    A tuple of Python types seen as an object, with a strategy drawing
    tuples of :class:`int` — the one-type universe the property matrix
    generates its functions over.
    """
    def __matmul__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return type(self)(tuple(self) + tuple(other))

    def __rmatmul__(self, other):
        if not isinstance(other, tuple):
            return NotImplemented
        return type(self)(tuple(other) + tuple(self))

    @classmethod
    def strategy(cls, *, min_length=0, max_length=3, **_):
        """Generate tuples of the integer type."""
        from hypothesis import strategies as st

        return st.integers(
            min_value=min_length, max_value=max_length).map(
                lambda length: cls(length * (int, )))

    @classmethod
    def equation_factory(cls, *terms):
        """ Tuples of types are compared on the nose. """
        from discopy.cat import Equation

        return Equation(*terms)


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
    """
    inside: Callable
    dom: type
    cod: type

    ob = tuple[type, ...]
    type_checking = True

    def __init__(self, inside: Callable, dom: type, cod: type):
        dom, cod = map(tuplify, (dom, cod))
        self.inside, self.dom, self.cod = inside, dom, cod

    @classmethod
    def id(cls, dom: type) -> Function:
        """
        The identity function on a given tuple of types :code:`dom`.

        Parameters:
            dom (type) : The typle of types on which to take the identity.
        """
        return cls(lambda *xs: untuplify(xs), tuplify(dom), tuplify(dom))

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
            assert_isinstance(arg, self.dom)
        result = self.inside(arg)
        if self.type_checking:
            assert_isinstance(result, self.cod)
        return result
