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

from discopy.cat import Composable
from discopy.utils import (
    Whiskerable, assert_iscomposable, assert_isinstance,
    tuplify, untuplify, classproperty)


@dataclass
class Function(Composable[type], Whiskerable):
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
