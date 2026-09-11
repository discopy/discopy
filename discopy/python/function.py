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

    def then(self, *others: Function) -> Function:
        """
        The sequential composition of ``n`` functions, called with
        :code:`>>`.

        Parameters:
            others : The other functions to compose in sequence.

        Example
        -------
        >>> succ = Function(lambda x: x + 1, (int, ), (int, ))
        >>> assert succ.then(succ, succ)(0) == 3
        >>> assert succ.then()(0) == 1

        Note
        ----
        The composite applies each function in a loop rather than nesting
        ``n`` closures, so calling it costs one Python frame whatever ``n``.
        """
        if not others:
            return self
        functions = (self, ) + others
        for other in others:
            assert_isinstance(other, type(self))
        for f, g in zip(functions, others):
            assert_iscomposable(f, g)

        def inside(*args):
            for function in functions:
                args = tuplify(function(*args))
            return untuplify(args)
        return type(self)(inside, self.dom, others[-1].cod)

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
