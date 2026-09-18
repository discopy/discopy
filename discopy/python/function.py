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
from discopy.monoidal import List
from discopy.utils import (
    assert_iscomposable, assert_isinstance,
    tuplify, untuplify, classproperty, factory)


Ty = List[type]
""" Lists of Python types, i.e. the free monoid on ``type``. """


@factory
@dataclass
class Function(Category):
    """
    Python function with sequential composition.

    Parameters:
        inside : The callable Python object inside the function.
        dom : The domain of the function, i.e. its input type.
        cod : The codomain of the function, i.e. its output type.

    Note
    ----
    The objects are :class:`monoidal.List` ``[type]``, the free monoid on
    Python's ``type``: a type or a tuple of types is ``cast`` into one.

    .. admonition:: Summary

        .. autosummary::

            id
            then
    """
    inside: Callable
    dom: Ty
    cod: Ty

    ob = Ty
    type_checking = True

    def __init__(self, inside: Callable, dom: Ty, cod: Ty):
        self.inside, self.dom, self.cod = (
            inside, self.ob.cast(dom), self.ob.cast(cod))

    @classmethod
    def id(cls, dom: type) -> Function:
        """
        The identity function on a given list of types :code:`dom`.

        Parameters:
            dom (type) : The list of types on which to take the identity.
        """
        return cls(lambda *xs: untuplify(xs), dom, dom)

    def then(self, *others: Function) -> Function:
        """
        The sequential composition of ``n`` functions, called with
        :code:`>>`.

        Parameters:
            others : The other functions to compose in sequence.

        Example
        -------
        >>> from discopy.python.multiplicative import Function
        >>> succ = Function(lambda x: x + 1, (int, ), (int, ))
        >>> assert succ.then(succ, succ)(0) == 3
        """
        if not others:
            return self
        factors = (self, ) + others
        for factor, other in zip(factors, others):
            assert_isinstance(other, type(self))
            assert_iscomposable(factor, other)
        *before, last = factors

        def inside(*args):
            for factor in before:
                args = tuplify(factor(*args))
            return last(*args)
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
            assert_isinstance(arg, self.dom.inside)
        result = self.inside(arg)
        if self.type_checking:
            assert_isinstance(result, self.cod.inside)
        return result
