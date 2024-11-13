# -*- coding: utf-8 -*-

"""
The category of Python functions with disjoint union as monoidal product.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Function

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        is_union
"""

from __future__ import annotations

from functools import cache

from discopy.utils import assert_isinstance, tuplify
from discopy.python import function


""" Lists of types interpreted as disjoint union. """
Ty = tuple[type, ...]


def tagged(tag, dom):
    return () if len(dom) == 1 else (tag, )


class Function(function.Function):
    """ Python functions with disjoint union as tensor. """
    ty_factory = Ty

    def __init__(self, inside, dom, cod, is_swap_of=None):
        self.is_swap_of = is_swap_of
        super().__init__(inside, dom, cod)

    def __call__(self, obj, tag=0):
        if self.type_checking:
            assert_isinstance(obj, self.dom[tag])
        result = self.inside(obj, *tagged(tag, self.dom))
        if self.type_checking:
            obj, tag = (result, 0) if len(self.cod) == 1 else result
            assert_isinstance(obj, self.cod[tag])
        return result

    def tensor(self, other: Function) -> Function:
        """
        The disjoint union of two functions, called with :code:`@`.

        Parameters:
            other : The other function to compose in sequence.
        """
        dom, cod = self.dom + other.dom, self.cod + other.cod

        def inside(obj, tag=0):
            if tag < len(self.dom):
                result = self(obj, *tagged(tag, self.dom))
                obj, tag = (result, 0) if len(self.cod) == 1 else result
            else:
                result = other(obj, *tagged(tag - len(self.dom), other.dom))
                obj, tag = (result, 0) if len(other.cod) == 1 else result
                tag += len(self.cod)
            return obj if len(cod) == 1 else (obj, tag)
        return Function(inside, dom, cod)

    @staticmethod
    @cache
    def swap(x: Ty, y: Ty) -> Function:
        """
        Swap the tags of a disjoint union from `x + y` to `y + x`.

        Parameters:
            x : The tuple of types on the left.
            y : The tuple of types on the right.
        """
        x, y = map(tuplify, (x, y))

        def inside(obj, tag=0):
            new_tag = tag + len(y) if tag < len(x) else tag - len(x)
            if len(x + y) == 1:
                assert new_tag == 0
                return obj
            return (obj, new_tag)
        return Function(inside, dom=x + y, cod=y + x, is_swap_of=(x, y))

    def dagger(self):
        if self.is_swap_of is None:
            raise ValueError
        return Function.swap(*self.is_swap_of[::-1])

    def trace(self, n=1, left=False):
        """
        The additive trace of a function.

        Parameters:
            n : The number of types to trace over.
        """
        if left:
            raise NotImplementedError
        dom, cod = self.dom[:-n], self.cod[:-n]

        def inside(obj, tag=0):
            run_at_least_once = True
            while run_at_least_once or tag >= len(cod):
                run_at_least_once = False
                result = self(obj, *tagged(tag, self.dom))
                obj, tag = (result, 0) if len(self.cod) == 1 else result
            return obj if len(cod) == 1 else result
        return Function(inside, dom, cod)


Function.braid = Function.swap
Function.twist = Function.id


class Category:
    ob = Ty
    ar = Function
