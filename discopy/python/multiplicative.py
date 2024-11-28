# -*- coding: utf-8 -*-

"""
The category of Python functions with tuple as monoidal product.

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

        exp
"""

from __future__ import annotations

from collections.abc import Callable

from discopy.cat import assert_isinstance
from discopy.utils import tuplify, untuplify
from discopy.python import function


""" Functions have lists of types as input and output. """
Ty = tuple[type, ...]


def exp(base: Ty, exponent: Ty) -> Ty:
    """
    The exponential of a tuple of Python types by another.

    Parameters:
        base (python.Ty) : The base type.
        exponent (python.Ty) : The exponent type.
    """
    return (Callable[list(exponent), tuple[base]], )


class Function(function.Function):
    """
    Python function with tuple as tensor.

    Parameters:
        inside : The callable Python object inside the function.
        dom : The domain of the function, i.e. its list of input types.
        cod : The codomain of the function, i.e. its list of output types.

    .. admonition:: Summary

        .. autosummary::

            tensor
            swap
            copy
            discard
            ev
            curry
            uncurry
            fix
            trace
    """
    __ambiguous_inheritance__ = True

    ty_factory = Ty

    def __call__(self, *xs):
        if self.type_checking:
            if len(xs) != len(self.dom):
                raise ValueError
            for (x, t) in zip(xs, self.dom):
                callable(x) or assert_isinstance(x, t)
        ys = self.inside(*xs)
        if self.type_checking:
            if len(self.cod) != 1 and (
                    not isinstance(ys, tuple) or len(self.cod) != len(ys)):
                raise RuntimeError
            for (y, t) in zip(tuplify(ys), self.cod):
                callable(y) or assert_isinstance(y, t)
        return ys

    def tensor(self, other: Function) -> Function:
        """
        The parallel composition of two functions, called with :code:`@`.

        Parameters:
            other : The other function to compose in sequence.
        """
        def inside(*xs):
            left, right = xs[:len(self.dom)], xs[len(self.dom):]
            return untuplify(tuplify(self(*left)) + tuplify(other(*right)))
        return Function(inside, self.dom + other.dom, self.cod + other.cod)

    @staticmethod
    def swap(x: Ty, y: Ty) -> Function:
        """
        The function for swapping two tuples of types :code:`x` and :code:`y`.

        Parameters:
            x : The tuple of types on the left.
            y : The tuple of types on the right.
        """
        def inside(*xs):
            return untuplify(tuplify(xs)[len(x):] + tuplify(xs)[:len(x)])
        return Function(inside, dom=x + y, cod=y + x)

    braid = swap

    @staticmethod
    def copy(x: Ty, n=2) -> Function:
        """
        The function for making :code:`n` copies of a tuple of types :code:`x`.

        Parameters:
            x : The tuple of types to copy.
            n : The number of copies.
        """
        return Function(lambda *xs: n * xs, dom=x, cod=n * x)

    @staticmethod
    def discard(dom: Ty) -> Function:
        """
        The function discarding a tuple of types, i.e. making zero copies.

        Parameters:
            dom : The tuple of types to discard.
        """
        return Function.copy(dom, 0)

    @staticmethod
    def ev(base: Ty, exponent: Ty, left=True) -> Function:
        """
        The evaluation function,
        i.e. take a function and apply it to an argument.

        Parameters:
            base : The output type.
            exponent : The input type.
            left : Whether to take the function on the left or right.
        """
        if left:
            dom, cod = Function.exp(base, exponent) + exponent, base
            return Function(lambda f, *xs: f(*xs), dom, cod)
        dom, cod = exponent + Function.exp(base, exponent), base
        return Function(lambda *xs: xs[-1](*xs[:-1]), dom, cod)

    def curry(self, n=1, left=True) -> Function:
        """
        Currying, i.e. turn a binary function into a function-valued function.

        Parameters:
            n : The number of types to curry.
            left : Whether to curry on the left or right.
        """
        if left:
            dom = self.dom[:len(self.dom) - n]
            cod = Function.exp(self.cod, self.dom[len(self.dom) - n:])
        else:
            dom, cod = self.dom[n:], Function.exp(self.cod, self.dom[:n])
        return Function(dom=dom, cod=cod, inside=lambda *xs: lambda *ys:
                        self(*(xs + ys) if left else (ys + xs)))

    def uncurry(self, left=True) -> Function:
        """
        Uncurrying,
        i.e. turn a function-valued function into a binary function.

        Parameters:
            left : Whether to uncurry on the left or right.
        """
        traced = self.cod[0].__args__
        base, exponent = traced[-1].__args__, traced[:-1]
        return self @ exponent >> Function.ev(base, exponent) if left\
            else exponent @ self >> Function.ev(base, exponent, left=False)

    def fix(self, n=1) -> Function:
        """
        The parameterised fixed point of a function.

        Parameters:
            n : The number of types to take the fixed point over.
        """
        def inside(*xs, y=None):
            result = self.inside(*xs + (() if y is None else (y, )))
            return y if result == y else inside(*xs, y=result)
        return self if n == 0\
            else Function(inside, self.dom[:-1], self.cod).fix(n - 1)

    def trace(self, n=1, left=False):
        """
        The multiplicative trace of a function.

        Parameters:
            n : The number of types to trace over.
        """
        if left:
            raise NotImplementedError
        dom, cod, traced = self.dom[:-n], self.cod[:-n], self.dom[-n:]
        fixed = (self >> self.discard(cod) @ traced).fix()
        return self.copy(dom) >> dom @ fixed\
            >> self >> cod @ self.discard(traced)

    exp = over = under = staticmethod(lambda x, y: exp(x, y))


class Category:
    ob = Ty
    ar = Function
