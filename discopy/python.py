# -*- coding: utf-8 -*-

"""
The monoidal category of Python types and functions with tuple as tensor.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    exp
    Function
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from discopy import cat
from discopy.cat import Category, Composable
from discopy.monoidal import Whiskerable


Ty = tuple[type, ...]

tuplify = lambda stuff: stuff if isinstance(stuff, tuple) else (stuff, )
untuplify = lambda stuff: stuff[0] if len(stuff) == 1 else stuff
is_tuple = lambda typ: getattr(typ, "__origin__", None) is tuple


def exp(base: Ty, exponent: Ty) -> Ty:
    """
    The exponential of a tuple of Python types by another.

    Parameters:
        base (python.Ty) : The base type.
        exponent (python.Ty) : The exponent type.
    """
    return (Callable[exponent, tuple[base]], )


@dataclass
class Function(Composable, Whiskerable):
    """
    A function is a callable :code:`inside`
    with a pair of types :code:`dom` and :code:`cod`.

    Parameters:
        inside : The callable Python object inside the function.
        dom : The domain of the function, i.e. its input type.
        cod : The codomain of the function, i.e. its output type.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            tensor
            swap
            copy
            discard
            eval
            curry
            uncurry
            fix
            trace
    """
    inside: Callable
    dom: Ty
    cod: Ty

    @classmethod
    def id(cls, dom: Ty) -> Function:
        """
        The identity function on a given tuple of types :code:`dom`.

        Parameters:
            dom (python.Ty) : The typle of types on which to take the identity.
        """
        return cls(lambda *xs: untuplify(xs), dom, dom)

    def then(self, other: Function) -> Function:
        """
        The sequential composition of two functions, called with :code:`>>`.

        Parameters:
            other : The other function to compose in sequence.
        """
        assert self.cod == other.dom
        inside = lambda *args: other(*tuplify(self(*args)))
        return Function(inside, self.dom, other.cod)

    def __call__(self, *xs):
        return self.inside(*xs)

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

    @classmethod
    def swap(cls, x: Ty, y: Ty) -> Function:
        """
        The function for swapping two tuples of types :code:`x` and :code:`y`.

        Parameters:
            x : The tuple of types on the left.
            y : The tuple of types on the right.
        """
        def inside(*xs):
            return untuplify(tuplify(xs)[len(x):] + tuplify(xs)[:len(x)])
        return cls(inside, dom=x + y, cod=y + x)

    braid = swap

    @staticmethod
    def copy(x: Ty, n: int) -> Function:
        """
        The function for making :code:`n` copies of a tuple of types :code:`x`.

        Parameters:
            x : The tuple of types to copy.
            n : The number of copies.
        """
        return Function(lambda *xs: n * xs, dom=x, cod=n * x)

    @classmethod
    def discard(cls, dom: Ty) -> Function:
        """
        The function discarding a tuple of types, i.e. making zero copies.

        Parameters:
            dom : The tuple of types to discard.
        """
        return cls.copy(dom, 0)

    @staticmethod
    def eval(base: Ty, exponent: Ty, left=True) -> Function:
        """
        The evaluation function,
        i.e. take a function and apply it to an argument.

        Parameters:
            base : The output type.
            exponent : The input type.
            left : Whether to take the function on the left or right.
        """
        if left:
            inside = lambda f, *xs: f(*xs)
            return Function(inside, exp(base, exponent) + exponent, base)
        inside = lambda *xs: xs[-1](*xs[:-1])
        return Function(inside, exponent + exp(base, exponent), base)

    def curry(self, n=1, left=True) -> Function:
        """
        Currying, i.e. turn a binary function into a function-valued function.

        Parameters:
            n : The number of types to curry.
            left : Whether to curry on the left or right.
        """
        inside = lambda *xs: lambda *ys: self(*(xs + ys) if left else (ys + xs))
        if left:
            dom = self.dom[:len(self.dom) - n]
            cod = exp(self.cod, self.dom[len(self.dom) - n:])
        else: dom, cod = self.dom[n:], exp(self.cod, self.dom[:n])
        return Function(inside, dom, cod)

    def uncurry(self, left=True) -> Function:
        """
        Uncurrying,
        i.e. turn a function-valued function into a binary function.

        Parameters:
            left : Whether to uncurry on the left or right.
        """
        base, exponent = self.cod[0].__args__[-1], self.cod[0].__args__[:-1]
        base = tuple(base.__args__) if is_tuple(base) else (base, )
        return self @ exponent >> Function.eval(base, exponent) if left\
            else exponent @ self >> Function.eval(base, exponent, left=False)

    exp = under = over = staticmethod(exp)

    def fix(self, n=1) -> Function:
        """
        The parameterised fixed point of a function.

        Parameters:
            n : The number of types to take the fixed point over.
        """
        if n > 1: return self.fix().fix(n - 1)
        dom, cod = self.dom[:-1], self.cod
        def inside(*xs, y=None):
            result = self.inside(*xs + (() if y is None else (y, )))
            return y if result == y else inside(*xs, y=result)
        return Function(inside, dom, cod)

    def trace(self, n=1):
        """
        The trace of a function.

        Parameters:
            n : The number of types to trace over.
        """
        dom, cod, traced = self.dom[:-n], self.cod[:-n], self.dom[-n:]
        fixed = (self >> self.discard(cod) @ traced).fix()
        return self.copy(dom) >> dom @ fixed\
            >> self >> cod @ self.discard(traced)
