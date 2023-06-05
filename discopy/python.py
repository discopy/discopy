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
        tuplify
        untuplify
        is_tuple
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from discopy.cat import Composable, assert_iscomposable
from discopy.utils import tuplify, untuplify, Whiskerable


Ty = tuple[type, ...]


def exp(base: Ty, exponent: Ty) -> Ty:
    """
    The exponential of a tuple of Python types by another.

    Parameters:
        base (python.Ty) : The base type.
        exponent (python.Ty) : The exponent type.
    """
    return (Callable[list(exponent), tuple[base]], )


def is_tuple(typ: type) -> bool:
    """
    Whether a given type is tuple or a paramaterised tuple.

    Parameters:
        typ : The type to check for equality with tuple.
    """
    return getattr(typ, "__origin__", typ) is tuple


@dataclass
class Function(Composable[Ty], Whiskerable):
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
            ev
            curry
            uncurry
            fix
            trace
    """
    inside: Callable
    dom: Ty
    cod: Ty

    def id(dom: Ty) -> Function:
        """
        The identity function on a given tuple of types :code:`dom`.

        Parameters:
            dom (python.Ty) : The typle of types on which to take the identity.
        """
        return Function(lambda *xs: untuplify(xs), dom, dom)

    def then(self, other: Function) -> Function:
        """
        The sequential composition of two functions, called with :code:`>>`.

        Parameters:
            other : The other function to compose in sequence.
        """
        assert_iscomposable(self, other)
        return Function(
            lambda *args: other(*tuplify(self(*args))), self.dom, other.cod)

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
        base, exponent = self.cod[0].__args__[-1], self.cod[0].__args__[:-1]
        base = tuple(base.__args__) if is_tuple(base) else (base, )
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
        The trace of a function.

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


@dataclass
class Dict(Composable[int], Whiskerable):
    inside: dict[int, int]
    dom: int
    cod: int

    def __getitem__(self, key):
        return self.inside[key]

    @staticmethod
    def id(x: int = 0):
        return Dict({i: i for i in range(x)}, x, x)

    def then(self, other: Dict) -> Dict:
        inside = {i: self[other[i]] for i in range(other.cod)}
        return Dict(inside, self.dom, other.cod)

    def tensor(self, other: Dict) -> Dict:
        inside = {i: self[i] for i in range(self.cod)}
        inside.update({
            self.cod + i: self.dom + other[i] for i in range(other.cod)})
        return Dict(inside, self.dom + other.dom, self.cod + other.cod)

    @staticmethod
    def swap(x: int, y: int) -> Dict:
        inside = {i: i + x if i < x else i - x for i in range(x + y)}
        return Dict(inside, x + y, x + y)

    @staticmethod
    def copy(x: int, n=2) -> Dict:
        return Dict({i: i % x for i in range(n * x)}, x, n * x)
