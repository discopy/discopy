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
    Functor
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from discopy import cat
from discopy.cat import Category, Composable
from discopy.monoidal import Whiskerable
from discopy.utils import inductive


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
        inside (Callable) : The callable Python object inside the function.
        dom (python.Ty) : The domain of the function, i.e. its input type.
        cod (python.Ty) : The codomain of the function, i.e. its output type.
    """
    inside: Callable
    dom: Ty
    cod: Ty

    @classmethod
    def id(cls, dom: type) -> Function:
        return cls(lambda *xs: untuplify(xs), dom, dom)

    @inductive
    def then(self, other: Function) -> Function:
        assert self.cod == other.dom
        inside = lambda *args: other(*tuplify(self(*args)))
        return Function(inside, self.dom, other.cod)

    def __call__(self, *xs):
        return self.inside(*xs)

    @inductive
    def tensor(self, other: Function) -> Function:
        def inside(*xs):
            left, right = xs[:len(self.dom)], xs[len(self.dom):]
            return untuplify(tuplify(self(*left)) + tuplify(other(*right)))
        return Function(inside, self.dom + other.dom, self.cod + other.cod)

    @classmethod
    def swap(cls, x: Ty, y: Ty) -> Function:
        def inside(*xs):
            return untuplify(tuplify(xs)[len(x):] + tuplify(xs)[:len(x)])
        return cls(inside, dom=x + y, cod=y + x)

    braid = swap

    @staticmethod
    def copy(x: Ty, n: int):
        return Function(lambda *xs: n * xs, dom=x, cod=n * x)

    def curry(self, n=1, left=True) -> Function:
        inside = lambda *xs: lambda *ys: self(*(xs + ys) if left else (ys + xs))
        if left:
            dom = self.dom[:len(self.dom) - n]
            cod = exp(self.cod, self.dom[len(self.dom) - n:])
        else: dom, cod = self.dom[n:], exp(self.cod, self.dom[:n])
        return Function(inside, dom, cod)

    @staticmethod
    def eval(base: Ty, exponent: Ty, left=True) -> Function:
        if left:
            inside = lambda f, *xs: f(*xs)
            return Function(inside, exp(base, exponent) + exponent, base)
        inside = lambda *xs: xs[-1](*xs[:-1])
        return Function(inside, exponent + exp(base, exponent), base)

    def uncurry(self, left=True) -> Function:
        base, exponent = self.cod[0].__args__[-1], self.cod[0].__args__[:-1]
        base = tuple(base.__args__) if is_tuple(base) else (base, )
        return self @ exponent >> Function.eval(base, exponent) if left\
            else exponent @ self >> Function.eval(base, exponent, left=False)

    exp = under = over = staticmethod(exp)

    def fix(self, n=1):
        if n > 1: return self.fix().fix(n - 1)
        dom, cod = self.dom[:-1], self.cod
        def inside(*xs, y=None):
            result = self.inside(*xs + (() if y is None else (y, )))
            return y if result == y else inside(*xs, y=result)
        return Function(inside, dom, cod)

    def trace(self, n=1):
        dom, cod, traced = self.dom[:-n], self.cod[:-n], self.dom[-n:]
        fixed = (self >> self.discard(cod) @ traced).fix()
        return self.copy(dom) >> dom @ fixed\
            >> self >> cod @ self.discard(traced)


class Functor(cat.Functor):
    dom = cod = Category(Ty, Function)

    def __call__(self, other):
        if isinstance(other, Function):
            return self.ar[other]
        if isinstance(other, tuple):
            return tuple(map(self, other))
        if isinstance(other, type):
            return self.ob[other]
        raise TypeError
