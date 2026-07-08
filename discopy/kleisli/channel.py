# -*- coding: utf-8 -*-

"""
The Kleisli category of a :class:`discopy.kleisli.monad.Monad`, i.e. Python
functions returning monadic values, with Kleisli composition.

The class :class:`Channel` is parameterised by a monad ``M``, e.g.
``Channel[Maybe]`` is the category of partial functions.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Channel

Example
-------
>>> from discopy.kleisli.monad import Maybe, Nothing
>>> half = Channel[Maybe](
...     lambda n: n // 2 if n % 2 == 0 else Nothing, int, int)
>>> assert (half >> half)(4) == 1 and (half >> half)(6) is Nothing
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from discopy.abc import Category, NamedGeneric
from discopy.utils import (
    assert_iscomposable, assert_isinstance, tuplify, untuplify, ar_factory)


@ar_factory
@dataclass
class Channel(Category, NamedGeneric['monad']):
    """
    A channel is a Python function from values to monadic values, i.e. an
    arrow in the Kleisli category of the :attr:`monad` parameter.

    Parameters:
        inside : The callable Python object inside the channel, which takes
            values as arguments and returns a monadic value.
        dom : The domain of the channel, i.e. its tuple of input types.
        cod : The codomain of the channel, i.e. its tuple of output types.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            lift
    """
    inside: Callable
    dom: tuple
    cod: tuple

    ob = tuple[type, ...]
    monad = None

    def __init__(self, inside: Callable, dom: tuple, cod: tuple):
        dom, cod = map(tuplify, (dom, cod))
        self.inside, self.dom, self.cod = inside, dom, cod

    @classmethod
    def id(cls, dom: tuple = ()) -> Channel:
        """
        The identity channel on a tuple of types ``dom``, i.e. ``pure``.

        Parameters:
            dom : The tuple of types on which to take the identity.
        """
        return cls(lambda *xs: cls.monad.pure(untuplify(xs)), dom, dom)

    def then(self, other: Channel) -> Channel:
        """
        The Kleisli composition of two channels, called with ``>>``.

        Parameters:
            other : The other channel to compose in sequence.
        """
        assert_isinstance(other, Channel)
        if self.monad != other.monad:
            raise ValueError(
                f"Cannot compose channels for {self.monad} and {other.monad}")
        assert_iscomposable(self, other)
        return type(self)(
            lambda *xs: self.monad.bind(
                self(*xs), lambda ys: other.inside(*tuplify(ys))),
            self.dom, other.cod)

    @classmethod
    def lift(cls, f: Callable, dom: tuple, cod: tuple) -> Channel:
        """
        The pure channel for a deterministic callable ``f``, i.e. the image of
        a Python function under the canonical functor into the Kleisli
        category.

        Parameters:
            f : A callable from values to values.
            dom : The domain of the channel.
            cod : The codomain of the channel.
        """
        return cls(lambda *xs: cls.monad.pure(f(*xs)), dom, cod)

    def __call__(self, *xs):
        return self.inside(*xs)
