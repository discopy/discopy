# -*- coding: utf-8 -*-

"""
The Int construction of Joyal, Street & Verity :cite:t:`JoyalEtAl96`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Exp
    Over
    Under
    Diagram
    Box
    Eval
    Curry
    Category
    Functor
"""

from __future__ import annotations
from dataclasses import dataclass

from discopy import balanced
from discopy.cat import Composable, assert_iscomposable
from discopy.monoidal import Whiskerable
from discopy.utils import NamedGeneric, mmap


@dataclass
class Ty(NamedGeneric('natural')):
    """
    An integer type is a pair of ``natural`` types.

    Parameters:
        positive (Ty) : The positive half of the type.
        negative (Ty) : The negative half of the type.

    Note
    ----
    The prefix operator ``-`` reverses positive and negative, e.g.

    >>> x, y, z = map(Ty[int], [1, 2, 3])
    >>> assert x @ -y @ z == Ty[int](1 + 3, 2)
    """
    positive: 'natural'
    negative: 'natural'

    def __init__(self, positive: 'natural', negative: 'natural' = None):
        if negative is None:
            negative = self.natural()
        positive, negative = (
            x if isinstance(x, self.natural) else self.natural(x)
            for x in (positive, negative))
        self.positive, self.negative = positive, negative

    def __iter__(self):
        yield self.positive
        yield self.negative

    def tensor(self, *others: Ty):
        positive, negative = (
            sum([getattr(x, attr) for x in (self, ) + others], self.natural())
            for attr in ["positive", "negative"])
        return Ty(positive, negative)

    __matmul__ = tensor

    def __neg__(self):
        positive, negative = self
        return Ty(negative, positive)


@dataclass
class Diagram(Composable[Ty], Whiskerable, NamedGeneric('natural')):
    """
    An integer diagram from ``x`` to ``y`` is a ``natural`` diagram
    from ``x.negative @ y.positive`` to ``x.positive @ y.negative``.
    """
    def __init__(self, inside: 'natural', dom: Ty, cod: Ty):
        assert_isinstance(inside, self.natural)
        if inside.dom != dom.negative @ cod.positive:
            raise ValueError
        if inside.cod != dom.positive @ cod.negative:
            raise ValueError
        self.inside, self.dom, self.cod = inside, dom, cod

    @mmap
    def then(self, other: Diagram):
        """
        The composition of two integer diagrams is given by the following
        composition of natural diagrams:

        >>> from discopy.compact import Ty as T, Diagram as D
        >>> u, v, w, x, y, z = map(Ty[T], "uvwxyz")
        >>> f = Diagram[D](Box('f', x @ v, y @ u), x @ -u, y @ -v)
        >>> g = Diagram[D](Box('g', y @ w, z @ v), y @ -v, z @ -w)
        >>> (f >> g).draw()
        """
        assert_iscomposable(self, other)
        x, v = self.dom
        y, u = self.cod
        z, w = other.cod.positive, other.dom.negative
        dom, cod = x @ w, z @ u
        inside = (
            x @ braid(w, v)
            >> self.inside @ w
            >> y @ braid(w, u)[::-1]
            >> other.inside @ u
            >> z @ braid(v, u)).trace()
        return type(self)(inside, dom, cod)
