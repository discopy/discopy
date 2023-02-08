# -*- coding: utf-8 -*-

"""
The Int construction.

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


@dataclass
class Ty:
    """
    An integer type is a pair of natural types.

    Parameters:
        positive (Ty) : The positive half of the type.
        negative (Ty) : The negative half of the type.

    Note
    ----
    The prefix operator ``-`` reverses positive and negative, e.g.

    >>> x, y, z = map(Ty, "xyz")
    >>>
    """
    positive: balanced.Ty
    negative: balanced.Ty

    def __init__(self, positive: balanced.Ty, negative: balanced.Ty = None):
        self.positive = positive
        self.negative = positive.factory() if negative is None else negative

    def __iter__(self):
        yield self.positive
        yield self.negative

    def __getitem__(self, key):
        if key == 0:
            return self.positive
        if key == 1:
            return self.negative
        raise IndexError

    def tensor(self, other: Ty):
        return Ty(self[0] @ other[0], self[1] @ other[1])

    __matmul__ = tensor

    def __neg__(self):
        return Ty(self[1], self[0])


@dataclass
class Diagram:
    """
    An integer diagram from ``x @ -y`` to ``z @ -w`` is a natural diagram from
    ``x @ z`` to ``y @ w``.
    """
    def __init__(self, inside: balanced.Diagram, dom: Ty, cod: Ty):
        self.inside, self.dom, self.cod = inside, dom, cod

    def then(self, other: Diagram = None, *others: Diagram):
        assert_iscomposable(self, other)
