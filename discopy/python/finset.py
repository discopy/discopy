# -*- coding: utf-8 -*-

"""
The category of finite sets implemented as Python dictionaries.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Function
"""

from __future__ import annotations

from dataclasses import dataclass
from discopy.cat import Composable
from discopy.utils import Whiskerable


@dataclass
class Function(Composable[int], Whiskerable):
    """
    A function between finite sets encoded as a Python dictionary.

    Parameters:
        inside : The dictionary from `range(dom)` to `range(cod)`.
        dom : The size of domain of the function.
        cod : The size of codomain of the function.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            tensor
            swap
            copy
    """
    inside: dict[int, int]
    dom: int
    cod: int

    def __getitem__(self, key):
        return self.inside[key]

    @staticmethod
    def id(x: int = 0):
        return Function({i: i for i in range(x)}, x, x)

    def then(self, other: Function) -> Function:
        inside = {i: self[other[i]] for i in range(other.cod)}
        return Function(inside, self.dom, other.cod)

    def tensor(self, other: Function) -> Function:
        inside = {i: self[i] for i in range(self.cod)}
        inside.update({
            self.cod + i: self.dom + other[i] for i in range(other.cod)})
        return Function(inside, self.dom + other.dom, self.cod + other.cod)

    @staticmethod
    def swap(x: int, y: int) -> Function:
        inside = {i: i + x if i < x else i - x for i in range(x + y)}
        return Function(inside, x + y, x + y)

    @staticmethod
    def copy(x: int, n=2) -> Function:
        return Function({i: i % x for i in range(n * x)}, x, n * x)
