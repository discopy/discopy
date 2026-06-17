# -*- coding: utf-8 -*-

"""
The category of finite sets implemented as Python dictionaries.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Permutation
    Function
"""

from __future__ import annotations

from dataclasses import dataclass

from discopy.abc import MonoidalCategory, SymmetricCategory


@dataclass
class Function(MonoidalCategory):
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

    ob = int

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
        inside = dict(enumerate(Permutation.swap(x, y)))
        return Function(inside, x + y, x + y)

    @staticmethod
    def copy(x: int, n=2) -> Function:
        return Function({i: i % x for i in range(n * x)}, x, n * x)


class Permutation(Function, SymmetricCategory):
    """
    A permutation of a finite set, seen as a bijective finite-set function.

    A permutation is represented by its action on ``range(n)``.

    Examples
    --------
    >>> Permutation((1, 0, 3, 2)).cycles()
    ((0, 1), (2, 3))
    >>> Permutation.from_cycles([(0, 1), (2, 3)], 4)
    (1, 0, 3, 2)
    >>> Permutation((1, 0)).is_fixpoint_free_involution()
    True
    """
    ob = int

    def __init__(self, inside=(), size: int | None = None):
        inside = tuple(inside)
        if size is None:
            size = len(inside)
        if len(inside) != size:
            raise ValueError
        if sorted(inside) != list(range(size)):
            raise ValueError
        super().__init__(dict(enumerate(inside)), size, size)

    def __iter__(self):
        return (self[i] for i in range(self.cod))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return tuple(self)[key]
        return super().__getitem__(key % len(self))

    def __len__(self):
        return self.cod

    def __repr__(self):
        return repr(tuple(self))

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return tuple(self) == tuple(other)
        if isinstance(other, tuple):
            return tuple(self) == other
        return super().__eq__(other)

    def __hash__(self):
        return hash(tuple(self))

    @classmethod
    def id(cls, dom: int = 0):
        return cls(range(dom), dom)

    @classmethod
    def identity(cls, size: int):
        """ The identity permutation on ``range(size)``. """
        return cls.id(size)

    @classmethod
    def from_cycles(cls, cycles, size: int):
        """ Build a permutation from cycles. """
        result = list(range(size))
        seen = set()
        for cycle in map(tuple, cycles):
            if len(set(cycle)) != len(cycle):
                raise ValueError
            for i in cycle:
                if i < 0 or i >= size or i in seen:
                    raise ValueError
                seen.add(i)
            for source, target in zip(cycle, cycle[1:] + cycle[:1]):
                result[source] = target
        return cls(result, size)

    @classmethod
    def from_transpositions(cls, transpositions, size: int):
        """ Build a permutation from disjoint 2-cycles. """
        result = list(range(size))
        seen = set()
        for left, right in transpositions:
            if left == right:
                raise ValueError
            if left < 0 or right < 0 or left >= size or right >= size:
                raise ValueError
            if left in seen or right in seen:
                raise ValueError
            seen.update([left, right])
            result[left], result[right] = right, left
        return cls(result, size)

    def cycles(self) -> tuple[tuple[int, ...], ...]:
        """ Return the cycles of the permutation. """
        result, seen = [], set()
        for i in range(len(self)):
            if i in seen:
                continue
            result.append(self.cycle(i, seen))
        return tuple(result)

    def cycle(
            self, start: int,
            seen: set[int] | None = None) -> tuple[int, ...]:
        """ Return the cycle reached from ``start``. """
        if start < 0 or start >= len(self):
            raise ValueError
        cycle, local_seen, i = [], set() if seen is None else seen, start
        while i not in local_seen:
            local_seen.add(i)
            cycle.append(i)
            i = self[i]
        return tuple(cycle)

    def then(self, other):
        return self.compose(other)

    def compose(self, other):
        """ Return ``self o other``, i.e. ``result[i] == self[other[i]]``. """
        other = type(self)(other, len(self))
        elems = (self[other[i]] for i in range(len(self)))
        return type(self)(elems, len(self))

    def inverse(self):
        """ Return the inverse permutation. """
        result = list(range(len(self)))
        for source, target in enumerate(self):
            result[target] = source
        return type(self)(result, len(self))

    def conjugate(self, by):
        """ Return ``by o self o by^-1``. """
        by = type(self)(by, len(self))
        return by.compose(self).compose(by.inverse())

    def tensor(self, other=None, *others):
        """ Return the disjoint union of permutations. """
        if other is None:
            return self
        other = type(self)(other)
        shift = len(self)
        result = type(self)(
            tuple(self) + tuple(shift + i for i in other),
            len(self) + len(other))
        return result.tensor(*others)

    @classmethod
    def swap(cls, left: int, right: int):
        inside = tuple(
            i + right if i < left else i - left
            for i in range(left + right))
        return cls(inside, left + right)

    def trace(self, n: int = 1, left: bool = False):
        raise NotImplementedError

    def is_fixpoint_free_involution(self) -> bool:
        """ Whether this is a product of disjoint 2-cycles. """
        return all(self[i] != i and self[self[i]] == i
                   for i in range(len(self)))
