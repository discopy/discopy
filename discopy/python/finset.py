# -*- coding: utf-8 -*-

"""
The category of finite sets implemented as Python lists.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Function
    Cycles
    Permutation
"""

from __future__ import annotations
from typing import Iterable, Self, Any
from collections.abc import Sequence

from dataclasses import dataclass

from discopy.abc import MonoidalCategory, SymmetricCategory


@dataclass
class Function(MonoidalCategory, Sequence):
    """
    A function between finite sets encoded as a Python list.

    Functions implement the standard Python sequence protocol.

    Parameters:
        inside : The list from ``range(cod)`` to ``range(dom)``.
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
    inside: list[int]
    dom: int
    cod: int

    ob = int

    def __post_init__(self):
        if isinstance(self.inside, dict):
            self.inside = [self.inside[i] for i in range(self.cod)]
        else:
            self.inside = list(self.inside)
        if len(self.inside) != self.cod:
            raise ValueError

    def __getitem__(self, key):
        return self.inside[key]

    def __len__(self) -> int:
        return self.cod

    @staticmethod
    def id(x: int = 0):
        return Function(list(range(x)), x, x)

    def then(self, other: Function) -> Function:
        inside = [self[other[i]] for i in range(other.cod)]
        return Function(inside, self.dom, other.cod)

    def tensor(self, other: Function) -> Function:
        inside = list(self.inside) + [
            self.dom + other[i] for i in range(other.cod)]
        return Function(inside, self.dom + other.dom, self.cod + other.cod)

    @staticmethod
    def swap(x: int, y: int) -> Function:
        inside = list(Permutation.swap(x, y))
        return Function(inside, x + y, x + y)

    @staticmethod
    def copy(x: int, n=2) -> Function:
        return Function([i % x for i in range(n * x)], x, n * x)


type Cycle = Iterable[int]
type Cycles = Iterable[Cycle]


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
        super().__init__(list(inside), size, size)

    def __iter__(self):
        return (self[i] for i in range(self.cod))

    def __getitem__(self, key: int) -> int:
        if isinstance(key, slice):
            return tuple(self)[key]
        return super().__getitem__(key)

    def __repr__(self) -> str:
        return repr(tuple(self))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Permutation):
            return tuple(self) == tuple(other)
        if isinstance(other, tuple):
            return tuple(self) == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(tuple(self))

    @classmethod
    def id(cls, dom: int = 0) -> Self:
        """ The identity permutation on ``range(size)``. """
        return cls(range(dom), dom)

    identity = id

    @property
    def is_identity(self) -> bool:
        """ Whether this is the identity permutation. """
        return list(self) == list(range(len(self)))

    @classmethod
    def from_cycles(cls, cycles: Cycles, size: int) -> Self:
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
    def from_transpositions(cls, transpositions: Cycles, size: int) -> Self:
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

    def cycles(self) -> Cycles:
        """ Return the cycles of the permutation. """
        result, seen = [], set()
        for i in range(len(self)):
            if i in seen:
                continue
            result.append(self.cycle(i, seen))
        return tuple(result)

    def cycle(
            self, start: int,
            seen: set[int] | None = None) -> Cycle:
        """ Return the cycle reached from ``start``. """
        if start < -len(self) or start >= len(self):
            raise ValueError
        start %= len(self)
        cycle, local_seen, i = [], set() if seen is None else seen, start
        while i not in local_seen:
            local_seen.add(i)
            cycle.append(i)
            i = self[i]
        return tuple(cycle)

    def then(self, other: Self) -> Self:
        """ Return ``self ; other``, i.e. ``result[i] == other[self[i]]``. """
        other = type(self)(other, len(self))
        elems = (other[self[i]] for i in range(len(self)))
        return type(self)(elems, len(self))

    def dagger(self) -> Self:
        """ Return the inverse permutation. """
        result = list(range(len(self)))
        for source, target in enumerate(self):
            result[target] = source
        return type(self)(result, len(self))

    def conjugate(self, other: Self) -> Self:
        """ Return ``other^-1 ; self ; other``. """
        other = type(self)(other, len(self))
        return other.dagger().then(self).then(other)

    def tensor(self, other=None, *others) -> Self:
        """ Return the disjoint union of permutations. """
        if other is None:
            return self
        other = type(self)(other)
        shift = len(self)
        result = type(self)(
            tuple(self) + tuple(shift + i for i in other),
            len(self) + len(other))
        return result.tensor(*others)

    def embed(self, injection: Iterable[int], size: int) -> Self:
        """ Embed into ``range(size)`` along ``injection``. """
        injection = tuple(injection)
        complement = tuple(i for i in range(size) if i not in injection)
        relabeling = type(self)(injection + complement, size)
        embedded = self.tensor(type(self).id(size - len(self)))
        return embedded.conjugate(relabeling)

    def coequalizer(self, other: Self) -> dict[int, int]:
        """
        Coequalize two permutations by quotienting generated orbits.

        This computes the quotient of ``range(len(self))`` by the equivalence
        relation generated by ``i ~ self[i]`` and ``i ~ other[i]``.

        Equivalently, it returns the orbits of the subgroup generated by the
        two permutations. The returned quotient map ``q`` satisfies
        ``q[i] == q[self[i]]`` and ``q[i] == q[other[i]]`` for every element
        ``i``, and is minimal with this property.

        Parameters:
            other : The second permutation, with the same size as ``self``.

        Returns:
            A dictionary mapping each element to its quotient component.
        """
        other = type(self)(other, len(self))
        component_of = {}
        component = 0
        for start in range(len(self)):
            if start in component_of:
                continue
            stack = [start]
            while stack:
                i = stack.pop()
                if i in component_of:
                    continue
                component_of[i] = component
                for j in (self[i], other[i]):
                    if j not in component_of:
                        stack.append(j)
            component += 1
        return component_of

    @classmethod
    def swap(cls, left: int, right: int) -> Self:
        inside = tuple(
            i + right if i < left else i - left
            for i in range(left + right))
        return cls(inside, left + right)

    def trace(self, n: int = 1, left: bool = False) -> Self:
        raise NotImplementedError

    def is_fixpoint_free_involution(self) -> bool:
        """ Whether this is a product of disjoint 2-cycles. """
        return all(self[i] != i and self[self[i]] == i
                   for i in range(len(self)))
