# -*- coding: utf-8 -*-

"""
The category of finite sets implemented as Python dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from discopy.cat import Composable, assert_iscomposable, assert_isinstance
from discopy.utils import Whiskerable


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
