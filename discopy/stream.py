from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

from discopy.utils import Composable, Whiskerable, NamedGeneric
from discopy import symmetric


@dataclass
class Ty(NamedGeneric['base']):
    base = symmetric.Ty

    now: base
    later: Callable[[], Ty[base]]

    def delay(self, n_steps=1):
        return self if not n_steps else self.later().delay(n_steps - 1)


class ConstantTy(Ty):
    def __init__(self, x):
        super().__init__(x, lambda: self)


@dataclass
class Stream(Composable, Whiskerable, NamedGeneric['category']):
    category = symmetric.Category

    dom: category.ob
    cod: category.ob
    now: category.ar
    later: Callable[[], Stream[category]]

    delay = Ty.delay

    def id(): ...
    def then(): ...
    def tensor(): ...
    def swap(): ...
    def feedback(): ...


class Constant(Stream):
    def __init__(self, diagram):
        dom, cod = map(ConstantTy, (diagram.dom, diagram.cod))
        super().__init__(dom, cod, diagram, lambda: self)
