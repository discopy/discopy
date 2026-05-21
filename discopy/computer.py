# -*- coding: utf-8 -*-

"""
TODO https://arxiv.org/pdf/2208.03817v4
"""

from __future__ import annotations

from discopy import symmetric, closed, monoidal
from discopy.closed import Ty
from discopy.cat import factory
from discopy.utils import assert_isatomic, factory_name


@factory
class Ty(closed.Ty):
    """"""
    __ambiguous_inheritance__ = (closed.Ty, )


P = Ty("ℙ")

@factory
class Diagram(symmetric.Diagram, closed.Diagram):
    __ambiguous_inheritance__ = True

    ty_factory = Ty

    # @classmethod
    # def copy(cls, x: monoidal.Ty, n=2) -> Diagram:

    # @classmethod
    # def discard(cls, x: monoidal.Ty, n=2) -> Diagram:


class Box(symmetric.Box, closed.Box, Diagram):
    __ambiguous_inheritance__ = (symmetric.Box, closed.Box, )


class Swap(symmetric.Swap, Box):
    __ambiguous_inheritance__ = (symmetric.Swap, )


class Trace(symmetric.Trace, Box):
    __ambiguous_inheritance__ = (symmetric.Trace, )


### cartesian copy
class Copy(Box):
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        name = f"Copy({x}" + ("" if n == 2 else f", {n}") + ")"
        Box.__init__(self, name, dom=x, cod=x ** n,
                     draw_as_spider=True, color="black", drawing_name="")

### cartesian discard
class Discard(Box):
    def __init__(self, x: monoidal.Ty):
        assert_isatomic(x, monoidal.Ty)
        Box.__init__(self, f"Discard({x})", dom=x, cod=Ty())


class Eval(Box):
    def __init__(self, base, exponent):
        dom, cod = base, exponent
        super().__init__("{}", P @ dom, cod)


class Sum(symmetric.Sum, closed.Sum, Box):
    __ambiguous_inheritance__ = (symmetric.Sum, closed.Sum, )


class Category(symmetric.Category, closed.Category):
    ob, ar = Ty, Diagram


class Functor(symmetric.Functor, closed.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Eval):
            assert other.dom[0] == P
            return self.cod.ar.ev(self(other.dom[1]), self(other.cod))
        if isinstance(other, Copy):
            return self.cod.ar.copy(self(other.dom))
        if isinstance(other, Discard):
            return self.cod.ar.discard(self(other.dom))
        return super().__call__(other)


Diagram.copy_factory = Copy
Diagram.braid_factory = Swap
Diagram.trace_factory = Trace
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Id = Diagram.id
