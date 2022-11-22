# -*- coding: utf-8 -*-

"""
The free closed monoidal category, i.e. with exponential objects.

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

Axioms
------

:meth:`Diagram.curry` and :meth:`Diagram.uncurry` are inverses.

>>> x, y, z = map(Ty, "xyz")
>>> f, g = Box('f', y, z << x), Box('g', y, z >> x)

>>> from discopy.python import Function
>>> F = Functor(
...     ob={x: complex, y: bool, z: float},
...     ar={f: lambda y: lambda x: abs(x) ** 2 if y else 0,
...         g: lambda y: lambda z: z + 1j if y else -1j},
...     cod=Category(tuple[type, ...], Function))

>>> assert F(f.uncurry().curry())(True)(1j) == F(f)(True)(1j)
>>> assert F(g.uncurry(left=False).curry(left=False))(True)(1.2) == F(g)(True)(1.2)
"""

from __future__ import annotations

from discopy import symmetric
from discopy.cat import Category
from discopy.symmetric import Ty, hexagon
from discopy.hypergraph import coherence


class Diagram(symmetric.Diagram):
    @classmethod
    def copy(cls, x: Ty, n=2) -> Diagram:
        def factory(a, b, x, _):
            assert a == 1
            return Copy(x, b)
        return coherence(factory).__func__(cls, 1, n, x)

class Box(symmetric.Box, Diagram):
    cast = Diagram.cast

class Swap(symmetric.Swap, Box): pass

Diagram.swap = Diagram.braid = hexagon(Swap)

class Copy(Box):
    def __init__(self, x: Ty, n: int = 2):
        assert len(x) == 1
        super().__init__(name="Copy({}, {})".format(x, n), dom=x, cod=x ** n)

class Functor(symmetric.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Copy):
            return self.cod.ar.copy(self(other.dom), len(other.cod))
        return super().__call__(other)
