# -*- coding: utf-8 -*-

"""
The category of parametric maps over a symmetric underlying `category`.

A parametric map from `x` to `y` with parameter space `p` is a morphism
`x @ p -> y` in the underlying category. Composition tensors the parameters
and tensor routes them to the right with a swap. Parametric maps first
appeared in the study of supervised learning :cite:p:`FongEtAl19`, the
construction is defined in generality with actegories in
:cite:p:`CapucciEtAl22`, see also section 3.2.1 of :cite:t:`Gavranovic24`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Para

Axioms
------

Composition tensors the parameter spaces:

>>> from discopy.symmetric import Ty, Box, Diagram
>>> x, y, z, w, p, q = map(Ty, "xyzwpq")
>>> f = Para(x, y, p, Box('f', x @ p, y))
>>> g = Para(y, z, q, Box('g', y @ q, z))
>>> assert (f >> g).param == p @ q
>>> assert (f >> g).inside == f.inside @ q >> g.inside
>>> (f >> g).inside.draw(doctest="docs/_static/para/then.svg")

.. image:: /_static/para/then.svg
    :align: center

So does the tensor, with a swap routing the parameters to the right:

>>> h = Para(z, w, q, Box('h', z @ q, w))
>>> assert (f @ h).param == p @ q
>>> assert (f @ h).inside\\
...     == x @ Diagram.swap(z, p) @ q >> f.inside @ h.inside
>>> (f @ h).inside.draw(doctest="docs/_static/para/tensor.svg")

.. image:: /_static/para/tensor.svg
    :align: center

Reparametrisation precomposes the parameters, contravariantly:

>>> p_, p__ = Ty("p'"), Ty("p''")
>>> r, s = Box('r', p_, p), Box('s', p__, p_)
>>> assert f.reparam(r).param == p_
>>> assert f.reparam(s >> r) == f.reparam(r).reparam(s)

The identity, swap and trace of :class:`Para` are those of the underlying
category, with the empty parameter space:

>>> assert Para.id(x).inside == Diagram.id(x)
>>> assert Para.swap(x, y).inside == Diagram.swap(x, y)
>>> t = Para(x @ y, z @ y, p, Box('t', x @ y @ p, z @ y))
>>> assert t.trace().dom == x and t.trace().param == p

Example
-------

Parametric maps compose like layers of a neural network, e.g. over
:class:`Function <discopy.python.Function>` with weight and bias parameters:

>>> from discopy.python import Function
>>> layer = Para[Function]((float, ), (float, ), (float, float),
...     Function(lambda x, w, b: w * x + b, (float, ) * 3, (float, )))
>>> network = layer >> layer
>>> assert network.param == (float, ) * 4
>>> network.inside(2., 3., 1., .5, 0.)
3.5
"""

from __future__ import annotations

from dataclasses import dataclass

from discopy import symmetric
from discopy.abc import NamedGeneric, SymmetricCategory
from discopy.utils import (
    AxiomError, assert_isinstance, classproperty, unbiased)


@dataclass
class Para(SymmetricCategory, NamedGeneric['category']):
    """
    A parametric map from `dom` to `cod` with parameter space `param` is a
    morphism `inside : dom @ param -> cod` in an underlying `category`.

    Parameters:
        dom (category.ob) : The domain of the parametric map.
        cod (category.ob) : The codomain of the parametric map.
        param (category.ob) : The parameter space of the map.
        inside (category) : The underlying morphism ``dom @ param -> cod``.

    .. admonition:: Summary

        .. autosummary::

            id
            then
            tensor
            swap
            trace
            reparam
    """
    category = symmetric.Diagram
    ob = classproperty(lambda cls: cls.category.ob)

    dom: ob
    cod: ob
    param: ob
    inside: category

    def __post_init__(self):
        assert_isinstance(self.inside, self.category)
        if self.inside.dom != self.dom + self.param:
            raise AxiomError(
                f"{self.inside.dom} != {self.dom + self.param}")
        if self.inside.cod != self.cod:
            raise AxiomError(f"{self.inside.cod} != {self.cod}")

    @classmethod
    def id(cls, dom: ob = None) -> Para:
        """
        The identity parametric map on `dom`, with empty parameter space.

        Parameters:
            dom : The domain of the identity, also its codomain.
        """
        dom = cls.ob() if dom is None else dom
        return cls(dom, dom, cls.ob(), cls.category.id(dom))

    @unbiased
    def then(self, other: Para) -> Para:
        """
        Sequential composition tensors the parameter spaces, i.e.
        `(p, f) >> (q, g) == (p @ q, f @ q >> g)`.

        Parameters:
            other : The parametric map to compose with.
        """
        assert_isinstance(other, type(self))
        if not self.is_composable(other):
            raise AxiomError(f"{self.cod} != {other.dom}")
        return type(self)(self.dom, other.cod, self.param + other.param,
                          self.inside @ other.param >> other.inside)

    @unbiased
    def tensor(self, other: Para) -> Para:
        """
        Parallel composition tensors the parameter spaces, with a swap
        routing them to the right of the domains.

        Parameters:
            other : The parametric map to tensor with.
        """
        assert_isinstance(other, type(self))
        inside = self.dom @ self.category.swap(other.dom, self.param)\
            @ other.param >> self.inside @ other.inside
        return type(self)(self.dom + other.dom, self.cod + other.cod,
                          self.param + other.param, inside)

    @classmethod
    def swap(cls, left: ob, right: ob) -> Para:
        """
        The swap of the underlying category, with empty parameter space.

        Parameters:
            left : The object on the left of the swap.
            right : The object on the right of the swap.
        """
        return cls(left + right, right + left, cls.ob(),
                   cls.category.swap(left, right))

    def trace(self, n: int = 1, left: bool = False) -> Para:
        """
        The trace of a parametric map is the trace of the underlying
        morphism, with the parameters swapped out of the way.

        Parameters:
            n : The number of objects to trace over.
            left : Whether to trace the wires on the left or right.
        """
        if n == 0:
            return self
        if left:
            return type(self)(self.dom[n:], self.cod[n:], self.param,
                              self.inside.trace(n, left=True))
        inside = self.dom[:-n] @ self.category.swap(
            self.param, self.dom[-n:]) >> self.inside
        return type(self)(
            self.dom[:-n], self.cod[:-n], self.param, inside.trace(n))

    def reparam(self, arrow: category) -> Para:
        """
        Precompose the parameter space with `arrow : q -> param`, i.e. the
        2-cells of :class:`Para`, kept as a method of the 1-cells the same
        way as :meth:`interchange <discopy.monoidal.Diagram.interchange>`.

        Parameters:
            arrow : The reparametrisation, a morphism into ``param``.
        """
        if arrow.cod != self.param:
            raise AxiomError(f"{arrow.cod} != {self.param}")
        return type(self)(self.dom, self.cod, arrow.dom,
                          self.dom @ arrow >> self.inside)
