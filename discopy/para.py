# -*- coding: utf-8 -*-

"""
The category of parametric maps over a symmetric underlying `category`.

A parametric map from `x` to `y` with parameter space `p` is a morphism
`x @ p -> y` in the underlying category. Composition tensors the parameters
and tensor routes them to the right with a swap. Parametric maps first
appeared in the study of supervised learning :cite:p:`FongEtAl19`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Symmetric
    Traced
    Markov
    Closed
    Feedback
    Compact
    Hypergraph

Axioms
------

Composition tensors the parameter spaces:

>>> from discopy.symmetric import Ty, Box, Diagram
>>> x, y, z, w, p, q = map(Ty, "xyzwpq")
>>> f = Symmetric(x, y, p, Box('f', x @ p, y))
>>> g = Symmetric(y, z, q, Box('g', y @ q, z))
>>> assert (f >> g).param == p @ q
>>> assert (f >> g).inside == f.inside @ q >> g.inside
>>> (f >> g).inside.draw(doctest="docs/_static/para/then.svg")

.. image:: /_static/para/then.svg
    :align: center

So does the tensor, with a swap routing the parameters to the right:

>>> h = Symmetric(z, w, q, Box('h', z @ q, w))
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

The identity and swap of :class:`Symmetric` are those of the underlying
category, with the empty parameter space:

>>> assert Symmetric.id(x) == Symmetric.lift(Diagram.id(x))
>>> assert Symmetric.swap(x, y) == Symmetric.lift(Diagram.swap(x, y))
>>> t = Traced(x @ y, z @ y, p, Box('t', x @ y @ p, z @ y))
>>> assert t.trace().dom == x and t.trace().param == p

The construction preserves each level of the hierarchy below symmetric:
:class:`Traced`, :class:`Markov`, :class:`Closed`, :class:`Feedback`,
:class:`Compact` and :class:`Hypergraph` lift the extra structure of their
underlying category with the empty parameter space, e.g.

>>> from discopy import frobenius
>>> X = frobenius.Ty('x')
>>> assert Hypergraph.spiders(1, 2, X)\\
...     == Hypergraph.lift(frobenius.Diagram.spiders(1, 2, X))

while the operations on morphisms swap the parameters out of the way,
the same as :meth:`Traced.trace`:

>>> from discopy import closed
>>> a, b, c, P = map(closed.Ty, "abcP")
>>> k = Closed(a @ b, c, P, closed.Box('k', a @ b @ P, c))
>>> assert k.curry(left=True).cod == c << b
>>> assert k.curry(left=False).cod == a >> c

A map may also carry a coparameter space `copar` on the codomain, i.e.
`inside : dom @ param -> cod @ copar`, empty by default — the type of one
time step of a stateful morphism sequence :cite:p:`DiLavoreEtAl22`, i.e. of
a :class:`Stream <discopy.stream.Stream>` with the delay forgotten.
Composition and tensor accumulate the hidden objects on both sides:

>>> m, n = Ty('m'), Ty('n')
>>> t = Symmetric(x, y, p, Box('t', x @ p, y @ m), m)
>>> u = Symmetric(y, z, q, Box('u', y @ q, z @ n), n)
>>> assert (t >> u).param == p @ q and (t >> u).copar == m @ n
>>> (t >> u).inside.draw(doctest="docs/_static/para/stateful-then.svg")

.. image:: /_static/para/stateful-then.svg
    :align: center

Coparametric maps, studied in categorical cybernetics
:cite:p:`CapucciEtAl21`, are the case of an empty `param`, composed by
accumulating the coparameters in forward order:

>>> f_ = Symmetric(x, y, Ty(), Box("f'", x, y @ m), m)
>>> g_ = Symmetric(y, z, Ty(), Box("g'", y, z @ n), n)
>>> assert (f_ >> g_).copar == m @ n
>>> assert (f_ >> g_).inside\\
...     == f_.inside >> g_.inside @ m >> z @ Diagram.swap(n, m)

Coreparametrisation post-composes the coparameters, covariantly where
:meth:`Symmetric.reparam` is contravariant:

>>> assert t.coreparam(Box('c', m, n)).copar == n

and the diagonal `param == copar` is closed under composition: it is the
free category with feedback of :cite:t:`KatisEtAl02`.

>>> s = Ty('s')
>>> v = Symmetric(x, y, s, Box('v', x @ s, y @ s), s)
>>> w = Symmetric(y, z, s, Box('w', y @ s, z @ s), s)
>>> assert (v >> w).param == (v >> w).copar == s @ s

Example
-------

Parametric maps compose like layers of a neural network, e.g. over
:class:`Function <discopy.python.Function>` with weight and bias parameters:

>>> from discopy.python import Function
>>> layer = Symmetric[Function]((float, ), (float, ), (float, float),
...     Function(lambda x, w, b: w * x + b, (float, ) * 3, (float, )))
>>> network = layer >> layer
>>> assert network.param == (float, ) * 4
>>> network.inside(2., 3., 1., .5, 0.)
3.5
"""

from __future__ import annotations

from dataclasses import dataclass

from discopy import symmetric, markov, closed, feedback, compact, frobenius
from discopy.abc import (
    ClosedCategory, CompactCategory, FeedbackCategory, HypergraphCategory,
    MarkovCategory, NamedGeneric, SymmetricCategory, TracedCategory)
from discopy.utils import (
    AxiomError, assert_iscomposable, assert_isinstance, classproperty,
    unbiased)


@dataclass
class Symmetric(SymmetricCategory, NamedGeneric['category']):
    """
    A parametric map from `dom` to `cod` with parameter space `param` is a
    morphism `inside : dom @ param -> cod` in an underlying `category`,
    optionally with a coparameter space `copar` on the codomain, i.e.
    `inside : dom @ param -> cod @ copar`.

    Parameters:
        dom (category.ob) : The domain of the parametric map.
        cod (category.ob) : The codomain of the parametric map.
        param (category.ob) : The parameter space of the map.
        inside (category) : The morphism ``dom @ param -> cod @ copar``.
        copar (category.ob) : The coparameter space, empty by default.

    .. admonition:: Summary

        .. autosummary::

            lift
            id
            then
            tensor
            swap
            reparam
            coreparam
    """
    category = symmetric.Diagram
    ob = classproperty(lambda cls: cls.category.ob)

    dom: ob
    cod: ob
    param: ob
    inside: category
    copar: ob = None

    def __post_init__(self):
        if self.copar is None:
            self.copar = self.ob()
        assert_isinstance(self.inside, self.category)
        if self.inside.dom != self.dom + self.param:
            raise AxiomError(
                f"{self.inside.dom} != {self.dom + self.param}")
        if self.inside.cod != self.cod + self.copar:
            raise AxiomError(
                f"{self.inside.cod} != {self.cod + self.copar}")

    @classmethod
    def lift(cls, inside: category) -> Symmetric:
        """
        A morphism of the underlying category as a parametric map with the
        empty parameter space, i.e. the injection functor from a category
        into its category of parametric maps.

        Parameters:
            inside : The morphism to lift.
        """
        return cls(inside.dom, inside.cod, cls.ob(), inside)

    @classmethod
    def id(cls, dom: ob = None) -> Symmetric:
        """
        The identity parametric map on `dom`, with empty parameter space.

        Parameters:
            dom : The domain of the identity, also its codomain.
        """
        return cls.lift(cls.category.id(cls.ob() if dom is None else dom))

    @unbiased
    def then(self, other: Symmetric) -> Symmetric:
        """
        Sequential composition tensors the hidden spaces on both sides,
        i.e. `(p, f) >> (q, g) == (p @ q, f @ q >> g)` for empty
        coparameters and the routing of :meth:`Stream.then
        <discopy.stream.Stream.then>` in general.

        Parameters:
            other : The parametric map to compose with.
        """
        assert_isinstance(other, type(self))
        assert_iscomposable(self, other)
        inside = self.inside @ other.param\
            >> self.cod @ self.category.swap(self.copar, other.param)\
            >> other.inside @ self.copar\
            >> other.cod @ self.category.swap(other.copar, self.copar)
        return type(self)(self.dom, other.cod, self.param + other.param,
                          inside, self.copar + other.copar)

    @unbiased
    def tensor(self, other: Symmetric) -> Symmetric:
        """
        Parallel composition tensors the hidden spaces on both sides, with
        swaps routing the parameters to the right of the domains and the
        coparameters to the right of the codomains.

        Parameters:
            other : The parametric map to tensor with.
        """
        assert_isinstance(other, type(self))
        inside = self.dom @ self.category.swap(other.dom, self.param)\
            @ other.param >> self.inside @ other.inside >> self.cod\
            @ self.category.swap(self.copar, other.cod) @ other.copar
        return type(self)(self.dom + other.dom, self.cod + other.cod,
                          self.param + other.param, inside,
                          self.copar + other.copar)

    @classmethod
    def swap(cls, left: ob, right: ob) -> Symmetric:
        """
        The swap of the underlying category, with empty parameter space.

        Parameters:
            left : The object on the left of the swap.
            right : The object on the right of the swap.
        """
        return cls.lift(cls.category.swap(left, right))

    def reparam(self, other: category) -> Symmetric:
        """
        Precompose the parameter space with `other : q -> param`, i.e. the
        2-cells of :class:`Symmetric`, kept as a method of the 1-cells the same
        way as :meth:`interchange <discopy.monoidal.Diagram.interchange>`.

        Parameters:
            other : The reparametrisation, a morphism into ``param``.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box
        >>> x, y, p, q = map(Ty, "xypq")
        >>> f = Symmetric(x, y, p, Box('f', x @ p, y))
        >>> r = Box('r', q, p)
        >>> f.reparam(r).inside.draw(doctest="docs/_static/para/reparam.svg")

        .. image:: /_static/para/reparam.svg
            :align: center
        """
        assert_iscomposable(self.dom @ other, self.inside)
        return type(self)(self.dom, self.cod, other.dom,
                          self.dom @ other >> self.inside, self.copar)

    def coreparam(self, other: category) -> Symmetric:
        """
        Post-compose the coparameter space with `other : copar -> q`,
        covariantly where :meth:`reparam` is contravariant.

        Parameters:
            other : The coreparametrisation, a morphism out of ``copar``.
        """
        assert_iscomposable(self.inside, self.cod @ other)
        return type(self)(self.dom, self.cod, self.param,
                          self.inside >> self.cod @ other, other.cod)


class Traced(Symmetric, TracedCategory):
    """
    Parametric maps over a traced symmetric underlying `category` form a
    traced category, with the parameters swapped out of the way.
    """
    def trace(self, n: int = 1, left: bool = False) -> Traced:
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
                              self.inside.trace(n, left=True), self.copar)
        inside = self.dom[:-n] @ self.category.swap(
            self.param, self.dom[-n:]) >> self.inside >> self.cod[:-n]\
            @ self.category.swap(self.cod[-n:], self.copar)
        return type(self)(self.dom[:-n], self.cod[:-n], self.param,
                          inside.trace(n), self.copar)


class Markov(Symmetric, MarkovCategory):
    """
    Parametric maps over a Markov underlying `category` form a Markov
    category, with the copy of the underlying category as :meth:`copy`.
    """
    category = markov.Diagram

    @classmethod
    def copy(cls, x: Symmetric.ob, n: int = 2) -> Markov:
        """
        The copy of the underlying category, with empty parameter space.

        Parameters:
            x : The object to copy.
            n : The number of copies.
        """
        return cls.lift(cls.category.copy(x, n))


class Closed(Markov, ClosedCategory):
    """
    Parametric maps over a closed underlying `category` form a closed
    category, currying with the parameters swapped out of the way.
    """
    category = closed.Diagram

    @classmethod
    def ev(cls, base: Symmetric.ob, exponent: Symmetric.ob, left: bool = True
           ) -> Closed:
        """
        The evaluation of the underlying category, with empty parameters.

        Parameters:
            base : The base of the exponential type.
            exponent : The exponent of the exponential type.
            left : Whether to take the left or right evaluation.
        """
        return cls.lift(cls.category.ev(base, exponent, left))

    def curry(self, n: int = 1, left: bool = False) -> Closed:
        """
        Curry the last `n` objects of the domain if `left` else the first,
        i.e. everything but the parameters, which a left currying swaps out
        of the way the same as :meth:`Traced.trace`.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry into a left or right exponential.
        """
        if not left:
            inside = self.inside.curry(n, left=False)
            return type(self)(self.dom[n:], inside.cod, self.param, inside)
        inside = self.dom[:-n] @ self.category.swap(
            self.param, self.dom[-n:]) >> self.inside
        inside = inside.curry(n, left=True)
        return type(self)(self.dom[:-n], inside.cod, self.param, inside)


class Feedback(Markov, FeedbackCategory):
    """
    Parametric maps over a feedback underlying `category` form a feedback
    category, with :meth:`delay` applied to all four components.
    """
    category = feedback.Diagram

    def delay(self, n_steps: int = 1) -> Feedback:
        """
        Delay a parametric map by delaying its underlying morphism together
        with its domain, codomain and parameter space.

        Parameters:
            n_steps : The number of time steps to delay.
        """
        return type(self)(*(x.delay(n_steps) for x in (
            self.dom, self.cod, self.param, self.inside, self.copar)))

    def feedback(self, dom: Symmetric.ob = None, cod: Symmetric.ob = None,
                 mem: Symmetric.ob = None) -> Feedback:
        """
        The feedback of the underlying category, with the parameters
        swapped out of the way the same as :meth:`Traced.trace`.

        Parameters:
            dom : The domain of the feedback.
            cod : The codomain of the feedback.
            mem : The memory type to trace over.
        """
        mem = self.cod[-1:] if mem is None else mem
        dom = self.dom[:len(self.dom) - len(mem)] if dom is None else dom
        cod = self.cod[:len(self.cod) - len(mem)] if cod is None else cod
        inside = dom @ self.category.swap(self.param, mem.delay())\
            >> self.inside >> cod @ self.category.swap(mem, self.copar)
        return type(self)(
            dom, cod, self.param,
            inside.feedback(dom + self.param, cod + self.copar, mem),
            self.copar)


class Compact(Traced, CompactCategory):
    """
    Parametric maps over a compact underlying `category` form a compact
    category, with the cups and caps of the underlying category.
    """
    category = compact.Diagram

    @classmethod
    def cups(cls, left: Symmetric.ob, right: Symmetric.ob) -> Compact:
        """
        The cups of the underlying category, with empty parameter space.

        Parameters:
            left : The left-hand side of the cups.
            right : Its adjoint, i.e. the right-hand side of the cups.
        """
        return cls.lift(cls.category.cups(left, right))

    @classmethod
    def caps(cls, left: Symmetric.ob, right: Symmetric.ob) -> Compact:
        """
        The caps of the underlying category, with empty parameter space.

        Parameters:
            left : The left-hand side of the caps.
            right : Its adjoint, i.e. the right-hand side of the caps.
        """
        return cls.lift(cls.category.caps(left, right))

    ev = classmethod(Closed.ev.__func__)
    curry = Closed.curry


class Hypergraph(Compact, Markov, HypergraphCategory):
    """
    Parametric maps over a hypergraph underlying `category` form a
    hypergraph category, with the spiders of the underlying category.
    """
    category = frobenius.Diagram

    @classmethod
    def spiders(cls, n_legs_in: int, n_legs_out: int, typ: Symmetric.ob
                ) -> Hypergraph:
        """
        The spiders of the underlying category, with empty parameters.

        Parameters:
            n_legs_in : The number of legs in for each spider.
            n_legs_out : The number of legs out for each spider.
            typ : The type of the spiders.
        """
        return cls.lift(cls.category.spiders(n_legs_in, n_legs_out, typ))
