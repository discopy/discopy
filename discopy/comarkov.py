# -*- coding: utf-8 -*-

"""
The free comarkov category, i.e. a symmetric category with a supply of
commutative monoid, the dual of :mod:`markov`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Permutation
    Merge
    Functor


Axioms
------

>>> x = Ty('x')

>>> merge, unit = Merge(x), Merge(x, n=0)

Commutative monoid
==================

>>> unitality = Equation(unit @ x >> merge, Id(x), x @ unit >> merge)
>>> associativity = Equation(merge @ x >> merge, x @ merge >> merge)
>>> commutativity = Equation(Swap(x, x) >> merge, merge)
>>> assert unitality and associativity and commutativity
>>> Equation(unitality, associativity, commutativity, symbol='').draw(
...     path="docs/_static/frobenius/monoid.svg")

.. image:: /_static/frobenius/monoid.svg
    :align: center

Coherence
=========

>>> assert Equation(Diagram.merge(x @ x, n=0), unit @ unit)
>>> assert Equation(Diagram.merge(x @ x),
...     x @ Swap(x, x) @ x >> merge @ merge)

Note
----
Equality of comarkov diagrams is computed by translation to hypergraph.
Merge boxes are translated to spiders, thus when they appear in the same
diagram as the copy boxes of :mod:`markov` they automatically satisfy the
:mod:`frobenius` axioms.
"""

from __future__ import annotations

from discopy import symmetric, monoidal, hypergraph, messages
from discopy.abc import ComarkovCategory
from discopy.cat import factory
from discopy.monoidal import Ty  # noqa: F401
from discopy.utils import AxiomError, assert_isatomic, factory_name


@factory
class Diagram(symmetric.Diagram, ComarkovCategory):
    """
    A comarkov diagram is a symmetric diagram with :class:`Merge` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        if phase is not None or n_legs_out != 1:
            raise ValueError
        return cls.merge_factory(typ, n_legs_in)

    @classmethod
    def merge(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Merge :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to merge.
            n : The number of copies.
        """
        from discopy import frobenius
        return frobenius.Diagram.spiders.__func__(cls, n, 1, x)

    @classmethod
    def unit(cls, x: monoidal.Ty) -> Diagram:
        """
        The unit of a type :code:`x`.

        Parameters:
            x : The type of the unit.
        """
        return cls.merge(x, 0)


class Box(symmetric.Box, Diagram):
    """
    A comarkov box is a symmetric box in a comarkov diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Swap(symmetric.Swap, Box):
    """
    Symmetric swap in a comarkov diagram.

    Parameters:
        left (monoidal.Ty) : The type on the top left and bottom right.
        right (monoidal.Ty) : The type on the top right and bottom left.
    """


class Permutation(symmetric.Permutation, Box):
    """
    A permutation in a comarkov category.

    Parameters:
        dom (monoidal.Ty) : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.
    """


class Trace(symmetric.Trace, Box):
    """
    A trace in a comarkov category.

    Parameters:
        arg : The diagram to trace.
        left : Whether to trace the wires on the left or right.

    See also
    --------
    :meth:`Diagram.trace`
    """


class Merge(Box):
    """
    The merge of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type of wires to merge.
        n : The number of wires to merge.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        name = f"Merge({x}" + ("" if n == 2 else f", {n}") + ")"
        Box.__init__(self, name, dom=x ** n, cod=x,
                     draw_as_spider=True, color="black", drawing_name="")

    def __new__(cls, x: monoidal.Ty, n: int = 2):
        return super().__new__(cls) if n else\
            cls.unit_factory.__new__(cls.unit_factory, x)

    def dagger(self):
        if not hasattr(self.ar, "copy_factory"):
            raise AxiomError(messages.NOT_A_DAGGER.format(self))
        return self.ar.copy_factory(self.cod, len(self.dom))

    def __repr__(self):
        return (
            factory_name(type(self)) + f"({repr(self.cod)}, {len(self.dom)})")


class Unit(Merge):
    """
    The unit of an atomic type :code:`x`.

    Parameters:
        x : The type of the unit.
    """
    def __init__(self, x: monoidal.Ty, *args, **kwargs):
        super().__init__(x, 0)


class Sum(symmetric.Sum, Box):
    """
    A comarkov sum is a symmetric sum and a comarkov box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(symmetric.Functor):
    """
    A comarkov functor is a symmetric functor that preserves merges.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Merge):
            return self.cod.merge(self(other.cod), len(other.dom))
        return super().__call__(other)


class CMap(symmetric.CMap):
    category = Diagram


Diagram.functor_factory = Functor
Diagram.map_factory = CMap
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.merge_factory = Merge
Diagram.braid_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.trace_factory = Trace
Diagram.unit_factory = Unit
Diagram.sum_factory = Sum
Id = Diagram.id


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of comarkov diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)
