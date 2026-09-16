# -*- coding: utf-8 -*-

"""
The free compact category, i.e. diagrams with swaps, cups and caps.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Cup
    Cap
    Permutation
    Swap
    Sum
    Bubble
    Functor

Axioms
------

>>> x, y = Ty('x'), Ty('y')

Snake equations
===============

>>> snake = Equation(Id(x.l).transpose(left=True), Id(x), Id(x.r).transpose())
>>> assert snake
>>> snake.draw(doctest="docs/_static/compact/snake.svg")

.. image:: /_static/compact/snake.svg
    :align: center

Yanking
=======
a.k.a. Reidemeister move 1

>>> cap_yanking = Equation(Cap(x, x.r) >> Swap(x, x.r), Cap(x.r, x))
>>> cup_yanking = Equation(Swap(x, x.r) >> Cup(x.r, x), Cup(x, x.r))
>>> assert cap_yanking and cup_yanking
>>> Equation(cap_yanking, cup_yanking, symbol='', space=1).draw(
...     doctest="docs/_static/compact/yanking_cup_and_cap.svg")

.. image:: /_static/compact/yanking_cup_and_cap.svg
    :align: center

Coherence
=========

>>> assert Equation(Diagram.caps(x @ y, y.r @ x.r),
...     Cap(x, x.r) @ Cap(y, y.r) >> x @ Diagram.swap(x.r, y @ y.r))
"""

from discopy import symmetric, ribbon, rigid, cmap, hypergraph
from discopy.abc import CompactCategory
from discopy.cat import factory, Generator
from discopy.utils import deprecated_alias
from discopy.pivotal import Wire, Ty  # noqa: F401


class Layer(symmetric.Layer, rigid.Layer):
    """ A compact layer with permutation plumbing and rigid rotation. """


@factory
class Diagram(symmetric.Diagram, ribbon.Diagram, CompactCategory):
    """
    A compact diagram is a symmetric diagram and a ribbon diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (pivotal.Ty) : The domain of the diagram, i.e. its input.
        cod (pivotal.Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    layer_factory = Layer

    @Generator()
    def permutation_factory(cls):
        return Permutation


Box, Cup, Cap = (
    Diagram.generator_factory, Diagram.cup_factory, Diagram.cap_factory)


class Permutation(symmetric.Permutation, Box):
    """
    A compact permutation is a symmetric permutation in a compact category.

    Parameters:
        dom (pivotal.Ty) : The domain, i.e. the wires to permute.
        perm : The permutation as a :class:`finset.Permutation` or a list.
    """
    def rotate(self, left=False):
        dom = self.cod.l if left else self.cod.r
        return type(self)(dom, self.perm.rotate())

    l = property(lambda self: self.rotate(left=True))
    r = property(lambda self: self.rotate(left=False))


Swap, Sum, Bubble = (
    Diagram.swap_factory, Diagram.sum_factory, Diagram.bubble_factory)


class Functor(symmetric.Functor, ribbon.Functor):
    """
    A compact functor is both a symmetric functor and a ribbon functor.

    Parameters:
        ob_map (Mapping[pivotal.Ty, pivotal.Ty]) :
            Map from atomic :class:`pivotal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, (symmetric.Swap, symmetric.Permutation)):
            return symmetric.Functor.__call__(self, other)
        return ribbon.Functor.__call__(self, other)


CMap = cmap.CMap[Diagram]

Id = Diagram.id

Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of compact diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)


__getattr__ = deprecated_alias(__name__, {"Ob": "Wire"})
