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
    Swap
    Functor

Axioms
------

>>> from discopy.drawing import Equation
>>> Diagram.use_hypergraph_equality = True
>>> x, y = Ty('x'), Ty('y')

Snake equations
===============

>>> snake = Equation(Id(x.l).transpose(left=True), Id(x), Id(x.r).transpose())
>>> assert snake
>>> snake.draw(path="docs/_static/compact/snake.png")

.. image:: /_static/compact/snake.png
    :align: center

Yanking
=======
a.k.a. Reidemeister move 1

>>> cap_yanking = Equation(Cap(x, x.r) >> Swap(x, x.r), Cap(x.r, x))
>>> cup_yanking = Equation(Swap(x, x.r) >> Cup(x.r, x), Cup(x, x.r))
>>> assert cap_yanking and cup_yanking
>>> Equation(cap_yanking, cup_yanking, symbol='', space=1).draw(
...     path="docs/_static/compact/yanking_cup_and_cap.png")

.. image:: /_static/compact/yanking_cup_and_cap.png
    :align: center

Coherence
=========

>>> assert Diagram.caps(x @ y, y.r @ x.r)\\
...     == Cap(x, x.r) @ Cap(y, y.r) >> x @ Diagram.swap(x.r, y @ y.r)

>>> Diagram.use_hypergraph_equality = False
"""

from discopy import symmetric, ribbon, hypergraph
from discopy.abc import CompactCategory
from discopy.cat import ar_factory
from discopy.pivotal import Ob, Ty  # noqa: F401


@ar_factory
class Diagram(symmetric.Diagram, ribbon.Diagram, CompactCategory):
    """
    A compact diagram is a symmetric diagram and a ribbon diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (pivotal.Ty) : The domain of the diagram, i.e. its input.
        cod (pivotal.Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty
    trace_factory = ribbon.Diagram.trace_factory


class Box(symmetric.Box, ribbon.Box, Diagram):
    """
    A compact box is a symmetric and ribbon box in a compact diagram.

    Parameters:
        name (str) : The name of the box.
        dom (pivotal.Ty) : The domain of the box, i.e. its input.
        cod (pivotal.Ty) : The codomain of the box, i.e. its output.
    """


class Cup(ribbon.Cup, Box):
    """
    A compact cup is a ribbon cup in a compact diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """


class Cap(ribbon.Cap, Box):
    """
    A compact cap is a ribbon cap in a compact diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """


class Swap(symmetric.Swap, ribbon.Braid, Box):
    """
    A compact swap is a symmetric swap and a ribbon braid.

    Parameters:
        left (pivotal.Ty) : The type on the top left and bottom right.
        right (pivotal.Ty) : The type on the top right and bottom left.
    """


class Functor(symmetric.Functor, ribbon.Functor):
    """
    A compact functor is both a symmetric functor and a ribbon functor.

    Parameters:
        ob (Mapping[pivotal.Ty, pivotal.Ty]) :
            Map from atomic :class:`pivotal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Swap):
            return symmetric.Functor.__call__(self, other)
        return ribbon.Functor.__call__(self, other)


class CMap(symmetric.CMap):
    category = Diagram
    require_oriented = False
    require_connected = False


Id = Diagram.id

Diagram.braid_factory = Swap
Diagram.functor_factory = Functor
Diagram.map_factory = CMap
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
