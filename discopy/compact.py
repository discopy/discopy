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
    Category
    Functor

Axioms
------

>>> D = Diagram
>>> x, y = Ty('x'), Ty('y')

* Snake equations:

>>> left_snake = lambda x: Id(x.r).transpose(left=True)
>>> right_snake = lambda x: Id(x.l).transpose(left=False)
>>> assert left_snake(x) == Id(x) == right_snake(x)
>>> assert left_snake(x @ y) == Id(x @ y) == right_snake(x @ y)

* Yanking (a.k.a. Reidemeister move 1):

>>> right_loop = lambda x: x @ D.caps(x, x.r)\\
...     >> D.swap(x, x) @ x.r >> x @ D.cups(x, x.r)
>>> left_loop = lambda x: D.caps(x.r, x) @ x\\
...     >> x.r @ D.swap(x, x) >> D.cups(x.r, x) @ x
>>> top_loop = lambda x: D.caps(x, x.r) >> D.swap(x, x.r)
>>> bottom_loop = lambda x: D.swap(x, x.r) >> D.cups(x.r, x)
>>> reidemeister1 = lambda x:\\
...     top_loop(x) == D.caps(x.r, x)\\
...     and bottom_loop(x) == D.cups(x, x.r)\\
...     and left_loop(x) == Id(x) == right_loop(x)
>>> assert reidemeister1(x) and reidemeister1(x @ y) and reidemeister1(Ty())

* Coherence:

>>> assert D.caps(x @ y, y.r @ x.r)\\
...     == D.caps(x, x.r) @ D.caps(y, y.r) >> x @ D.swap(x.r, y @ y.r)
"""

from discopy import symmetric, ribbon, hypergraph
from discopy.cat import factory
from discopy.pivotal import Ty


@factory
class Diagram(symmetric.Diagram, ribbon.Diagram):
    """
    A compact diagram is a symmetric diagram and a ribbon diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (pivotal.Ty) : The domain of the diagram, i.e. its input.
        cod (pivotal.Ty) : The codomain of the diagram, i.e. its output.
    """
    ty_factory = Ty


class Box(symmetric.Box, ribbon.Box, Diagram):
    """
    A compact box is a symmetric and ribbon box in a compact diagram.

    Parameters:
        name (str) : The name of the box.
        dom (pivotal.Ty) : The domain of the box, i.e. its input.
        cod (pivotal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (symmetric.Box, ribbon.Box, )


class Cup(ribbon.Cup, Box):
    """
    A compact cup is a ribbon cup in a compact diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (ribbon.Cup, )


class Cap(ribbon.Cap, Box):
    """
    A compact cap is a ribbon cap in a compact diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (ribbon.Cap, )


class Swap(symmetric.Swap, ribbon.Braid, Box):
    """
    A compact swap is a symmetric swap and a ribbon braid.

    Parameters:
        left (pivotal.Ty) : The type on the top left and bottom right.
        right (pivotal.Ty) : The type on the top right and bottom left.
    """
    __ambiguous_inheritance__ = (symmetric.Swap, ribbon.Braid, )


class Category(symmetric.Category, ribbon.Category):
    """
    A compact category is both a symmetric category and a ribbon category.

    Parameters:
        ob : The objects of the category, default is :class:`pivotal.Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(symmetric.Functor, ribbon.Functor):
    """
    A compact functor is both a symmetric functor and a ribbon functor.

    Parameters:
        ob (Mapping[pivotal.Ty, pivotal.Ty]) :
            Map from atomic :class:`pivotal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Category()

    def __call__(self, other):
        if isinstance(other, Swap):
            return symmetric.Functor.__call__(self, other)
        return ribbon.Functor.__call__(self, other)


class Hypergraph(hypergraph.Hypergraph):
    category = Category()


Id = Diagram.id

Diagram.braid_factory = Swap
Diagram.hypergraph_factory, Diagram.functor_factory = Hypergraph, Functor
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
