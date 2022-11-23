# -*- coding: utf-8 -*-

"""
The free tortile category, i.e. diagrams with braids, cups and caps.

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
    Braid
    Category
    Functor
"""

from __future__ import annotations

from discopy import braided, pivotal
from discopy.cat import factory
from discopy.pivotal import Ty


@factory
class Diagram(pivotal.Diagram, braided.Diagram):
    """
    A tortile diagram is a pivotal diagram and a braided diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (pivotal.Ty) : The domain of the diagram, i.e. its input.
        cod (pivotal.Ty) : The codomain of the diagram, i.e. its output.
    """

class Box(pivotal.Box, braided.Box, Diagram):
    """
    A tortile box is a pivotal and braided box in a tortile diagram.

    Parameters:
        name (str) : The name of the box.
        dom (pivotal.Ty) : The domain of the box, i.e. its input.
        cod (pivotal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (pivotal.Box, braided.Box, )

class Cup(pivotal.Cup, Box):
    """
    A tortile cup is a pivotal cup in a tortile diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (pivotal.Cup, )

class Cap(pivotal.Cap, Box):
    """
    A tortile cap is a pivotal cap in a tortile diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (pivotal.Cap, )

class Braid(braided.Braid, Box):
    """
    A tortile braid is a braided braid in a tortile diagram.

    Parameters:
        left (pivotal.Ty) : The type on the top left and bottom right.
        right (pivotal.Ty) : The type on the top right and bottom left.
        is_dagger (bool) : Braiding over or under.
    """
    __ambiguous_inheritance__ = (braided.Braid, )

Diagram.braid_factory = Braid
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap


class Category(pivotal.Category, braided.Category):
    """
    A tortile category is both a pivotal category and a braided category.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(pivotal.Functor, braided.Functor):
    """
    A tortile functor is both a pivotal functor and a braided functor.

    Parameters:
        ob (Mapping[pivotal.Ty, pivotal.Ty]) :
            Map from atomic :class:`pivotal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Braid):
            return braided.Functor.__call__(self, other)
        return pivotal.Functor.__call__(self, other)
