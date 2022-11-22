# -*- coding: utf-8 -*-

"""
The free traced category, i.e.
diagrams with swaps where outputs can feedback into inputs.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Trace
    Category
    Functor
"""
from __future__ import annotations

from discopy import monoidal, symmetric
from discopy.cat import factory
from discopy.monoidal import Ty


@factory
class Diagram(symmetric.Diagram):
    """
    A traced diagram is a symmetric diagram with :class:`Trace` boxes.

    Parameters:
        inside (tuple[monoidal.Layer, ...]) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    def trace(self, n=1) -> Diagram:
        """
        Wrapper around :class:`Trace`, called by :class:`Functor`.

        Parameters:
            n : The number of output wires to feedback into inputs.
        """
        return Trace(self, n)


class Box(symmetric.Box, Diagram):
    """
    A traced box is a symmetric box in a traced diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (symmetric.Box, )


class Trace(Box, monoidal.Bubble):
    """
    Make :code:`n` output wires feedback into inputs.

    Parameters:
        arg : The diagram to trace.
        n : The number of output wires to feedback into inputs.
    """
    def __init__(self, arg: Diagram, n=1):
        name = "Trace({}, {})".format(arg, n)
        dom, cod = arg.dom[:-n], arg.cod[:-n]
        monoidal.Bubble.__init__(self, arg, dom, cod)
        Box.__init__(self, name, dom, cod)


class Category(symmetric.Category):
    """
    A traced category is a symmetric category with a method :code:`trace`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(symmetric.Functor):
    """
    A cartesian functor is a symmetric functor that preserves traces.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Trace):
            n = len(self(other.diagram.dom)) - len(self(other.dom))
            return self.cod.ar.trace(self(other.diagram), n)
        return super().__call__(other)
