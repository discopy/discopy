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
"""

from discopy import symmetric, tortile
from discopy.cat import factory
from discopy.tortile import Ty


@factory
class Diagram(symmetric.Diagram, tortile.Diagram):
    """
    A compact diagram is a symmetric diagram and a tortile diagram.

    Parameters:
        inside (tuple[rigid.Layer, ...]) : The layers of the diagram.
        dom (pivotal.Ty) : The domain of the diagram, i.e. its input.
        cod (pivotal.Ty) : The codomain of the diagram, i.e. its output.
    """
    def trace(self, n=1):
        """
        The trace of a compact diagram.

        Parameters:
            n : The number of wires to trace.
        """
        return self.dom[:-n] @ self.caps(self.dom[-n:], self.dom[-n:].r)\
            >> self @ self.dom[-n:].r\
            >> self.cod[:-n] @ self.cups(self.cod[-n:], self.cod[-n:].r)

    def cup(self, x, y):
        if min(x, y) < 0 or max(x, y) >= len(self.cod):
            raise ValueError(f'Indices {x, y} are out of range.')
        x, y = min(x, y), max(x, y)
        for i in range(x, y - 1):
            t0, t1 = self.cod[i:i + 1], self.cod[i + 1:i + 2]
            self >>= Id(self.cod[:i]) @ Swap(t0, t1) @ Id(self.cod[i + 2:])
        t0, t1 = self.cod[y - 1:y], self.cod[y:y + 1]
        self >>= Id(self.cod[:y - 1]) @ Cup(t0, t1) @ Id(self.cod[y + 1:])
        return self


class Box(symmetric.Box, tortile.Box, Diagram):
    """
    A compact box is a symmetric and tortile box in a compact diagram.

    Parameters:
        name (str) : The name of the box.
        dom (pivotal.Ty) : The domain of the box, i.e. its input.
        cod (pivotal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (symmetric.Box, tortile.Box, )


class Cup(tortile.Cup, Box):
    """
    A compact cup is a tortile cup in a compact diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (tortile.Cup, )


class Cap(tortile.Cap, Box):
    """
    A compact cap is a tortile cap in a compact diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (tortile.Cap, )


class Swap(symmetric.Swap, tortile.Braid, Box):
    """
    A compact swap is a symmetric swap and a tortile braid.

    Parameters:
        left (pivotal.Ty) : The type on the top left and bottom right.
        right (pivotal.Ty) : The type on the top right and bottom left.
    """
    __ambiguous_inheritance__ = (symmetric.Swap, tortile.Braid, )


Diagram.braid_factory = Swap
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap


class Category(symmetric.Category, tortile.Category):
    """
    A compact category is both a symmetric category and a tortile category.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(symmetric.Functor, tortile.Functor):
    """
    A compact functor is both a symmetric functor and a tortile functor.

    Parameters:
        ob (Mapping[pivotal.Ty, pivotal.Ty]) :
            Map from atomic :class:`pivotal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Swap):
            return symmetric.Functor.__call__(self, other)
        return tortile.Functor.__call__(self, other)
