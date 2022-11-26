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

Axioms
------
A tortile category is a braided pivotal category where wires can be untwisted.

>>> x = Ty('x')
>>> from discopy.drawing import Equation
>>> twist_l = Braid(x, x).trace(left=True)
>>> twist_r = Braid(x, x).trace(left=False)
>>> eq = Equation(twist_l >> twist_l[::-1], Id(x), twist_r >> twist_r[::-1])
>>> eq.draw(figsize=(6, 4), margins=(.2, .1),
...         path='docs/imgs/tortile/twist-untwist.png')

.. image:: /imgs/tortile/twist-untwist.png
    :align: center

A tortile category is strict whenever the twist is the identity.

>>> eq_strict = Equation(twist_l, Id(x), twist_r)
>>> eq_strict.draw(figsize=(4, 2), margins=(.2, .1),
...                path='docs/imgs/tortile/strict.png')

.. image:: /imgs/tortile/strict.png
    :align: center

Note
----
The diagram of tortile categories should be drawn with ribbons, i.e. two
parallel wires with the twist drawn as the braid.

Strict tortile categories have diagrams with knots, i.e. ribbons where the two
parallel wires coincide and the twist is the identity.
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
    def trace(self, n=1, left=False):
        """
        The trace of a tortile diagram.

        Parameters:
            n : The number of wires to trace.
        """
        if left:
            return self.caps(self.dom[:n].r, self.dom[:n]) @ self.dom[n:]\
                >> self.dom[:n].r @ self\
                >> self.cups(self.cod[:n].r, self.cod[:n]) @ self.cod[n:]
        return self.dom[:-n] @ self.caps(self.dom[-n:], self.dom[-n:].r)\
            >> self @ self.dom[-n:].r\
            >> self.cod[:-n] @ self.cups(self.cod[-n:], self.cod[-n:].r)

    def cup(self, x, y):
        """
        Post-compose a tortile diagram with a cup between wires ``i`` and ``j``
        by introducing braids.

        Parameters:
            i : The wire on the left of the cup.
            j : The wire on the right of the cup.
        """
        if min(x, y) < 0 or max(x, y) >= len(self.cod):
            raise ValueError(f'Indices {x, y} are out of range.')
        x, y = min(x, y), max(x, y)
        for i in range(x, y - 1):
            braid = self.braid_factory(self.cod[i], self.cod[i + 1])
            self = self >> self.cod[:i] @ braid @ self.cod[i + 2:]
        cup = self.cup_factory(self.cod[y - 1], self.cod[y])
        return self >> self.cod[:y - 1] @ cup @ self.cod[y + 1:]

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
    z = 0

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


Id = Diagram.id
