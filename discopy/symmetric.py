# -*- coding: utf-8 -*-

"""
The free symmetric category, i.e. diagrams with swaps.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Category
    Functor

Axioms
------
The dagger of :code:`Swap(x, y)` is :code:`Swap(y, x)`.

>>> x, y, z = map(Ty, "xyz")
>>> assert Diagram.swap(x, y @ z)[::-1] == Diagram.swap(y @ z, x)

Swaps have their dagger as inverse, up to :meth:`braided.Diagram.simplify`.

>>> swap_unswap = Swap(x, y) >> Swap(y, x)
>>> assert swap_unswap.simplify() == Id(x @ y)
>>> from discopy import drawing
>>> drawing.equation(swap_unswap, Id(x @ y),
...     path='docs/_static/imgs/symmetric/inverse.png', figsize=(3, 2))

.. image:: ../_static/imgs/symmetric/inverse.png
    :align: center

The hexagon equations hold on the nose.

>>> left_hexagon = Swap(x, y) @ z >> y @ Swap(x, z)
>>> assert left_hexagon == Diagram.swap(x, y @ z)
>>> right_hexagon = x @ Swap(y, z) >> Swap(x, z) @ y
>>> assert right_hexagon == Diagram.swap(x @ y, z)
>>> drawing.equation(left_hexagon, right_hexagon, symbol='', space=2,
...     path='docs/_static/imgs/symmetric/hexagons.png', figsize=(5, 2))

.. image:: ../_static/imgs/symmetric/hexagons.png
    :align: center
"""

from __future__ import annotations

from discopy import monoidal, braided
from discopy.cat import factory
from discopy.monoidal import Ty, PRO
from discopy.utils import BinaryBoxConstructor, assert_isatomic, factory_name


@factory
class Diagram(braided.Diagram):
    """
    A symmetric diagram is a braided diagram with :class:`Swap` boxes.

    Parameters:
        inside (tuple[monoidal.Layer, ...]) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    @classmethod
    def swap(cls, left: monoidal.Ty, right: monoidal.Ty) -> Diagram:
        """
        Returns a diagram that swaps the left with the right wires.

        Parameters:
            left : The type at the top left and bottom right.
            right : The type at the top right and bottom left.

        Note
        ----
        This calls :func:`braided.hexagon` and :attr:`braid_factory`.
        """
        return cls.braid(left, right)

    @classmethod
    def permutation(cls, xs: list[int], dom: monoidal.Ty = None) -> Diagram:
        """
        Construct the diagram representing a given permutation.

        Parameters:
            xs : A list of integers representing a permutation.
            dom : A type of the same length as :code:`permutation`,
                  default is :code:`PRO(len(permutation))`.
        """
        if set(range(len(xs))) != set(xs):
            raise ValueError("Input should be a permutation of range(n).")
        dom = PRO(len(xs)) if dom is None else dom
        if len(dom) != len(xs):
            raise ValueError(
                "Domain and permutation should have the same length.")
        if len(dom) <= 1:
            return cls.id(dom)
        i = xs[0]
        return cls.swap(dom[:i], dom[i]) @ dom[i + 1:]\
            >> dom[i] @ cls.permutation(
                [x - 1 if x > i else x for x in xs[1:]], dom[:i] + dom[i + 1:])

    def permute(self, *xs: int) -> Diagram:
        """
        Returns :code:`self >> self.permutation(list(xs), self.dom)`.

        Parameters:
            xs : A list of integers representing a permutation.

        Examples
        --------
        >>> x, y, z = Ty('x'), Ty('y'), Ty('z')
        >>> assert Id(x @ y @ z).permute(2, 1, 0).cod == z @ y @ x
        """
        return self >> self.permutation(list(xs), self.cod)


class Box(braided.Box, Diagram):
    """
    A symmetric box is a braided box in a symmetric diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Swap(braided.Braid, Box):
    """
    The swap of atomic types :code:`left` and :code:`right`.

    Parameters:
        left : The type on the top left and bottom right.
        right : The type on the top right and bottom left.

    Important
    ---------
    :class:`Swap` is only defined for atomic types (i.e. of length 1).
    For complex types, use :meth:`Diagram.swap` instead.
    """
    def __init__(self, left, right):
        braided.Braid.__init__(self, left, right)
        Box.__init__(self, self.name, self.dom, self.cod,
                     draw_as_wires=True, draw_as_braid=False)

    def dagger(self):
        return type(self)(self.right, self.left)


class Category(braided.Category):
    """
    A symmetric category is a braided category with a method :code:`swap`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(monoidal.Functor):
    """
    A symmetric functor is a monoidal functor that preserves swaps.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Swap):
            return self.cod.ar.swap(self(other.dom[0]), self(other.dom[1]))
        return super().__call__(other)


Diagram.braid_factory = Swap
Id = Diagram.id
