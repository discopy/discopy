# -*- coding: utf-8 -*-

"""
The free cartesian category, i.e. with natural cocommutative comonoids.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Copy
    Category
    Functor
"""

from __future__ import annotations

from discopy import symmetric, monoidal
from discopy.cat import factory
from discopy.monoidal import Ty
from discopy.frobenius import coherence
from discopy.utils import assert_isatomic


@factory
class Diagram(symmetric.Diagram):
    """
    A cartesian diagram is a symmetric diagram with :class:`Copy` boxes.

    Parameters:
        inside (tuple[monoidal.Layer, ...]) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    @classmethod
    def copy(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Make :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        def factory(a, b, x, _):
            assert a == 1
            return Copy(x, b)
        return coherence(factory).__func__(cls, 1, n, x)


class Box(symmetric.Box, Diagram):
    """
    A cartesian box is a symmetric box in a cartesian diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (symmetric.Box, )


class Swap(symmetric.Swap, Box):
    """
    A cartesian swap is a symmetric swap in a cartesian diagram.

    Parameters:
        left (monoidal.Ty) : The type on the top left and bottom right.
        right (monoidal.Ty) : The type on the top right and bottom left.
    """
    __ambiguous_inheritance__ = (symmetric.Swap, )


Diagram.braid_factory = Swap


class Copy(Box):
    """
    The copy of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type to copy.
        n : The number of copies.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        super().__init__(name="Copy({}, {})".format(x, n), dom=x, cod=x ** n)


class Category(symmetric.Category):
    """
    A cartesian category is a symmetric category with a method :code:`copy`.

    Parameters:
        ob : The type of objects.
        ar : The type of arrows.
    """
    ob, ar = monoidal.Ty, Diagram


class Functor(symmetric.Functor):
    """
    A cartesian functor is a symmetric functor that preserves copies.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.
    """
    dom = cod = Category(monoidal.Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Copy):
            return self.cod.ar.copy(self(other.dom), len(other.cod))
        return super().__call__(other)
