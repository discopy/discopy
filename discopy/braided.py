# -*- coding: utf-8 -*-

"""
The free braided category, i.e. diagrams with braids.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Braid
    Category
    Functor

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        hexagon

Axioms
------
Braids have their dagger as inverse, up to :meth:`Diagram.simplify`.

>>> x, y, z = map(Ty, "xyz")
>>> LHS = Braid(x, y) >> Braid(x, y)[::-1]
>>> RHS = Braid(y, x)[::-1] >> Braid(y, x)
>>> assert LHS.simplify() == Id(x @ y) == RHS.simplify()

>>> from discopy.drawing import Equation
>>> Equation(LHS, Id(x @ y), RHS).draw(
...     path='docs/_static/braided/inverse.png', figsize=(5, 2))

.. image:: /_static/braided/inverse.png
    :align: center

The hexagon equations hold on the nose.

>>> left_hexagon = Braid(x, y) @ z >> y @ Braid(x, z)
>>> assert left_hexagon == Diagram.braid(x, y @ z)
>>> right_hexagon = x @ Braid(y, z) >> Braid(x, z) @ y
>>> assert right_hexagon == Diagram.braid(x @ y, z)

>>> Equation(left_hexagon, right_hexagon, symbol='').draw(
...     space=2, path='docs/_static/braided/hexagons.png', figsize=(5, 2))

.. image:: /_static/braided/hexagons.png
    :align: center
"""

from __future__ import annotations

from collections.abc import Callable

from discopy import monoidal
from discopy.cat import factory
from discopy.monoidal import Ty, assert_isatomic
from discopy.utils import factory_name, from_tree


@factory
class Diagram(monoidal.Diagram):
    """
    A braided diagram is a monoidal diagram with :class:`Braid` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    __ambiguous_inheritance__ = True

    @classmethod
    def braid(cls, left: monoidal.Ty, right: monoidal.Ty) -> Diagram:
        """
        The diagram braiding :code:`left` over :code:`right`.

        Parameters:
            left : The type on the top left and bottom right.
            right : The type on the top right and bottom left.

        Note
        ----
        This calls :func:`hexagon` and :attr:`braid_factory`.
        """
        return hexagon(cls, cls.braid_factory)(left, right)

    def simplify(self) -> Diagram:
        """ Remove braids followed by their dagger. """
        for i, ((x, f, _), (y, g, _)) in enumerate(
                zip(self.inside, self.inside[1:])):
            if x == y and isinstance(f, Braid) and f == g[::-1]:
                inside = self.inside[:i] + self.inside[i + 2:]
                return self.factory(
                    inside, self.dom, self.cod, _scan=False).simplify()
        return self


class Box(monoidal.Box, Diagram):
    """
    A braided box is a monoidal box in a braided diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (monoidal.Box, )


class BinaryBoxConstructor:
    """
    Box constructor with attributes ``left`` and ``right`` as input.

    Parameters:
        left : Some attribute on the left.
        right : Some attribute on the right.
    """
    def __init__(self, left, right):
        self.left, self.right = left, right

    def __repr__(self):
        return factory_name(type(self))\
            + f"({repr(self.left)}, {repr(self.right)})"

    def to_tree(self) -> dict:
        """ Serialise a binary box constructor. """
        left, right = self.left.to_tree(), self.right.to_tree()
        return dict(factory=factory_name(type(self)), left=left, right=right)

    @classmethod
    def from_tree(cls, tree: dict) -> BinaryBoxConstructor:
        """ Decode a serialised binary box constructor. """
        return cls(*map(from_tree, (tree['left'], tree['right'])))


class Braid(BinaryBoxConstructor, Box):
    """
    The braiding of atomic types :code:`left` and :code:`right`.

    Parameters:
        left : The type on the top left and bottom right.
        right : The type on the top right and bottom left.
        is_dagger : Braiding over or under.

    Important
    ---------
    :class:`Braid` is only defined for atomic types (i.e. of length 1).
    For complex types, use :meth:`Diagram.braid` instead.
    """
    def __init__(self, left: monoidal.Ty, right: monoidal.Ty, is_dagger=False):
        assert_isatomic(left, monoidal.Ty)
        assert_isatomic(right, monoidal.Ty)
        name = type(self).__name__\
            + (f"({right}, {left})" if is_dagger else f"({left}, {right})")
        dom, cod = left @ right, right @ left
        Box.__init__(
            self, name, dom, cod, is_dagger=is_dagger, draw_as_braid=True)
        BinaryBoxConstructor.__init__(self, left, right)

    def __repr__(self):
        str_is_dagger = ", is_dagger=True" if self.is_dagger else ""
        return factory_name(type(self)) + \
            f"({repr(self.left)}, {repr(self.right)}{str_is_dagger})"

    def dagger(self):
        return type(self)(self.right, self.left, not self.is_dagger)


def hexagon(cls: type, factory: Callable) -> Callable[[Ty, Ty], Diagram]:
    """
    Take a ``factory`` for braids of atomic types and extend it recursively.

    Parameters:
        cls : A diagram factory, e.g. :class:`Diagram`.
        factory : A factory for braids of atomic types, e.g. :class:`Braid`.
    """
    def method(left: Ty, right: Ty) -> Diagram:
        if len(left) == 0:
            return cls.id(right)
        if len(right) == 0:
            return cls.id(left)
        if len(left) == len(right) == 1:
            return factory(left[0], right[0])
        if len(left) == 1:
            return method(left, right[:1]) @ right[1:]\
                >> right[:1] @ method(left, right[1:])
        return left[:1] @ method(left[1:], right)\
            >> method(left[:1], right) @ left[1:]

    return method


class Category(monoidal.Category):
    """
    A braided category is a monoidal category with a method :code:`braid`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(monoidal.Functor):
    """
    A braided functor is a monoidal functor that preserves braids.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Braid) and not other.is_dagger:
            return self.cod.ar.braid(self(other.dom[0]), self(other.dom[1]))
        return super().__call__(other)


Diagram.braid_factory = Braid
Id = Diagram.id
