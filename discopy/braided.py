# -*- coding: utf-8 -*-

"""
The free braided category,
i.e. with three-dimensional diagrams where wires can knot.

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
"""

from __future__ import annotations

from discopy import cat, monoidal
from discopy.cat import factory
from discopy.monoidal import Ty
from discopy.utils import BinaryBoxConstructor, assert_isatomic, factory_name


@factory
class Diagram(monoidal.Diagram):
    """
    A braided diagram is a monoidal diagram with braids.

    Parameters:
        inside (tuple[monoidal.Layer, ...]) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
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
        return hexagon(cls.braid_factory)(left, right)

    def simplify(self) -> Diagram:
        """ Remove braids followed by their dagger. """
        for i, ((x, f, _), (y, g, _)) in enumerate(
                zip(self.inside, self.inside[1:])):
            if x == y and isinstance(f, Braid) and f == g[::-1]:
                inside = self.inside[:i] + self.inside[i + 2:]
                return self.factory(inside, self.dom, self.cod).simplify()
        return self


class Box(monoidal.Box, Diagram):
    """
    A braided box is a monoidal box in a braided diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


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
        name = factory_name(type(self)) + (
            "({}, {})[::-1]".format(right, left) if is_dagger
            else "({}, {})".format(left, right))
        dom, cod = left @ right, right @ left
        Box.__init__(
            self, name, dom, cod, is_dagger=is_dagger, draw_as_braid=True)
        BinaryBoxConstructor.__init__(self, left, right)

    def __repr__(self):
        return factory_name(type(self)) + "({}, {}{})".format(
            repr(self.left), repr(self.right),
            ", is_dagger=True" if self.is_dagger else "")

    def dagger(self):
        return type(self)(self.right, self.left, not self.is_dagger)


Diagram.braid_factory = Braid


def hexagon(factory: Callable) -> Callable:
    """
    Take a binary braid :code:`factory` and extend it recursively.

    Parameters:
        factory : A binary braid constructor, e.g. :class:`Braid`.
    """
    def method(x: Ty, y: Ty) -> Diagram:
        if len(x) == 0: return factory.id(y)
        if len(x) == 1:
            if len(y) == 1: return factory(x[0], y[0])
            return method(x, y[:1]) @ y[1:] >> y[:1] @ method(x, y[1:])
        return x[:1] @ method(x[1:], y) >> method(x[:1], y) @ x[1:]

    return method


class Category(monoidal.Category):
    """
    A braided category is just a pair of Python types :code:`ob` and
    :code:`ar` with appropriate methods :code:`dom`, :code:`cod`, :code:`id`,
    :code:`then`, :code:`tensor` and :code:`braid`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(monoidal.Functor):
    """
    A braided functor is a pair of maps :code:`ob` and :code:`ar` and an
    optional braided category :code:`cod`.

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
