# -*- coding: utf-8 -*-

"""
The free balanced category, i.e. diagrams with braids and a twist.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Braid
    Twist
    Category
    Functor

Axioms
------
The axiom for the twist holds on the nose.

>>> x, y = Ty('x'), Ty('y')
>>> assert Diagram.twist(x @ y) == (Braid(x, y)
...     >> Twist(y) @ Twist(x) >> Braid(y, x))
>>> Diagram.twist(x @ y).draw(path="docs/_static/balanced/twist.png")

.. image:: /_static/balanced/twist.png
"""

from __future__ import annotations

from discopy import monoidal, braided
from discopy.cat import factory
from discopy.monoidal import Ty, assert_isatomic
from discopy.utils import factory_name


@factory
class Diagram(braided.Diagram):
    """
    A balanced diagram is a braided diagram with :class:`Twist`.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    __ambiguous_inheritance__ = True

    @classmethod
    def twist(cls, dom: monoidal.Ty) -> Diagram:
        """
        The twist on an object.

        Parameters:
            dom : The domain of the twist.

        Note
        ----
        This calls :attr:`twist_factory`.
        """
        if len(dom) == 0:
            return cls.id()
        return cls.braid(dom[0], dom[1:])\
            >> cls.twist(dom[1:]) @ cls.twist_factory(dom[0])\
            >> cls.braid(dom[1:], dom[0])

    def to_braided(self):
        """
        Doubles evry object and sends the twist to the braid.

        Example
        -------

        >>> x = Ty('x')
        >>> braided_twist = Diagram.twist(x).to_braided()

        >>> from discopy.drawing import Equation
        >>> Equation(Twist(x), braided_twist, symbol='$\\\\mapsto$').draw(
        ...     draw_type_labels=False,
        ...     path="docs/_static/balanced/twist_dual_rail.png")

        .. image:: /_static/balanced/twist_dual_rail.png
        """
        class DualRail(Functor):
            cod = braided.Category()

            def __call__(self, other):
                if isinstance(other, Twist):
                    braid = braided.Braid(other.dom, other.dom)
                    return braid >> braid
                return super().__call__(other)

        return DualRail(lambda x: x @ x, lambda f: f.name)(self)


class Box(braided.Box, Diagram):
    """
    A braided box is a monoidal box in a braided diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (braided.Box, )


class Braid(braided.Braid, Box):
    """
    Braid in a balanced category.
    """


class Twist(Box):
    """
    The twist on atomic type :code:`dom`.

    Parameters:
        dom : the domain of the twist.
        phase: the phase of the twist in integer multiples of ``2 * pi``.

    Important
    ---------
    :class:`Twist` is only defined for atomic types (i.e. of length 1).
    For complex types, use :meth:`Diagram.twist` instead.
    """
    def __init__(self, dom: monoidal.Ty, is_dagger=False):
        assert_isatomic(dom, monoidal.Ty)
        name = type(self).__name__ + f"({dom})"
        Box.__init__(self, name, dom, dom, is_dagger=is_dagger)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        return factory_name(type(self)) + f"({self.dom!r})"

    def dagger(self):
        return type(self)(self.dom, not self.is_dagger)


class Category(braided.Category):
    """
    A braided category is a monoidal category with a method :code:`braid`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(braided.Functor):
    """
    A balanced functor is a braided functor that twists.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Twist):
            return self.cod.ar.twist(self(other.dom))
        return super().__call__(other)


Diagram.twist_factory, Diagram.braid_factory = Twist, Braid
Id = Diagram.id
