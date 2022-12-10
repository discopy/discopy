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
The axiom for the twist holds on the nose.

>>> x, y = Ty('x'), Ty('y')
>>> assert Diagram.twist(x @ y) == (Braid(x, y)
...     >> Twist(y) @ Twist(x) >> Braid(y, x))
>>> Diagram.twist(x @ y).draw(path="docs/imgs/balanced/twist.png")

.. image:: /imgs/balanced/twist.png
"""

from __future__ import annotations
from collections.abc import Callable

from discopy import cat, monoidal, braided
from discopy.cat import factory
from discopy.monoidal import Ty, assert_isatomic
from discopy.utils import factory_name, from_tree


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
        if len(dom) == 1:
            return cls.twist_factory(dom)
        return cls.braid(dom[0], dom[1:])\
            >> cls.twist(dom[1:]) @ cls.twist(dom[0])\
            >> cls.braid(dom[1:], dom[0])


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
    def __init__(self, dom: monoidal.Ty, phase: int = 0):
        assert_isatomic(dom, monoidal.Ty)
        name = type(self).__name__ + "({})".format(dom)
        Box.__init__(self, name, dom, dom)

    def __repr__(self):
        return factory_name(type(self)) + "({})".format(self.dom)

    def dagger(self):
        return type(self)(self.dom, - self.phase)


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

    Example
    -------

    The dual rail encoding is a balanced Functor into braided diagrams.

    >>> x = Ty('x')
    >>> ob = lambda x: x @ x
    >>> ar = lambda box: Box(box.name, box.dom @ box.dom, box.cod @ box.cod)
    >>> braided.Diagram.twist = lambda dom: Braid(
    ...     dom[0], dom[1]) >> Braid(dom[1], dom[0])
    >>> F = Functor(ob, ar, cod=Category(Ty, braided.Diagram))
    >>> braided_twist = F(Diagram.twist(x))
    >>> del braided.Diagram.twist
    >>> assert braided_twist == Braid(x, x) >> Braid(x, x)
    >>> from discopy import drawing
    >>> drawing.equation(Twist(x), braided_twist, symbol='->',
    ...     draw_type_labels=False,
    ...     path="docs/imgs/balanced/twist_dual_rail.png")

    .. image:: /imgs/balanced/twist_dual_rail.png
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Twist):
            return self.cod.ar.twist(self(other.dom))
        return super().__call__(other)


Diagram.twist_factory, Diagram.braid_factory = Twist, Braid
Id = Diagram.id
