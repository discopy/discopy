# -*- coding: utf-8 -*-

"""
The free ribbon category, i.e. diagrams with braids, cups and caps.

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
A ribbon category is a braided pivotal category, such that
the trace of the braid is unitary.

>>> x = Ty('x')
>>> from discopy.drawing import Equation
>>> twist_l = Braid(x, x).trace(left=True)
>>> twist_r = Braid(x, x).trace(left=False)
>>> eq = Equation(twist_l >> twist_l[::-1], Id(x), twist_r >> twist_r[::-1])
>>> eq.draw(figsize=(6, 4), margins=(.2, .1),
...         path='docs/_static/ribbon/twist-untwist.png')

.. image:: /_static/ribbon/twist-untwist.png
    :align: center

Equivalently, a ribbon category is a balanced pivotal category, such that
the twist is the trace of the braid :cite:`Selinger10`.
This is made explicit by drawing wires as ribbons,
i.e. two parallel wires with the twist drawn as the double braid.

>>> ribbon_twist = Diagram.twist(x).to_ribbons()
>>> eq = Equation(ribbon_twist, twist_l.to_ribbons())
>>> eq.draw(symbol='$\\\\mapsto$', draw_type_labels=False,
...     path="docs/_static/balanced/ribbon_twist.png")

.. image:: /_static/balanced/ribbon_twist.png

A ribbon category is strict whenever the twist is the identity.
Strict ribbon categories have diagrams with knots, i.e. ribbons where the two
parallel wires coincide and the twist is the identity.

>>> eq_strict = Equation(twist_l, Id(x), twist_r)
>>> eq_strict.draw(figsize=(4, 2), margins=(.2, .1),
...                path='docs/_static/ribbon/strict.png')

.. image:: /_static/ribbon/strict.png
    :align: center
"""

from discopy import rigid, pivotal, balanced
from discopy.cat import factory
from discopy.pivotal import Ty, PRO  # noqa: F401


@factory
class Diagram(pivotal.Diagram, balanced.Diagram):
    """
    A ribbon diagram is a pivotal diagram and a balanced diagram.

    Parameters:
        inside(Layer) : The layers of the diagram.
        dom (pivotal.Ty) : The domain of the diagram, i.e. its input.
        cod (pivotal.Ty) : The codomain of the diagram, i.e. its output.
    """
    def trace(self, n=1, left=False):
        """
        The trace of a ribbon diagram.

        Parameters:
            n : The number of wires to trace.
        """
        if not n:
            return self
        if left:
            return self.caps(self.dom[:n].r, self.dom[:n]) @ self.dom[n:]\
                >> self.dom[:n].r @ self\
                >> self.cups(self.cod[:n].r, self.cod[:n]) @ self.cod[n:]
        return self.dom[:-n] @ self.caps(self.dom[-n:], self.dom[-n:].r)\
            >> self @ self.dom[-n:].r\
            >> self.cod[:-n] @ self.cups(self.cod[-n:], self.cod[-n:].r)

    def cup(self, x, y):
        """
        Post-compose a ribbon diagram with a cup between wires ``i`` and ``j``
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

    def to_ribbons(self):
        """
        Doubles evry object and sends the twist to the braid.

        Example
        -------

        >>> x = Ty('x')
        >>> braided_twist = Diagram.twist(x).to_ribbons()

        .. image:: /_static/balanced/twist_dual_rail.png
        """
        class DualRail(Functor):
            def __call__(self, other):
                if isinstance(other, Twist):
                    braid = Braid(other.dom, other.dom)
                    return braid >> braid
                return super().__call__(other)

        return DualRail(lambda x: x @ x, lambda f: f)(self)


class Box(pivotal.Box, balanced.Box, Diagram):
    """
    A ribbon box is a pivotal and balanced box in a ribbon diagram.

    Parameters:
        name (str) : The name of the box.
        dom (pivotal.Ty) : The domain of the box, i.e. its input.
        cod (pivotal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (pivotal.Box, balanced.Box, )


class Cup(pivotal.Cup, Box):
    """
    A ribbon cup is a pivotal cup in a ribbon diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (pivotal.Cup, )


class Cap(pivotal.Cap, Box):
    """
    A ribbon cap is a pivotal cap in a ribbon diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """
    __ambiguous_inheritance__ = (pivotal.Cap, )


class Braid(balanced.Braid, Box):
    """
    A ribbon braid is a balanced braid in a ribbon diagram.

    Parameters:
        left (pivotal.Ty) : The type on the top left and bottom right.
        right (pivotal.Ty) : The type on the top right and bottom left.
        is_dagger (bool) : Braiding over or under.
    """
    __ambiguous_inheritance__ = (balanced.Braid, )

    z = 0

    def rotate(self, left=False):
        del left
        braid = type(self)(*self.cod.r)
        return braid.dagger() if self.is_dagger else braid


class Twist(balanced.Twist, Box):
    """
    Balanced twist in a ribbon category.

    Parameters:
        left (pivotal.Ty) : The type on the top left and bottom right.
        right (pivotal.Ty) : The type on the top right and bottom left.
        is_dagger (bool) : Braiding over or under.
    """

    z = 0

    def rotate(self, left=False):
        del left
        return self


class Sum(rigid.Sum, Box):
    """
    A ribbon sum is a sum of ribbon diagrams.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """
    __ambiguous_inheritance__ = (rigid.Sum, )


class Category(pivotal.Category, balanced.Category):
    """
    A ribbon category is both a pivotal category and a balanced category.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(pivotal.Functor, balanced.Functor):
    """
    A ribbon functor is both a pivotal functor and a balanced functor.

    Parameters:
        ob (Mapping[pivotal.Ty, pivotal.Ty]) :
            Map from atomic :class:`pivotal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Braid):
            return balanced.Functor.__call__(self, other)
        return pivotal.Functor.__call__(self, other)


Diagram.braid_factory = Braid
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.twist_factory = Twist

Id = Diagram.id
