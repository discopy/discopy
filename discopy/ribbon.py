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
>>> eq.draw(margins=(.2, 0), path='docs/_static/ribbon/twist-untwist.png')

.. image:: /_static/ribbon/twist-untwist.png
    :align: center

Equivalently, a ribbon category is a balanced pivotal category, such that
the twist is the trace of the braid :cite:`Selinger10`.
This is made explicit by drawing wires as ribbons,
i.e. two parallel wires with the twist drawn as the double braid.

>>> ribbon_twist = Diagram.twist(x).to_ribbons()
>>> eq = Equation(ribbon_twist, twist_l.to_ribbons())
>>> eq.draw(symbol='$\\\\mapsto$', wire_labels=False,
...     path="docs/_static/balanced/ribbon_twist.png")

.. image:: /_static/balanced/ribbon_twist.png

A ribbon category is strict whenever the twist is the identity.
Strict ribbon categories have diagrams with knots, i.e. ribbons where the two
parallel wires coincide and the twist is the identity.

>>> eq_strict = Equation(twist_l, Id(x), twist_r)
>>> eq_strict.draw(margins=(.2, .1), path='docs/_static/ribbon/strict.png')

.. image:: /_static/ribbon/strict.png
    :align: center
"""

from discopy import rigid, pivotal, balanced
from discopy.abc import RibbonCategory
from discopy.cat import factory
from discopy.pivotal import Ty, PRO  # noqa: F401


@factory
class Diagram(pivotal.Diagram, balanced.Diagram, RibbonCategory):
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

    def to_ribbons(self, width: float = None):
        """
        Doubles every object and sends the twist to the braid, folding cups
        and caps into a single box.

        Parameters:
            width : The width of a ribbon, i.e. the gap between the two wires
                encoding each object, defaults to the ``ribbon_width`` in
                :data:`discopy.config.DRAWING_DEFAULT`. Set to ``0`` to return
                the diagram as is, i.e. without doubling it into dual rails.

        Example
        -------

        >>> x = Ty('x')
        >>> braided_twist = Diagram.twist(x).to_ribbons()

        .. image:: /_static/balanced/twist_dual_rail.png
        """
        return self.to_braided(width)


class Box(pivotal.Box, balanced.Box, Diagram):
    """
    A ribbon box is a pivotal and balanced box in a ribbon diagram.

    Parameters:
        name (str) : The name of the box.
        dom (pivotal.Ty) : The domain of the box, i.e. its input.
        cod (pivotal.Ty) : The codomain of the box, i.e. its output.
    """


class Cup(pivotal.Cup, Box):
    """
    A ribbon cup is a pivotal cup in a ribbon diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """


class Cap(pivotal.Cap, Box):
    """
    A ribbon cap is a pivotal cap in a ribbon diagram.

    Parameters:
        left (pivotal.Ty) : The atomic type.
        right (pivotal.Ty) : Its adjoint.
    """


class Braid(balanced.Braid, Box):
    """
    A ribbon braid is a balanced braid in a ribbon diagram.

    Parameters:
        left (pivotal.Ty) : The type on the top left and bottom right.
        right (pivotal.Ty) : The type on the top right and bottom left.
        is_dagger (bool) : Braiding over or under.
    """

    z = 0

    def rotate(self, left=False):
        del left
        braid = type(self)(*self.cod.r)
        return braid.dagger() if self.is_dagger else braid


class DualRailBraid(balanced.DualRailBraid, Box):
    """
    The crossing of two ribbons in the dual rail encoding of a ribbon swap.

    See also
    --------
    :class:`discopy.balanced.DualRailBraid`
    """

    z = 0

    def rotate(self, left=False):
        del left
        return type(self)(self.right, self.left, self.is_dagger)


class DualRailTwist(balanced.DualRailTwist, Box):
    """
    The twist of a ribbon in the dual rail encoding of a ribbon twist.

    See also
    --------
    :class:`discopy.balanced.DualRailTwist`
    """

    z = 0

    def rotate(self, left=False):
        del left
        return self


class DualRailCup(Box):
    """
    A cup joining two ribbons in the dual rail encoding, drawn as a single
    constant-width fold. It is only used by :meth:`Diagram.to_ribbons`.

    Parameters:
        left : The ribbon (doubled type) on the outside left.
        right : The ribbon on the outside right.
    """
    z = 0

    def __init__(self, left, right, is_dagger=False):
        self.left, self.right = left, right
        name = type(self).__name__ + f"({left}, {right})"
        Box.__init__(
            self, name, left @ right, type(left)(),
            is_dagger=is_dagger, draw_as_dual_rail_cup=True)

    def rotate(self, left=False):
        del left
        return DualRailCap(self.left.r, self.right.r)

    def dagger(self):
        return DualRailCap(self.left, self.right, not self.is_dagger)


class DualRailCap(Box):
    """
    A cap joining two ribbons in the dual rail encoding, see
    :class:`DualRailCup`.
    """
    z = 0

    def __init__(self, left, right, is_dagger=False):
        self.left, self.right = left, right
        name = type(self).__name__ + f"({left}, {right})"
        Box.__init__(
            self, name, type(left)(), left @ right,
            is_dagger=is_dagger, draw_as_dual_rail_cup=True)

    def rotate(self, left=False):
        del left
        return DualRailCup(self.left.r, self.right.r)

    def dagger(self):
        return DualRailCup(self.left, self.right, not self.is_dagger)


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


class Functor(pivotal.Functor, balanced.Functor):
    """
    A ribbon functor is both a pivotal functor and a balanced functor.

    Parameters:
        ob_map (Mapping[pivotal.Ty, pivotal.Ty]) :
            Map from atomic :class:`pivotal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Braid):
            return balanced.Functor.__call__(self, other)
        return pivotal.Functor.__call__(self, other)


class DualRail(balanced.DualRail, Functor):
    """
    The functor sending a ribbon diagram to its dual rail encoding, extending
    :class:`discopy.balanced.DualRail` with the folding of :class:`Cup` and
    :class:`Cap` into a single box each.

    See also
    --------
    :meth:`Diagram.to_ribbons`
    """
    cod = Diagram
    dual_rail_twist_factory = DualRailTwist
    dual_rail_braid_factory = DualRailBraid

    def __call__(self, other):
        if isinstance(other, Cup):
            return DualRailCup(self(other.dom[:1]), self(other.dom[1:]))
        if isinstance(other, Cap):
            return DualRailCap(self(other.cod[:1]), self(other.cod[1:]))
        return super().__call__(other)


Diagram.braid_factory = Braid
Diagram.cup_factory, Diagram.cap_factory = Cup, Cap
Diagram.twist_factory = Twist
Diagram.dual_rail_factory = DualRail

Id = Diagram.id
