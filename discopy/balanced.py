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
    Sum
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

from copy import copy

from discopy import monoidal, braided, traced, hypergraph
from discopy.abc import BalancedCategory
from discopy.cat import factory
from discopy.monoidal import Ty  # noqa: F401
from discopy.utils import factory_name, assert_isatomic


def set_rail_margins(typ: monoidal.Ty, width=0.25) -> monoidal.Ty:
    """
    Sets the :attr:`min_right_margin` of each object of an already-doubled type
    by position, so that the two rails of every ribbon (i.e. each consecutive
    pair) are drawn ``width`` apart rather than at the usual minimal width.

    Unlike setting the margin once at doubling time, this is robust to
    rotation: ``.l`` and ``.r`` reverse the order of the rails and drop the
    margin (it is a per-object attribute that cannot know about its pair-mate),
    so the margins are re-asserted by position on each doubled type built.
    """
    rails = []
    for i, ob in enumerate(typ.inside):
        ob = copy(ob)
        ob.min_right_margin = width - 1 if i % 2 == 0 else 0
        rails.append(ob)
    return type(typ)(*rails)


def double_rail(typ: monoidal.Ty, width=0.25) -> monoidal.Ty:
    """
    Doubles every object of a type into the two rails of a ribbon, setting the
    :attr:`min_right_margin` of the first rail to ``width - 1`` so that the two
    rails are drawn ``width`` apart rather than at the usual minimal width.
    """
    return set_rail_margins(
        type(typ)(*[ob for ob in typ.inside for _ in range(2)]), width)


@factory
class Diagram(braided.Diagram, traced.Diagram, BalancedCategory):
    """
    A balanced diagram is a braided diagram with :class:`Twist`.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.

    Note
    ----
    By default, our balanced diagrams are traced. Although not every balanced
    category embeds faithfully into a traced one (see the `nLab`_), the free
    balanced category does have the desired cancellation property and it does
    in fact embed faithfully into the free balanced traced category.

    .. _nLab: https://ncatlab.org/nlab/show/traced+monoidal+category)
    """

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

    def to_braided(self, width=0.25):
        """
        Doubles evry object and sends the twist to the braid.

        Parameters:
            width : The width of a ribbon, i.e. the gap between the two wires
                encoding each object, default is ``0.25`` (four times closer
                than the minimal width). If ``None``, the underlying braided
                diagram is returned rather than its drawing.

        Example
        -------

        >>> x = Ty('x')
        >>> braided_twist = Diagram.twist(x).to_braided()

        >>> from discopy.drawing import Equation
        >>> Equation(Twist(x), braided_twist, symbol='$\\\\mapsto$').draw(
        ...     wire_labels=False,
        ...     path="docs/_static/balanced/twist_dual_rail.png")

        .. image:: /_static/balanced/twist_dual_rail.png
        """
        def double(x):
            return x @ x if width is None else double_rail(x, width)

        class DualRail(Functor):
            cod = braided.Diagram

            def __call__(self, other):
                if width is not None and isinstance(other, monoidal.Ty):
                    return set_rail_margins(super().__call__(other), width)
                if isinstance(other, Twist):
                    return DualRailTwist(self(other.dom))
                if isinstance(other, Braid):
                    return DualRailBraid(
                        self(other.left), self(other.right), other.is_dagger)
                return super().__call__(other)

        return DualRail(double, lambda f: f.name)(self)


class Box(braided.Box, traced.Box, Diagram):
    """
    A braided box is a monoidal box in a braided diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Braid(braided.Braid, Box):
    """
    Braid in a balanced category.
    """


class DualRailBraid(braided.Box):
    """
    The crossing of two ribbons in the dual rail encoding of a swap.

    Unlike the braid of the doubled types (which decomposes into four wire
    crossings via the hexagon equation), this box is drawn as the two ribbons
    crossing as a whole. It is only used by :meth:`Diagram.to_braided`.

    Parameters:
        left : The ribbon (doubled type) on the top left and bottom right.
        right : The ribbon on the top right and bottom left.
        is_dagger (bool) : Which ribbon goes over the other.
    """
    def __init__(self, left: monoidal.Ty, right: monoidal.Ty, is_dagger=False):
        self.left, self.right = left, right
        name = type(self).__name__ + f"({left}, {right})"
        braided.Box.__init__(
            self, name, left @ right, right @ left,
            is_dagger=is_dagger, draw_as_dual_rail_braid=True)

    def __repr__(self):
        str_is_dagger = ", is_dagger=True" if self.is_dagger else ""
        return factory_name(type(self))\
            + f"({self.left!r}, {self.right!r}{str_is_dagger})"

    def dagger(self):
        return type(self)(self.right, self.left, not self.is_dagger)


class DualRailTwist(braided.Box):
    """
    The twist of a ribbon in the dual rail encoding, i.e. its two rails
    crossing each other twice. It is only used by :meth:`Diagram.to_braided`.

    Parameters:
        dom : The ribbon (doubled type) being twisted.
        is_dagger (bool) : Which way the rails twist.
    """
    def __init__(self, dom: monoidal.Ty, is_dagger=False):
        name = type(self).__name__ + f"({dom})"
        braided.Box.__init__(
            self, name, dom, dom,
            is_dagger=is_dagger, draw_as_dual_rail_twist=True)

    def __repr__(self):
        str_is_dagger = ", is_dagger=True" if self.is_dagger else ""
        return factory_name(type(self)) + f"({self.dom!r}{str_is_dagger})"

    def dagger(self):
        return type(self)(self.dom, not self.is_dagger)


class Trace(traced.Trace, Box):
    """
    A trace in a balanced category.

    Parameters:
        arg : The diagram to trace.
        left : Whether to trace the wires on the left or right.

    See also
    --------
    :meth:`Diagram.trace`
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
    drawing_name = "Twist"

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


class Sum(braided.Sum, Box):
    """
    A balanced sum is a braided sum and a balanced box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(braided.Functor, traced.Functor):
    """
    A balanced functor is a braided functor that twists.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Twist):
            return self.cod.twist(self(other.dom))
        if isinstance(other, Trace):
            return traced.Functor.__call__(self, other)
        return braided.Functor.__call__(self, other)


Diagram.functor_factory = Functor
Diagram.map_factory = traced.CMap
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.braid_factory = Braid
Diagram.twist_factory = Twist
Diagram.trace_factory = Trace
Diagram.sum_factory = Sum
Id = Diagram.id
