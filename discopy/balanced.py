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
>>> Diagram.twist(x @ y).draw(path="docs/_static/balanced/twist.svg")

.. image:: /_static/balanced/twist.svg
"""

from __future__ import annotations

from copy import copy

from discopy import config, monoidal, braided, traced, hypergraph
from discopy.abc import BalancedCategory
from discopy.cat import factory
from discopy.config import RIBBON_COLORS
from discopy.monoidal import Ty  # noqa: F401
from discopy.utils import factory_name, assert_isatomic


class Ribbon:
    """
    The colour region of a ribbon in the dual rail encoding of a balanced or
    ribbon diagram, see :meth:`double_rail`. It is shared by the two rails of a
    ribbon and carries the colour filling the inside of the ribbon. Being a
    property of the region rather than of the rails, it is preserved when the
    adjoint of a compound type reverses the order of the two rails.

    Parameters:
        color : The colour filling the inside of the ribbon, or ``None``.
    """
    def __init__(self, color=None):
        self.color = color

    def __repr__(self):
        return f"Ribbon(color={self.color!r})"


def set_rail_margins(typ: monoidal.Ty, width: float = None) -> monoidal.Ty:
    """
    Sets the :attr:`min_right_margin` of each object of an already-doubled type
    by position, so that the two rails of every ribbon are drawn ``width``
    apart. This is re-applied after rotation, which reverses the rails and
    drops the margin (a per-object attribute that cannot know its pair-mate).

    Parameters:
        typ : An already-doubled type, i.e. with an even number of objects.
        width : The gap between the two rails, defaults to the ``ribbon_width``
            in :data:`discopy.config.DRAWING_DEFAULT`.
    """
    width = config.DRAWING_DEFAULT["ribbon_width"] if width is None else width
    for i, ob in enumerate(typ.inside):
        ob.min_right_margin = width - 1 if i % 2 == 0 else 0
    return typ


def double_rail(
        typ: monoidal.Ty, width: float = None, color=None) -> monoidal.Ty:
    """
    Doubles every object of a type into the two rails of a ribbon ``width``
    apart, copying each object so the two rails hold independent margins. The
    two rails share a :class:`Ribbon` carrying the colour that fills the inside
    of the ribbon.

    Parameters:
        typ : The type to double.
        width : The gap between the two rails, defaults to the ``ribbon_width``
            in :data:`discopy.config.DRAWING_DEFAULT`.
        color : The colour with which to fill the inside of each ribbon. It can
            be a colour name (used for every ribbon) or a function from object
            to colour name (or ``None`` for no fill). Defaults to ``None``.
    """
    rails = []
    for ob in typ.inside:
        left, right = copy(ob), copy(ob)
        left.ribbon = right.ribbon = Ribbon(
            color(ob) if callable(color) else color)
        rails += [left, right]
    return set_rail_margins(type(typ)(*rails), width)


def ribbon_color_map(diagram, color="auto"):
    """
    Resolves the ``color`` argument of :meth:`Diagram.to_braided` into a
    function from object to colour name (or ``None`` for no fill).

    ``color`` can be ``None`` (no fill), a colour name (used for every ribbon),
    a mapping from object name to colour name, a function from object to colour
    name, or ``"auto"`` (the default) which cycles through
    :data:`RIBBON_COLORS` assigning one colour per distinct object. An object
    and its adjoint encode the same wire, hence share the same ribbon colour.
    """
    if color is None or callable(color):
        return color
    if hasattr(color, "get"):  # A mapping from object name to colour name.
        return lambda ob: color.get(ob.name)
    if color != "auto":  # A single colour name used for every ribbon.
        return lambda ob: color
    names = sorted({ob.name for ob in _atoms(diagram)})
    palette = {name: RIBBON_COLORS[i % len(RIBBON_COLORS)]
               for i, name in enumerate(names)}
    return lambda ob: palette.get(ob.name)


def _atoms(diagram):
    # Every object appearing in the domain, codomain or boxes of a diagram.
    obs = list(getattr(diagram.dom, "inside", ()))
    for box in diagram.boxes:
        obs += list(box.dom.inside) + list(box.cod.inside)
    return obs


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

    def to_braided(self, width: float = None, color="auto"):
        """
        Doubles every object and sends the twist to the braid.

        Parameters:
            width : The width of a ribbon, i.e. the gap between the two wires
                encoding each object, defaults to the ``ribbon_width`` in
                :data:`discopy.config.DRAWING_DEFAULT`. Set to ``0`` to return
                the diagram as is, i.e. without doubling it into dual rails.
            color : The colour with which to fill the inside of each ribbon,
                see :func:`ribbon_color_map`. Defaults to ``"auto"``, i.e. one
                colour per distinct object. Use ``None`` for no fill.

        Example
        -------

        >>> x = Ty('x')
        >>> braided_twist = Diagram.twist(x).to_braided()

        >>> Equation(Twist(x), braided_twist, symbol='$\\\\mapsto$').draw(
        ...     wire_labels=False,
        ...     path="docs/_static/balanced/twist_dual_rail.svg")

        .. image:: /_static/balanced/twist_dual_rail.svg
        """
        get_color = ribbon_color_map(self, color)
        width = config.DRAWING_DEFAULT["ribbon_width"]\
            if width is None else width
        return self if not width\
            else self.dual_rail_factory(width, get_color)(self)


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


class DualRail(Functor):
    """
    The functor sending a balanced diagram to its dual rail encoding in
    :class:`discopy.braided.Diagram`, i.e. doubling every object into the two
    rails of a ribbon and sending every :class:`Twist` and :class:`Braid` to
    a single box crossing the two ribbons of a wire as a whole.

    Parameters:
        width : The gap between the two rails of each ribbon, defaults to the
            ``ribbon_width`` in :data:`discopy.config.DRAWING_DEFAULT`.
        color : The colour filling the inside of each ribbon, either ``None``
            or a function from object to colour name, see :func:`double_rail`.

    See also
    --------
    :meth:`Diagram.to_braided`
    """
    cod = braided.Diagram
    dual_rail_twist_factory = DualRailTwist
    dual_rail_braid_factory = DualRailBraid

    def __init__(self, width: float = None, color=None):
        self.width = config.DRAWING_DEFAULT["ribbon_width"]\
            if width is None else width
        self.color = color
        super().__init__(
            ob_map=lambda x: double_rail(x, self.width, self.color),
            ar_map=lambda f: f.name)

    def __call__(self, other):
        if isinstance(other, monoidal.Ty):
            return set_rail_margins(super().__call__(other), self.width)
        if isinstance(other, Twist):
            return self.dual_rail_twist_factory(self(other.dom))
        if isinstance(other, Braid):
            return self.dual_rail_braid_factory(
                self(other.left), self(other.right), other.is_dagger)
        return super().__call__(other)


Diagram.functor_factory = Functor
Diagram.map_factory = traced.CMap
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.braid_factory = Braid
Diagram.twist_factory = Twist
Diagram.trace_factory = Trace
Diagram.sum_factory = Sum
Diagram.dual_rail_factory = DualRail
Id = Diagram.id


class Equation(braided.Equation):
    """ The :class:`braided.Equation` of balanced diagrams. """
