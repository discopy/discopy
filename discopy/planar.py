# -*- coding: utf-8 -*-

"""
The free planar traced category, i.e. diagrams where outputs can feedback
into inputs without crossing any wire, so that e.g. :mod:`pivotal` diagrams
are traced in this sense. See :mod:`traced` for the usual notion of trace.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Trace
    Functor

Axioms
------

A monoidal category is right-traced when it comes with an operator of shape:

>>> from discopy.monoidal import Equation
>>> x, y, z = map(Ty, "xyz")
>>> f = Box("f", x @ z, y @ z)
>>> Equation(f, f.trace(), symbol="$\\\\mapsto$").draw(
...     doctest='docs/_static/traced/right-trace.svg')

It is left-traced when it comes with an operator of the following shape:

>>> g = Box("g", z @ x, z @ y)
>>> Equation(g, g.trace(left=True), symbol="$\\\\mapsto$").draw(
...     doctest='docs/_static/traced/left-trace.svg')


These are subject to the axioms stated in :mod:`traced`, where the wires can
also cross; note that equality of planar traced diagrams is not implemented.
"""

from discopy import monoidal, cmap, hypergraph
from discopy.abc import TracedCategory
from discopy.cat import factory
from discopy.monoidal import Ty  # noqa: F401
from discopy.utils import (
    factory_name,
    assert_isinstance,
    assert_istraceable,
)


@factory
class Diagram(monoidal.Diagram, TracedCategory):
    """
    A traced diagram is a monoidal diagram with :class:`Trace` boxes.

    Parameters:
        inside(monoidal.Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """
    def trace(self, n=1, left=False):
        """
        Feed ``n`` outputs back into inputs.

        Parameters:
            n : The number of output wires to feedback into inputs.
            left : Whether to trace the wires on the left or right.

        Example
        -------
        >>> from discopy.monoidal import Equation as Eq
        >>> x = Ty('x')
        >>> f = Box('f', x @ x, x @ x)
        >>> LHS, RHS = f.trace(left=True), f.trace(left=False)
        >>> Eq(Eq(LHS, f, symbol="$\\\\mapsfrom$"),
        ...     RHS, symbol="$\\\\mapsto$").draw(
        ...         doctest="docs/_static/traced/trace.svg")

        .. image:: /_static/traced/trace.svg
        """
        return self if n == 0\
            else self.trace_factory(self, left).trace(n - 1, left)

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)


class Box(monoidal.Box, Diagram):
    """
    A traced box is a monoidal box in a traced diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Trace(Box, monoidal.Bubble):
    """
    A trace is a diagram ``arg`` with an output wire fed back into an input.

    Parameters:
        arg : The diagram to trace.
        left : Whether to trace the wires on the left or right.

    See also
    --------
    :meth:`Diagram.trace`
    """
    def __init__(self, arg: Diagram, left=False):
        assert_isinstance(arg, self.ar)
        assert_istraceable(arg, n=1, left=left)
        self.left = left
        name = f"Trace({arg}, left=True)" if left else f"Trace({arg})"
        dom, cod = (arg.dom[1:], arg.cod[1:]) if left\
            else (arg.dom[:-1], arg.cod[:-1])
        monoidal.Bubble.__init__(self, arg, dom=dom, cod=cod)
        Box.__init__(self, name, dom, cod)

    def __str__(self):
        return self.name

    def __repr__(self):
        return factory_name(type(self)) + f"({self.arg}, left={self.left})"

    def dagger(self):
        return self.arg.dagger().trace(left=self.left)

    def to_drawing(self):
        return self.ar.to_drawing(self)


class Functor(monoidal.Functor):
    """
    A traced functor is a monoidal functor that preserves traces.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Example
    -------
    Let's compute the golden ratio by applying a (hacky) traced functor.

    >>> from math import sqrt
    >>> from discopy import python
    >>> x = Ty('$\\\\mathbb{R}$')
    >>> f = Box('$\\\\lambda x . (x, 1 + 1 / x)$', x, x @ x)
    >>> g = Box('$\\\\frac{1 + \\\\sqrt{5}}{2}$', Ty(), x)
    >>> F = Functor(
    ...     ob_map={x: (float, )},
    ...     ar_map={
    ...         f: lambda x=1.: (x, 1 + 1. / x),
    ...         g: lambda: (1 + sqrt(5)) / 2},
    ...     cod=python.Function)
    >>> with python.Function.no_type_checking:
    ...     assert F(f.trace())() == F(g)()

    >>> from discopy.monoidal import Equation
    >>> Equation(f.trace(), g).draw(doctest="docs/_static/traced/golden.svg")

    .. image:: /_static/traced/golden.svg
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Trace):
            n = len(self(other.arg.dom)) - len(self(other.dom))
            return self.cod.trace(self(other.arg), n, left=other.left)
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Diagram.functor_factory = Functor
Diagram.trace_factory = Trace
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id
