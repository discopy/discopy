# -*- coding: utf-8 -*-

"""
The free traced category, i.e. diagrams where outputs can feedback into inputs.

Note that these diagrams are planar traced so that e.g. :mod:`pivotal` diagrams
are traced in this sense. See :mod:`symmetric` for the usual notion of trace.

Whenever the diagrams are also :mod:`symmetric`, their equality can be checked
by translation to monogamous :mod:`hypergraph`.

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
...     path='docs/_static/traced/right-trace.svg')

It is left-traced when it comes with an operator of the following shape:

>>> g = Box("g", z @ x, z @ y)
>>> Equation(g, g.trace(left=True), symbol="$\\\\mapsto$").draw(
...     path='docs/_static/traced/left-trace.svg')


These are subjects to the axioms listed below. Note however that at the moment
equality of planar traced diagrams is not implemented, only symmetric traced.

>>> from discopy.symmetric import Ty, Box, Swap, Id, Equation
>>> from discopy import symmetric
>>> x = Ty('x')
>>> f, g = Box('f', x @ x, x @ x), Box('g', x, x)

Vanishing
=========

>>> assert f.trace(n=0) == f == f.trace(n=0, left=True)
>>> assert f.trace(n=2) == f.trace().trace()
>>> assert f.trace(n=2, left=True) == f.trace(left=True).trace(left=True)

Superposing
===========

>>> assert Equation((x @ f).trace(), x @ f.trace())
>>> assert Equation((f @ x).trace(left=True), f.trace(left=True) @ x)

Yanking
=======

>>> yanking = Equation(
...     Swap(x, x).trace(left=True), Id(x), Swap(x, x).trace())
>>> yanking.draw(
...     path='docs/_static/traced/yanking.svg',
...     wire_labels=False, figsize=(4, 1))

.. image:: /_static/traced/yanking.svg
    :align: center

>>> assert yanking

Naturality
==========

>>> tightening_left = Equation(
...     (x @ g >> f >> x @ g).trace(left=True),
...     g >> f.trace(left=True) >> g)
>>> tightening_left.draw(
...     path='docs/_static/traced/tightening-left.svg', wire_labels=False)

.. image:: /_static/traced/tightening-left.svg
    :align: center

>>> tightening_right = Equation(
...     (g @ x >> f >> g @ x).trace(),
...     g >> f.trace() >> g)
>>> tightening_right.draw(
...     path='docs/_static/traced/tightening-right.svg',
...     wire_labels=False)

.. image:: /_static/traced/tightening-right.svg
    :align: center

>>> assert tightening_left and tightening_right

Dinaturality
============

>>> sliding_left = Equation(
...     (f >> g @ x).trace(left=True),
...     (g @ x >> f).trace(left=True))
>>> sliding_left.draw(
...     path='docs/_static/traced/sliding-left.svg', wire_labels=False)

.. image:: /_static/traced/sliding-left.svg
    :align: center

>>> sliding_right = Equation(
...     (f >> x @ g).trace(),
...     (x @ g >> f).trace())
>>> sliding_right.draw(
...     path='docs/_static/traced/sliding-right.svg', wire_labels=False)

.. image:: /_static/traced/sliding-right.svg
    :align: center

>>> assert sliding_left and sliding_right
"""

from discopy import monoidal, hypergraph
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
        ...         path="docs/_static/traced/trace.svg")

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
        name = f"Trace({arg}" + ", left=True)" if left else ")"
        dom, cod = (arg.dom[1:], arg.cod[1:]) if left\
            else (arg.dom[:-1], arg.cod[:-1])
        monoidal.Bubble.__init__(self, arg, dom=dom, cod=cod)
        Box.__init__(self, name, dom, cod)

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
    >>> Equation(f.trace(), g).draw(path="docs/_static/traced/golden.svg")

    .. image:: /_static/traced/golden.svg
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Trace):
            n = len(self(other.arg.dom)) - len(self(other.dom))
            return self.cod.trace(self(other.arg), n, left=other.left)
        return super().__call__(other)


class CMap(monoidal.CMap):
    category = Diagram
    require_causal = False


Diagram.functor_factory = Functor
Diagram.map_factory = CMap
Diagram.trace_factory = Trace
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id
