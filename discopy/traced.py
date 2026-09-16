# -*- coding: utf-8 -*-

"""
The free traced category, i.e. feedback diagrams where the delay is trivial.

A traced category is a feedback category where the delay is the identity and
the feedback is given by the trace, see :class:`discopy.abc.TracedCategory`.
The trace operator feeds outputs back into inputs, on the right:

>>> from discopy.monoidal import Equation as Eq
>>> x, y, z = map(Ty, "xyz")
>>> f = Box("f", x @ z, y @ z)
>>> Eq(f, f.trace(), symbol="$\\mapsto$").draw(
...     doctest='docs/_static/traced/right-trace.svg')

or on the left:

>>> g = Box("g", z @ x, z @ y)
>>> Eq(g, g.trace(left=True), symbol="$\\mapsto$").draw(
...     doctest='docs/_static/traced/left-trace.svg')

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Wire
    Ty
    Diagram
    Box
    Trace
    Functor

Axioms
------

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
...     doctest='docs/_static/traced/yanking.svg',
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
...     doctest='docs/_static/traced/tightening-left.svg', wire_labels=False)

.. image:: /_static/traced/tightening-left.svg
    :align: center

>>> tightening_right = Equation(
...     (g @ x >> f >> g @ x).trace(),
...     g >> f.trace() >> g)
>>> tightening_right.draw(
...     doctest='docs/_static/traced/tightening-right.svg',
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
...     doctest='docs/_static/traced/sliding-left.svg', wire_labels=False)

.. image:: /_static/traced/sliding-left.svg
    :align: center

>>> sliding_right = Equation(
...     (f >> x @ g).trace(),
...     (x @ g >> f).trace())
>>> sliding_right.draw(
...     doctest='docs/_static/traced/sliding-right.svg', wire_labels=False)

.. image:: /_static/traced/sliding-right.svg
    :align: center

>>> assert sliding_left and sliding_right

Feedback
========

>>> assert f.delay() == f and x.delay() == x
>>> assert f.feedback() == f.trace()
>>> assert f.feedback(mem=x @ x) == f.trace(n=2)
"""

from __future__ import annotations

from discopy import monoidal, feedback, hypergraph, cmap
from discopy.abc import TracedCategory
from discopy.cat import factory


class Wire(feedback.Wire):
    """ A traced wire is a feedback wire with a trivial delay. """
    def delay(self, n_steps: int = 1) -> Wire:
        return self


@factory
class Ty(feedback.Ty):
    """ A traced type is a feedback type with a trivial delay. """
    generator_factory = Wire

    def delay(self, n_steps: int = 1) -> Ty:
        return self


@factory
class Diagram(feedback.Diagram, TracedCategory):
    """
    A traced diagram is a feedback diagram with :class:`Trace` bubbles,
    where the delay is trivial and the feedback is given by the trace.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty

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
        >>> assert f.trace(2) == Trace(Trace(f))
        >>> LHS, RHS = f.trace(left=True), f.trace(left=False)
        >>> Eq(Eq(LHS, f, symbol="$\\mapsfrom$"),
        ...     RHS, symbol="$\\mapsto$").draw(
        ...         doctest="docs/_static/traced/trace.svg")

        .. image:: /_static/traced/trace.svg
        """
        return TracedCategory.trace(self, n, left)

    def delay(self, n_steps: int = 1) -> Diagram:
        """
        The delay of a traced diagram is trivial, i.e. the identity.

        Parameters:
            n_steps : The number of time steps to delay.
        """
        return self

    feedback = TracedCategory.feedback


class Box(feedback.Box, Diagram):
    """
    A traced box is a feedback box in a traced diagram.

    Parameters:
        name (str) : The name of the box.
        dom (Ty) : The domain of the box, i.e. its input.
        cod (Ty) : The codomain of the box, i.e. its output.
    """
    delay = Diagram.delay


class Trace(monoidal.Trace, Box):
    """
    A trace is a diagram ``arg`` with an output wire fed back into an input.

    Parameters:
        arg : The diagram to trace.
        left : Whether to trace the wires on the left or right.

    See also
    --------
    :meth:`Diagram.trace`
    """


class Permutation(feedback.Permutation, Box):
    "A permutation in a traced diagram."


class Swap(feedback.Swap, Permutation):
    "A swap in a traced diagram."


class Functor(feedback.Functor):
    """
    A traced functor is a feedback functor that also preserves traces.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain, :code:`Diagram` by default.

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
    """
    dom = cod = Diagram


CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]

Diagram.functor_factory = Functor
Diagram.trace_factory = Trace
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Id = Diagram.id


class Equation(feedback.Equation):
    """ The :class:`feedback.Equation` of traced diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)
