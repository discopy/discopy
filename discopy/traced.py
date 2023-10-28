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
    Category
    Functor

Axioms
------

>>> from discopy.drawing import Equation
>>> from discopy.symmetric import Ty, Box, Swap, Id
>>> from discopy import symmetric
>>> symmetric.Diagram.use_hypergraph_equality = True
>>> x = Ty('x')
>>> f, g = Box('f', x @ x, x @ x), Box('g', x, x)

* Vanishing

>>> assert f.trace(n=0) == f == f.trace(n=0, left=True)
>>> assert f.trace(n=2) == f.trace().trace()
>>> assert f.trace(n=2, left=True) == f.trace(left=True).trace(left=True)

* Superposing

>>> assert (x @ f).trace() == x @ f.trace()
>>> assert (f @ x).trace(left=True) == f.trace(left=True) @ x

* Yanking

>>> yanking = Equation(
...     Swap(x, x).trace(left=True), Id(x), Swap(x, x).trace())
>>> yanking.draw(
...     path='docs/_static/traced/yanking.png', draw_type_labels=False)

.. image:: /_static/traced/yanking.png
    :align: center

>>> assert yanking

* Naturality

>>> tightening_left = Equation(
...     (x @ g >> f >> x @ g).trace(left=True),
...     g >> f.trace(left=True) >> g)
>>> tightening_left.draw(
...     path='docs/_static/traced/tightening-left.png', draw_type_labels=False)

.. image:: /_static/traced/tightening-left.png
    :align: center

>>> tightening_right = Equation(
...     (g @ x >> f >> g @ x).trace(),
...     g >> f.trace() >> g)
>>> tightening_right.draw(
...     path='docs/_static/traced/tightening-right.png',
...     draw_type_labels=False)

.. image:: /_static/traced/tightening-right.png
    :align: center

>>> assert tightening_left and tightening_right

* Dinaturality

>>> sliding_left = Equation(
...     (f >> g @ x).trace(left=True),
...     (g @ x >> f).trace(left=True))
>>> sliding_left.draw(
...     path='docs/_static/traced/sliding-left.png', draw_type_labels=False)

.. image:: /_static/traced/sliding-left.png
    :align: center

>>> sliding_right = Equation(
...     (f >> x @ g).trace(),
...     (x @ g >> f).trace())
>>> sliding_right.draw(
...     path='docs/_static/traced/sliding-right.png', draw_type_labels=False)

.. image:: /_static/traced/sliding-right.png
    :align: center

>>> assert sliding_left and sliding_right

>>> symmetric.Diagram.use_hypergraph_equality = False
"""

from discopy import monoidal
from discopy.cat import factory
from discopy.monoidal import Ty
from discopy.utils import factory_name, assert_isinstance, assert_istraceable


@factory
class Diagram(monoidal.Diagram):
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
        >>> from discopy.drawing import Equation as Eq
        >>> x = Ty('x')
        >>> f = Box('f', x @ x, x @ x)
        >>> LHS, RHS = f.trace(left=True), f.trace(left=False)
        >>> Eq(Eq(LHS, f, symbol="$\\\\mapsfrom$"),
        ...     RHS, symbol="$\\\\mapsto$").draw(
        ...         path="docs/_static/traced/trace.png")

        .. image:: /_static/traced/trace.png
        """
        return self if n == 0\
            else self.trace_factory(self, left).trace(n - 1, left)


class Box(monoidal.Box, Diagram):
    """
    A traced box is a monoidal box in a traced diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (monoidal.Box, )


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
        assert_isinstance(arg, self.factory)
        assert_istraceable(arg, n=1, left=left)
        self.left = left
        name = f"Trace({arg}" + ", left=True)" if left else ")"
        dom, cod = (arg.dom[1:], arg.cod[1:]) if left\
            else (arg.dom[:-1], arg.cod[:-1])
        monoidal.Bubble.__init__(self, arg, dom, cod)
        Box.__init__(self, name, dom, cod)

    def __repr__(self):
        return factory_name(type(self)) + f"({self.arg}, left={self.left})"

    def to_drawing(self):
        traced_wire = self.arg.dom[:1] if self.left else self.arg.dom[-1:]
        dom, cod, traced_wire = map(
            self.ty_factory.to_drawing, [self.dom, self.cod, traced_wire])
        cup = Box('cup', traced_wire ** 2, Ty(), draw_as_wires=True)
        cap = Box('cap', Ty(), traced_wire ** 2, draw_as_wires=True)
        cup, cap, arg = map(self.factory.to_drawing, [cup, cap, self.arg])
        result = cap @ dom >> traced_wire @ arg >> cup @ cod\
            if self.left\
            else dom @ cap >> arg @ traced_wire >> cod @ cup
        return result

    def dagger(self):
        return self.arg.dagger().trace(left=self.left)


class Category(monoidal.Category):
    """
    A traced category is a monoidal category with a method :code:`trace`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(monoidal.Functor):
    """
    A traced functor is a monoidal functor that preserves traces.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.

    Example
    -------
    Let's compute the golden ratio by applying a traced functor.

    >>> from math import sqrt
    >>> from discopy import python
    >>> x = Ty('$\\\\mathbb{R}$')
    >>> f = Box('$\\\\lambda x . (x, 1 + 1 / x)$', x, x @ x)
    >>> g = Box('$\\\\frac{1 + \\\\sqrt{5}}{2}$', Ty(), x)
    >>> F = Functor(
    ...     ob={x: int},
    ...     ar={f: lambda x=1: (x, 1 + 1 / x), g: lambda: (1 + sqrt(5)) / 2},
    ...     cod=Category(python.Ty, python.Function))
    >>> assert F(f.trace())() == F(g)()

    >>> from discopy.drawing import Equation
    >>> Equation(f.trace(), g).draw(path="docs/_static/traced/golden.png")

    .. image:: /_static/traced/golden.png
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Trace):
            n = len(self(other.arg.dom)) - len(self(other.dom))
            return self.cod.ar.trace(self(other.arg), n, left=other.left)
        return super().__call__(other)


class Hypergraph(monoidal.Hypergraph):
    category, functor = Category, Functor


Diagram.hypergraph_factory = Hypergraph
Diagram.trace_factory = Trace
Id = Diagram.id
