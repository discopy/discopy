# -*- coding: utf-8 -*-

"""
The free traced category, i.e.
diagrams with swaps where outputs can feedback into inputs.

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
"""

from discopy import monoidal, messages
from discopy.cat import factory, AxiomError
from discopy.monoidal import Ty
from discopy.utils import factory_name, assert_isinstance


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
        self.left = left
        assert_isinstance(arg, Diagram)
        name = f"Trace({arg}" + ", left=True)" if left else ")"
        traced_dom, traced_cod = (arg.dom[:1], arg.cod[:1]) if left\
            else (arg.dom[-1:], arg.cod[-1:])
        if traced_dom != traced_cod:
            raise AxiomError(
                messages.NOT_TRACEABLE.format(traced_dom, traced_cod))
        dom, cod = (arg.dom[1:], arg.cod[1:]) if left\
            else (arg.dom[:-1], arg.cod[:-1])
        monoidal.Bubble.__init__(self, arg, dom, cod)
        Box.__init__(self, name, dom, cod)

    def __repr__(self):
        return factory_name(type(self)) + f"({self.arg}, left={self.left})"

    def to_drawing(self):
        traced_wire = self.arg.dom[:1] if self.left else self.arg.dom[-1:]
        cup = Box('cup', traced_wire ** 2, Ty(), draw_as_wires=True)
        cap = Box('cap', Ty(), traced_wire ** 2, draw_as_wires=True)
        result = cap @ self.dom >> traced_wire @ self.arg >> cup @ self.cod\
            if self.left\
            else self.dom @ cap >> self.arg @ traced_wire >> self.cod @ cup
        return result.to_drawing()


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
    A cartesian functor is a monoidal functor that preserves traces.

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


Diagram.trace_factory = Trace

Id = Diagram.id
