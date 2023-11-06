# -*- coding: utf-8 -*-

"""
The free feedback category, i.e.
diagrams with a later endofunctor and a feedback operator.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    Ty
    Layer
    Diagram
    Box
    Swap
    Feedback
    Category
    Functor
"""

from __future__ import annotations

from discopy import cat, monoidal, symmetric, messages
from discopy.utils import factory, assert_isinstance


class Ob(cat.Ob):
    """ A feedback object is an object with a `delay`. """
    def __init__(self, name: str, delay: int = 0):
        assert_isinstance(delay, int)
        if delay < 0: raise NotImplementedError
        self.delay = delay
        super().__init__(name)

    def later(self, n_steps=1):
        return type(self)(self.name, self.delay + n_steps)

    def __repr__(self):
        delay = "" if not self.delay else f", {self.delay}"
        return factory_name(type(self)) + f"({self.name}{delay})"

    def __str__(self):
        result = super().__str__()
        if self.delay == 1:
            result += ".later()"
        elif self.delay > 1:
            result += f".later(n={self.delay})"
        return result

@factory
class Ty(monoidal.Ty):
    """ A feedback type is a monoidal type with a `later` method. """
    ob_factory = Ob

    def later(self, n_steps=1):
        return type(self)(*tuple(x.later(n_steps) for x in self.inside))


class Layer(monoidal.Layer):
    """ A feedback layer is a monoidal layer with a `later` method. """
    def later(self, n_steps=1):
        boxes_or_types = tuple(x.later(n_steps) for x in self.boxes_or_types)
        return type(self)(*boxes_or_types)


@factory
class Diagram(symmetric.Diagram):
    """
    A feedback diagram is a symmetric diagram with a :meth:`later` endofunctor
    and a :meth:`feedback` operator.

    Parameters:
        inside(monoidal.Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ty_factory = Ty
    layer_factory = Layer

    def later(self, n_steps=1):
        dom, cod = self.dom.later(n_steps), self.cod.later(n_steps)
        inside = tuple(box.later(n_steps) for box in self.inside)
        return type(self)(inside, dom, cod)

    def feedback(self, n_wires=1):
        return self.feedback_factory(self, n_wires)


class Box(symmetric.Box, Diagram):
    """
    A feedback box is a symmetric box in a feedback diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (symmetric.Box, )

    def __init__(self, name, dom, cod, delay: int = 0, **params):
        self.delay, self._params = delay, params
        super().__init__(name, dom, cod, **params)

    def later(self, n_steps=1):
        dom, cod = self.dom.later(n_steps), self.cod.later(n_steps)
        delay = self.delay + n_steps
        return type(self)(
            self.name, dom, cod, delay=delay, **self._params)


class Swap(symmetric.Swap, Box):
    """
    The swap of feedback types :code:`left` and :code:`right`.

    Parameters:
        left : The type on the top left and bottom right.
        right : The type on the top right and bottom left.
    """
    def __init__(self, left, right):
        symmetric.Swap.__init__(self, left, right)
        Box.__init__(self, self.name, self.dom, self.cod)

    def later(self, n_steps=1):
        return type(self)(self.left.later(n_steps), self.right.later(n_steps))


class Feedback(monoidal.Bubble, Box):
    """
    Feedback is a bubble that takes a diagram from `x @ y.later()` to `z @ y`
    and returns a box from `x` to `z`.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> Box('f', x @ y.later(), x @ y).feedback().draw()
    """
    to_drawing = symmetric.Trace.to_drawing

    def __init__(self, inside: Diagram,
                 n_wires: int = 1, delay:int = 0, left=False):
        if n_wires <= 0: raise ValueError
        if left: raise NotImplementedError
        future, past = inside.dom[-n_wires:], inside.cod[-n_wires:]
        if future != past.later(): raise AxiomError
        dom = inside.dom[:-n_wires].later(delay)
        cod = inside.cod[:-n_wires].later(delay)
        self.n_wires, self.delay, self.left = n_wires, delay, left
        monoidal.Bubble.__init__(self, inside, dom, cod)
        Box.__init__(self, self.name, dom, cod, delay)

    def later(self, n_steps=1):
        return type(self)(self.inside, self.n_wires, self.delay + n_steps)


class Category(symmetric.Category):
    """
    A feedback category is a symmetric category with methods :code:`later`
    and :code:`feedback`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(symmetric.Functor):
    """
    A feedback functor is a symmetric one that preserves later and feedback.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Ob) and other.delay:
            result = self(type(other)(other.name))
            for _ in range(other.delay):
                result = result.later()
            return result
        if isinstance(other, Feedback):
            n_wires = len(self(other.inside.dom[-other.n_wires:]))
            return self(other.inside).feedback(n_wires)
        return super().__call__(other)


class Hypergraph(symmetric.Hypergraph):
    category, functor = Category, Functor


Diagram.hypergraph_factory = Hypergraph
Diagram.braid_factory = Swap
Diagram.feedback_factory = Feedback
Id = Diagram.id
