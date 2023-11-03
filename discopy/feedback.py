# -*- coding: utf-8 -*-

"""
The free feedback category, i.e. diagrams with delay.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Diagram
    Box
    Swap
    Sum
    Category
    Functor
"""

from __future__ import annotations

from discopy import cat, monoidal, symmetric, messages
from discopy.cat import factory


class Ob(cat.Ob):
    """ A feedback object is an object with a `time_step`. """
    def __init__(self, name: str, time_step: int = 0):
        self.time_step = time_step
        super().__init__(name)

    def delay(self, n_steps=1):
        return type(self)(self.name, self.time_step + n_steps)

    @property
    def d(self):
        return self.delay()

    @property
    def f(self):
        return self.delay(n=-1)

    def __repr__(self):
        time_step = "" if not self.time_step else f", {self.time_step}"
        return factory_name(type(self)) + f"({self.name}{time_step})"

    def __str__(self):
        result = super().__str__()
        if self.time_step > 0:
            for _ in range(self.time_step):
                result += ".d"
        elif self.time_step < 0:
            for _ in range(self.time_step):
                result += ".f"
        return result

@factory
class Ty(monoidal.Ty):
    """ A feedback type is a monoidal type with a `delay` method. """
    ob_factory = Ob

    def delay(self, n_steps=1):
        return type(self)(*tuple(x.delay(n_steps) for x in self.inside))

    @property
    def d(self):
        if len(self) != 1: raise TypeError
        return type(self)(self.inside[0].d)

    @property
    def f(self):
        if len(self) != 1: raise TypeError
        return type(self)(self.inside[0].f)


class Layer(monoidal.Layer):
    """ A feedback layer is a monoidal layer with a `delay` method. """
    def delay(self, n_steps=1):
        boxes_or_types = tuple(x.delay(n_steps) for x in self.boxes_or_types)
        return type(self)(*boxes_or_types)


@factory
class Diagram(symmetric.Diagram):
    """
    A feedback diagram is a symmetric diagram with a :meth:`delay` endofunctor
    and a :meth:`feedback` operator.

    Parameters:
        inside(monoidal.Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.
    """
    ty_factory = Ty
    layer_factory = Layer

    def delay(self, n_steps=1):
        dom, cod = self.dom.delay(n_steps), self.cod.delay(n_steps)
        inside = tuple(box.delay(n_steps) for box in self.inside)
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

    def __init__(self, name, dom, cod, time_step: int = 0, **params):
        self.time_step, self.params = time_step, params
        super().__init__(name, dom, cod, **params)

    def delay(self, n_steps=1):
        dom, cod = self.dom.delay(n_steps), self.cod.delay(n_steps)
        time_step = self.time_step + n_steps
        return type(self)(
            self.name, dom, cod, time_step=time_step, **self.params)


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

    def delay(self, n_steps=1):
        return type(self)(self.left.delay(n_steps), self.right.delay(n_steps))


class Feedback(monoidal.Bubble, Box):
    """
    Feedback is a bubble that takes a diagram from `x @ y.delay()` to `z @ y`
    and returns a box from `x` to `z`.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> Box('f', x @ y.delay(), x @ y).feedback().draw()
    """
    to_drawing = symmetric.Trace.to_drawing

    def __init__(self, inside: Diagram,
                 n_wires: int = 1, time_step:int = 0, left=False):
        if n_wires <= 0: raise ValueError
        if left: raise NotImplementedError
        future, past = inside.dom[-n_wires:], inside.cod[-n_wires:]
        if future != past.delay(): raise AxiomError
        dom = inside.dom[:-n_wires].delay(time_step)
        cod = inside.cod[:-n_wires].delay(time_step)
        self.n_wires, self.time_step, self.left = n_wires, time_step, left
        monoidal.Bubble.__init__(self, inside, dom, cod)
        Box.__init__(self, self.name, dom, cod, time_step)

    def delay(self, n_steps=1):
        return type(self)(self.inside, self.n_wires, self.time_step + n_steps)


class Category(symmetric.Category):
    """
    A feedback category is a symmetric category with methods :code:`delay`
    and :code:`feedback`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(symmetric.Functor):
    """
    A feedback functor is a symmetric one that preserves delay and feedback.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
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
