# -*- coding: utf-8 -*-

"""
The free feedback category, i.e.
diagrams with a delay endofunctor and a feedback operator.

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

from discopy import cat, monoidal, symmetric
from discopy.utils import factory, factory_name, assert_isinstance, AxiomError


class Ob(cat.Ob):
    """ A feedback object is an object with a `time_step`. """
    def __init__(self, name: str, time_step: int = 0):
        assert_isinstance(time_step, int)
        if time_step < 0:
            raise NotImplementedError
        self.time_step = time_step
        super().__init__(name)

    def delay(self, n_steps=1):
        return type(self)(self.name, self.time_step + n_steps)

    def reset(self) -> Ob:
        return type(self)(self.name)

    def __repr__(self):
        time_step = "" if not self.time_step else f", {self.time_step}"
        return factory_name(type(self)) + f"({self.name}{time_step})"

    def __str__(self):
        result = super(type(self), self).__str__()
        if self.time_step == 1:
            result += ".delay()"
        elif self.time_step > 1:
            result += f".delay({self.time_step})"
        return result


@factory
class Ty(monoidal.Ty):
    """ A feedback type is a monoidal type with a `delay` method. """
    ob_factory = Ob

    def delay(self, n_steps=1):
        return type(self)(*tuple(x.delay(n_steps) for x in self.inside))


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

    def feedback(self, dom=None, cod=None, mem=None):
        return self.feedback_factory(self, dom=dom, cod=cod, mem=mem)


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
        self.time_step, self._params = time_step, params
        super().__init__(name, dom, cod, **params)

    def delay(self, n_steps=1):
        dom, cod = self.dom.delay(n_steps), self.cod.delay(n_steps)
        time_step = self.time_step + n_steps
        return type(self)(
            self.name, dom, cod, time_step=time_step, **self._params)

    def reset(self):
        return type(self)(self.name, self.dom, self.cod, **self._params)

    __str__ = Ob.__str__


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
    Feedback is a bubble that takes a diagram from `dom @ mem.delay()` to
    `cod @ mem` and returns a box from `dom` to `cod`.

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> Box('f', x @ y.delay(), x @ y).feedback().draw(
    ...     path="docs/_static/feedback/feedback-example.png")

    .. image:: /_static/feedback/feedback-example.png
        :align: center
    """
    to_drawing = symmetric.Trace.to_drawing

    def __init__(self, arg: Diagram, dom=None, cod=None, mem=None, left=False):
        if left:
            raise NotImplementedError
        mem = arg.cod[-1:] if mem is None else mem
        dom = arg.dom[:-len(mem)] if dom is None else dom
        cod = arg.cod[:-len(mem)] if cod is None else cod
        if arg.dom != dom @ mem.delay():
            raise AxiomError
        if arg.cod != cod @ mem:
            raise AxiomError
        self.mem, self.left = mem, left
        monoidal.Bubble.__init__(self, arg, dom, cod)
        Box.__init__(self, self.name, dom, cod)

    def delay(self, n_steps=1):
        return type(self)(self.arg.delay(n_steps), mem=self.mem.delay(n_steps))

    def reset(self):
        return type(self)(self.arg, mem=self.mem)


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
        if isinstance(other, (Ob, Box)) and other.time_step:
            result = self(other.reset())
            for _ in range(other.time_step):
                result = result.delay()
            return result
        if isinstance(other, Feedback):
            dom, cod, mem = map(self, (other.dom, other.cod, other.mem))
            return self(other.arg).feedback(dom=dom, cod=cod, mem=mem)
        return super().__call__(other)


class Hypergraph(symmetric.Hypergraph):
    category, functor = Category, Functor


Diagram.hypergraph_factory = Hypergraph
Diagram.braid_factory = Swap
Diagram.feedback_factory = Feedback
Id = Diagram.id
