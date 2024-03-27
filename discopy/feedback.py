# -*- coding: utf-8 -*-

"""
The free feedback category, i.e. diagrams with delayed feedback loops.

We follow the definition of :cite:t:`DiLavoreEtAl22` with some extra structure
for the head and tail of streams with the :class:`FollowedBy` generator.

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

Axioms
------

A feedback category is a symmetric monoidal category with a monoidal
endofunctor :meth:`Diagram.delay`, shortened to `.d` and a method
:meth:`Diagram.feedback` of the following shape:

>>> from discopy.drawing import Equation

>>> x, y, z = map(Ty, "xyz")
>>> Box('f', x @ y.delay(), z @ y).feedback().draw(
...     path="docs/_static/feedback/feedback-example.png")

.. image:: /_static/feedback/feedback-example.png
    :align: center

such that the following equations are satisfied:

* Vanishing

>>> assert Box('f', x, y).feedback(mem=Ty()) == Box('f', x, y)

* Joining

>>> f = Box('f', x @ (y @ y).delay(), z @ y @ y)
>>> assert f.feedback(mem=y @ y) == f.feedback().feedback()

* Strength

>>> f, g = Box('f', x @ y.delay(), z @ y), Box('g', x, y)
>>> Equation(g @ f.feedback(), (g @ f).feedback()).draw(
...     path='docs/_static/feedback/strength.png', draw_type_labels=False)

.. image:: /_static/feedback/strength.png
    :align: center

* Sliding

>>> h = Box('h', y, y)
>>> Equation((f >> z @ h).feedback(), (x @ h.d >> f).feedback()).draw(
...     path='docs/_static/feedback/sliding.png', draw_type_labels=False)

.. image:: /_static/traced/sliding.png
    :align: center

"""

from __future__ import annotations

from discopy import cat, monoidal, markov
from discopy.utils import (
    factory, factory_name, assert_isinstance, AxiomError)


class Ob(cat.Ob):
    """
    A feedback object is an object with a `time_step` and an optional argument
    `is_constant` for whether the object is interpreted as a constant stream.
    """
    def __init__(
            self, name: str, time_step: int = 0, is_constant: bool = True):
        assert_isinstance(time_step, int)
        assert_isinstance(is_constant, bool)
        if time_step < 0:
            raise NotImplementedError
        self.time_step, self.is_constant = time_step, is_constant
        super().__init__(name)

    def delay(self, n_steps=1):
        """ The delay of a feedback object. """
        return Ob(self.name, self.time_step + n_steps, self.is_constant)

    @property
    def head(self) -> Head | None:
        """ Syntactic sugar for :class:`Head` or `None` if self is delayed. """
        return None if self.time_step else Head(self)

    @property
    def tail(self) -> Ob | None:
        """ Syntactic sugar for :class:`Tail` or `self` if `is_constant`. """
        return self.delay(-1) if self.time_step > 0 else (
            self if self.is_constant else Tail(self))

    def reset(self) -> Ob:
        return Ob(self.name, time_step=0, is_constant=self.is_constant)

    def __eq__(self, other):
        return (
            super().__eq__(other) and self.time_step == other.time_step
            and self.is_constant == other.is_constant)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        is_constant = "" if self.is_constant else ", is_constant=False"
        return factory_name(
            type(self)) + f"({repr(self.name)}{time_step}{is_constant})"

    def __str__(self, _super=cat.Ob):
        result = _super.__str__(self)
        if self.time_step == 1:
            result += ".d"
        elif self.time_step > 1:
            result += f".delay({self.time_step})"
        return result
    
    d = property(lambda self: self.delay())


class Head(Ob):
    """
    The head of a feedback object, interpreted as the first element of a stream
    followed by the constant stream on the empty type.

    Note the object `arg: Ob` cannot be itself a `Head` or be delayed.
    """
    def __init__(self, arg: Ob, time_step: int = 0):
        assert_isinstance(arg, Ob)
        if isinstance(arg, Head) or arg.time_step:
            return ValueError
        self.arg = arg
        super().__init__(f"{arg}.head", time_step, is_constant=False)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return factory_name(type(self)) + f"({repr(self.arg)}{time_step})"

    def delay(self, n_steps=1):
        return type(self)(self.arg, self.time_step + n_steps)

    def reset(self) -> Head:
        return type(self)(self.arg)

    @property
    def head(self):
        return None if self.time_step else self

    @property
    def tail(self):
        return self.delay(-1) if self.time_step else None


class Tail(Ob):
    """
    The tail of a non-constant feedback object, interpreted as the stream
    starting from the second time step.
    """
    def __init__(self, arg: Ob, time_step: int = 0):
        assert_isinstance(arg, Ob)
        if isinstance(arg, Head) or arg.is_constant or arg.time_step > 0:
            return ValueError
        self.arg = arg
        super().__init__(f"{arg}.tail", time_step, is_constant=False)

    delay, reset, __repr__ = Head.delay, Head.reset, Head.__repr__


@factory
class Ty(monoidal.Ty):
    """ A feedback type is a monoidal type with `delay`, `head` and `tail`. """
    ob_factory = Ob

    def delay(self, n_steps=1):
        """ The delay of a feedback type by `n_steps`. """
        return type(self)(*tuple(x.delay(n_steps) for x in self.inside))

    @property
    def head(self):
        """ The head of a feedback type, see :class:`Head`. """
        return type(self)(*(x.head for x in self.inside if x.head))

    @property
    def tail(self):
        """ The tail of a feedback type, see :class:`Tail`. """
        return type(self)(*(x.tail for x in self.inside if x.tail))


class Layer(monoidal.Layer):
    """ A feedback layer is a monoidal layer with a `delay` method. """
    def delay(self, n_steps=1):
        boxes_or_types = tuple(x.delay(n_steps) for x in self.boxes_or_types)
        return type(self)(*boxes_or_types)


@factory
class Diagram(markov.Diagram):
    """
    A feedback diagram is a markov diagram with a :meth:`delay` endofunctor
    and a :meth:`feedback` operator.

    Parameters:
        inside(monoidal.Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.

    Example
    -------
    >>> x = Ty('x')
    >>> zero = Box('0', Ty(), x.head)
    >>> rand = Box('rand', Ty(), x)
    >>> plus = Box('+', x @ x, x)
    >>> walk = (rand.delay() @ x.delay() >> zero @ plus.delay()
    ...         >> FollowedBy(x) >> Copy(x)).feedback()
    >>> walk.draw(path="docs/_static/feedback/feedback-random-walk.png",
    ...           figsize=(5, 5), margins=(0.25, 0.01))

    .. image:: /_static/feedback/feedback-random-walk.png
        :align: center
    """
    ty_factory = Ty
    layer_factory = Layer

    def delay(self, n_steps=1):
        """ The delay of a feedback diagram. """
        dom, cod = self.dom.delay(n_steps), self.cod.delay(n_steps)
        inside = tuple(box.delay(n_steps) for box in self.inside)
        return type(self)(inside, dom, cod)

    def feedback(self, dom=None, cod=None, mem=None):
        if mem is None or len(mem) == 1:
            return self.feedback_factory(self, dom=dom, cod=cod, mem=mem)
        return self if not mem else self.feedback(mem=mem[:-1]).feedback()

    @classmethod
    def wait(cls, dom: Ty) -> Diagram:
        """ Wait one time step, i.e. `Swap(x, x.delay()).feedback()` """
        return cls.swap(dom, dom.delay()).feedback()

    d = property(lambda self: self.delay())


class Box(markov.Box, Diagram):
    """
    A feedback box is a markov box in a feedback diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """
    __ambiguous_inheritance__ = (markov.Box, )

    time_step = 0

    def __init__(self, name, dom, cod, time_step: int = 0, **params):
        self.time_step, self._params = time_step, params
        super().__init__(name, dom, cod, **params)

    def delay(self, n_steps=1):
        dom, cod = self.dom.delay(n_steps), self.cod.delay(n_steps)
        time_step = self.time_step + n_steps
        return type(self)(
            self.name, dom, cod, time_step=time_step, **self._params)

    def reset(self):
        dom, cod = self.dom.reset(), self.cod.reset()
        return type(self)(self.name, dom, cod, **self._params)

    def __str__(self):
        return Ob.__str__(self, _super=markov.Box)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return super().__repr__()[:-1] + time_step + ")"


class Swap(markov.Swap, Box):
    """
    The swap of feedback types :code:`left` and :code:`right`.

    Parameters:
        left : The type on the top left and bottom right.
        right : The type on the top right and bottom left.
    """
    def __init__(self, left, right):
        markov.Swap.__init__(self, left, right)
        Box.__init__(self, self.name, self.dom, self.cod)

    def delay(self, n_steps=1):
        return type(self)(self.left.delay(n_steps), self.right.delay(n_steps))


class Copy(markov.Copy, Box):
    """
    The copy of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type to copy.
        n : The number of copies.
    """
    def __init__(self, x: Ty, n: int = 2):
        markov.Copy.__init__(self, x, n)
        Box.__init__(self, self.name, self.dom, self.cod)

    def delay(self, n_steps=1):
        return type(self)(self.dom.delay(n_steps), len(self.cod))


class Merge(markov.Merge, Box):
    """
    The merge of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type of wires to merge.
        n : The number of wires to merge.
    """
    def __init__(self, x: Ty, n: int = 2):
        markov.Merge.__init__(self, x, n)
        Box.__init__(self, self.name, self.dom, self.cod)

    def delay(self, n_steps=1):
        return type(self)(self.cod.delay(n_steps), len(self.dom))


class Feedback(monoidal.Bubble, Box):
    """
    Feedback is a bubble that takes a diagram from `dom @ mem.delay()` to
    `cod @ mem` and returns a box from `dom` to `cod`.

    Examples
    --------
    >>> x, y, z = map(Ty, "xyz")
    >>> Box('f', x @ y.delay(), z @ y).feedback().draw(
    ...     path="docs/_static/feedback/feedback-example.png")

    .. image:: /_static/feedback/feedback-example.png
        :align: center
    """
    to_drawing = markov.Trace.to_drawing

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
        mem_name = "" if len(mem) == 1 else f"mem={mem}"
        self.name = f"({self.arg}).feedback({mem_name})"

    def delay(self, n_steps=1):
        return type(self)(self.arg.delay(n_steps), mem=self.mem.delay(n_steps))

    def __repr__(self):
        arg, mem = map(repr, (self.arg, self.mem))
        return factory_name(type(self)) + f"({arg}, mem={mem})"

    __str__ = Box.__str__


class FollowedBy(Box):
    """ The isomorphism between `x.head @ x.tail.delay()` and `x`. """
    def __init__(self, cod: Ty, is_dagger=False, time_step=0):
        dagger_name = ", is_dagger=True" if is_dagger else ""
        name = f"FollowedBy({cod}{dagger_name})"
        dom, cod = cod.head @ cod.tail.delay(), cod
        dom, cod = (cod, dom) if is_dagger else (dom, cod)
        dom, cod = [x.delay(time_step) for x in (dom, cod)]
        super().__init__(name, dom, cod, time_step, is_dagger=is_dagger)

    def __repr__(self):
        arg = (self.dom if self.is_dagger else self.cod).delay(-self.time_step)
        is_dagger = ", is_dagger=True" if self.is_dagger else ""
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return f"FollowedBy({repr(arg)}{is_dagger}{time_step})"

    def delay(self, n_steps=1):
        arg = self.dom if self.is_dagger else self.cod
        return type(self)(arg, self.is_dagger, self.time_step + n_steps)

    def reset(self):
        arg = self.dom if self.is_dagger else self.cod
        return type(self)(arg, self.is_dagger)


class Category(markov.Category):
    """
    A feedback category is a markov category with methods :code:`delay`
    and :code:`feedback`.

    Parameters:
        ob : The objects of the category, default is :class:`Ty`.
        ar : The arrows of the category, default is :class:`Diagram`.
    """
    ob, ar = Ty, Diagram


class Functor(markov.Functor):
    """
    A feedback functor is a markov one that preserves delay and feedback.

    Parameters:
        ob (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod.ar`.
        cod (Category) :
            The codomain, :code:`Category(Ty, Diagram)` by default.

    Example
    -------
    Let's compute the Fibonacci sequence as a stream of Python functions:
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, (Ob, Box)) and other.time_step:
            result = self(other.reset())
            for _ in range(other.time_step):
                result = result.delay()
            return result
        if isinstance(other, Head):
            return self(other.arg).head
        if isinstance(other, Tail):
            return self(other.arg).tail
        if isinstance(other, FollowedBy):
            arg = other.dom if other.is_dagger else other.cod
            return self.cod.ar.followed_by(self(arg))
        if isinstance(other, Feedback):
            dom, cod, mem = map(self, (other.dom, other.cod, other.mem))
            return self(other.arg).feedback(dom=dom, cod=cod, mem=mem)
        return super().__call__(other)


class Hypergraph(markov.Hypergraph):
    category, functor = Category, Functor


Diagram.hypergraph_factory = Hypergraph
Diagram.braid_factory = Swap
Diagram.copy_factory, Diagram.merge_factory = Copy, Merge
Diagram.feedback_factory = Feedback
Id = Diagram.id
