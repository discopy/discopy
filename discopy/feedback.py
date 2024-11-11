# -*- coding: utf-8 -*-

"""
The free feedback category, i.e. diagrams with delayed feedback loops.

We follow the definition of :cite:t:`DiLavoreEtAl22` with some extra structure
for the head and tail of streams with the :class:`FollowedBy` generator.

The main example of a feedback category is given by :mod:`discopy.stream`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    HeadOb
    TailOb
    Ty
    Layer
    Diagram
    Box
    Swap
    Feedback
    FollowedBy
    Head
    Tail
    Category
    Functor

Axioms
------
A feedback category is a symmetric monoidal category with a monoidal
endofunctor :meth:`Diagram.delay`, shortened to `.d` and a method
:meth:`Diagram.feedback` of the following shape:

>>> from discopy.drawing import Equation

>>> x, y, m = map(Ty, "xym")
>>> f = Box('f', x @ m.delay(), y @ m)
>>> Equation(f, f.feedback(), symbol="$\\\\mapsto$").draw(
...     path="docs/_static/feedback/feedback-operator.png")

.. image:: /_static/feedback/feedback-operator.png
    :align: center

such that the following equations are satisfied:

Vanishing
=========

>>> assert Box('f', x, y).feedback(mem=Ty()) == Box('f', x, y)

Joining
=======

>>> f = Box('f', x @ (m @ m).delay(), y @ m @ m)
>>> assert f.feedback(mem=m @ m) == f.feedback().feedback()

Strength
========
This can only be checked up to a functor into streams.

>>> from discopy import stream
>>> F0 = Functor(lambda x: stream.Ty.sequence(x.name), cod=stream.Category())
>>> F = Functor(
...     F0, lambda f: stream.Stream.sequence(f.name, F0(f.dom), F0(f.cod)),
...     cod=stream.Category())
>>> all_eq = lambda xs: len(set(xs)) == 1
>>> eq_up_to_F = lambda *fs, n=2: all_eq(F(f).unroll(2).now for f in fs)

>>> f, g = Box('f', x @ m.delay(), y @ m), Box('g', x, y)
>>> strength = Equation(g @ f.feedback(), (g @ f).feedback())
>>> assert eq_up_to_F(*strength.terms)
>>> strength.draw(
...     path='docs/_static/feedback/strength.png', wire_labels=False)

.. image:: /_static/feedback/strength.png
    :align: center

Sliding
=======
This can only be checked up to extensional equivalence of streams.

>>> from discopy import symmetric
>>> n = Ty("n")
>>> h = Box('h', m, n)  # assume h is an isomorphism
>>> f = Box('f', x @ n.d, y @ m)
>>> sliding = Equation((f >> y @ h).feedback(), (x @ h.d >> f).feedback())
>>> sliding.draw(
...     path='docs/_static/feedback/sliding.png', wire_labels=False)

.. image:: /_static/feedback/sliding.png
    :align: center

>>> LHS, RHS = sliding.terms
>>> eq = Equation(*map(lambda f: F(f).unroll(2).now, sliding.terms),
...     symbol="$\\\\sim$").draw(path='docs/_static/feedback/slide-unroll.png')
>>> with symmetric.Diagram.hypergraph_equality:
...     assert F(LHS).unroll(2).now == F(RHS).unroll(2).now\\
...         >> F(y).unroll(2).now @ F(h).later.later.now

.. image:: /_static/feedback/slide-unroll.png
    :align: center

Note
----
Every traced symmetric category is a feedback category with a trivial delay:

>>> from discopy import symmetric
>>> symmetric.Ty.delay = symmetric.Diagram.delay = lambda self: self
>>> symmetric.Diagram.feedback = lambda self, dom=None, cod=None, mem=None:\\
...     self.trace(len(mem))

>>> F0 = Functor(
...     ob=lambda x: symmetric.Ty(x.name), ar={}, cod=symmetric.Category)
>>> assert F0(x.delay()) == F0(x)

>>> F = Functor(
...     ob=F0, ar=lambda f: symmetric.Box(f.name, F0(f.dom), F0(f.cod)),
...     cod=symmetric.Category)
>>> f = Box('f', x @ m.delay(), y @ m)
>>> assert F(f.delay()) == F(f) and F(f.feedback()) == F(f).trace()

Note
----
We also implement endofunctors :class:`Head` and :class:`Tail` together with an
isomorphism :class:`FollowedBy` between `x` and `x.head @ x.tail.delay()`.

This satisfies the following equations:

>>> assert x.head.head == x.head
>>> assert x.head.tail == Ty()
>>> assert x.delay().head == Ty()
>>> assert x.delay().tail == x

In the category of streams, this is just the identity.
"""

from __future__ import annotations

from discopy import cat, monoidal, markov
from discopy.utils import (
    factory, factory_name, assert_isinstance, AxiomError)


def str_delayed(time_step: int):
    return time_step * ".d" if time_step <= 3 else f".delay({time_step})"


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
    def head(self) -> HeadOb | None:
        """ Syntactic sugar for :class:`HeadOb` or `None` if delayed. """
        return None if self.time_step else HeadOb(self)

    @property
    def tail(self) -> Ob | None:
        """ Syntactic sugar for :class:`TailOb` or `self` if `is_constant`. """
        return self.delay(-1) if self.time_step > 0 else (
            self if self.is_constant else TailOb(self))

    def reset(self) -> Ob:
        """ Reset an object to time step zero, used in :class:`Functor`. """
        return Ob(self.name, time_step=0, is_constant=self.is_constant)

    def __eq__(self, other):
        return (
            super().__eq__(other) and self.time_step == other.time_step
            and self.is_constant == other.is_constant)

    def __hash__(self):
        return hash((self.name, self.time_step, self.is_constant))

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        is_constant = "" if self.is_constant else ", is_constant=False"
        return factory_name(
            type(self)) + f"({repr(self.name)}{time_step}{is_constant})"

    def __str__(self):
        return super().__str__() + str_delayed(self.time_step)

    @property
    def d(self):
        """ Syntactic sugar for meth:`delay`. """
        return self.delay()


class HeadOb(Ob):
    """
    The head of a feedback object, interpreted as the first element of a stream
    followed by the constant stream on the empty type.

    Note the object `arg: Ob` cannot be itself a `HeadOb` or be delayed.
    """
    def __init__(self, arg: Ob, time_step: int = 0):
        assert_isinstance(arg, Ob)
        if isinstance(arg, HeadOb) or arg.time_step:
            raise ValueError
        self.arg = arg
        super().__init__(f"{arg}.head", time_step, is_constant=False)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return factory_name(type(self)) + f"({repr(self.arg)}{time_step})"

    def delay(self, n_steps=1):
        return type(self)(self.arg, self.time_step + n_steps)

    def reset(self) -> HeadOb:
        return type(self)(self.arg)

    @property
    def head(self):
        return None if self.time_step else self

    @property
    def tail(self):
        return self.delay(-1) if self.time_step else None


class TailOb(Ob):
    """
    The tail of a non-constant feedback object, interpreted as the stream
    starting from the second time step.

    Example
    -------
    >>> x = Ob('x', is_constant=False)
    >>> assert x.tail == TailOb(x)
    """
    def __init__(self, arg: Ob, time_step: int = 0):
        assert_isinstance(arg, Ob)
        if isinstance(arg, HeadOb) or arg.is_constant or arg.time_step > 0:
            raise ValueError
        self.arg = arg
        super().__init__(f"{arg}.tail", time_step, is_constant=False)

    delay, reset, __repr__ = HeadOb.delay, HeadOb.reset, HeadOb.__repr__


@factory
class Ty(monoidal.Ty):
    """ A feedback type is a monoidal type with `delay`, `head` and `tail`. """
    ob_factory = Ob

    def delay(self, n_steps=1):
        """ The delay of a feedback type by `n_steps`. """
        return type(self)(*(x.delay(n_steps) for x in self.inside))

    @property
    def head(self):
        """ The head of a feedback type, see :class:`HeadOb`. """
        return type(self)(*(x.head for x in self.inside if x.head))

    @property
    def tail(self):
        """ The tail of a feedback type, see :class:`TailOb`. """
        return type(self)(*(x.tail for x in self.inside if x.tail))

    d = Ob.d


class Layer(monoidal.Layer):
    """ A feedback layer is a monoidal layer with a `delay` method. """
    def delay(self, n_steps=1):
        return type(self)(*[x.delay(n_steps) for x in self.boxes_or_types])


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
    >>> walk.draw(path="docs/_static/feedback/feedback-random-walk.png")

    .. image:: /_static/feedback/feedback-random-walk.png
        :align: center
    """
    ty_factory = Ty
    layer_factory = Layer

    def delay(self, n_steps=1):
        """ The delay of a feedback diagram. """
        dom, cod = self.dom.delay(n_steps), self.cod.delay(n_steps)
        inside = tuple(box.delay(n_steps) for box in self.inside)
        return type(self)(inside, dom, cod, _scan=False)

    def feedback(self, dom=None, cod=None, mem=None):
        """ Syntactic sugar for :class:`Feedback`. """
        if mem is None or len(mem) == 1:
            return self.feedback_factory(self, dom=dom, cod=cod, mem=mem)
        return self if not mem else self.feedback(mem=mem[:-1]).feedback()

    @classmethod
    def wait(cls, dom: Ty) -> Diagram:
        """
        Wait one time step, i.e. `Swap(x, x.delay()).feedback()`.

        Example
        -------
        >>> x = Ty('x')
        >>> assert Diagram.wait(x) == Swap(x, x.delay()).feedback()
        >>> Diagram.wait(x).draw(path="docs/_static/feedback/wait.png")

        .. image:: /_static/feedback/wait.png
            :align: center
        """
        return cls.swap(dom, dom.delay()).feedback()

    @property
    def time_step(self) -> int:
        """
        The time step of a diagram is defined only if it is in fact a box.

        This is used for checking equality between boxes and diagrams.

        Example
        -------
        >>> f = Box('f', 'x', 'y')
        >>> assert f.delay(42).time_step == 42
        """
        if len(self) != 1 or self != self.boxes[0]:
            raise ValueError
        return self.boxes[0].time_step

    @property
    def head(self):
        """ Syntactic sugar for :class:`Head`. """
        return Head(self)

    @property
    def tail(self):
        """ Syntactic sugar for :class:`Tail`. """
        return Tail(self)

    d = Ob.d


class Box(markov.Box, Diagram):
    """
    A feedback box is a markov box in a feedback diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
        _time_step (int) : The number of times the box has been delayed.
    """
    __ambiguous_inheritance__ = (markov.Box, )

    _time_step = 0
    time_step = property(lambda self: self._time_step)

    def __init__(self, name, dom, cod, time_step: int = 0, **params):
        self._time_step, self._params = time_step, params
        super().__init__(name, dom, cod, **params)

    def to_drawing(self):
        result = super().to_drawing()
        if result.box.drawing_name:
            result.box.drawing_name += str_delayed(self.time_step)
        return result

    def delay(self, n_steps=1):
        dom, cod = self.dom.delay(n_steps), self.cod.delay(n_steps)
        time_step = self._time_step + n_steps
        return type(self)(self.name, dom, cod, time_step, **self._params)

    def reset(self):
        """ Reset a box to time step zero, used in :class:`Functor`. """
        dom, cod = [x.delay(-self.time_step) for x in (self.dom, self.cod)]
        return type(self)(self.name, dom, cod, **self._params)

    def __str__(self):
        return super().__str__() + str_delayed(self.time_step)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return super().__repr__()[:-1] + time_step + ")"

    def __eq__(self, other):
        return super().__eq__(other) and self.time_step == other.time_step

    def __hash__(self):
        return hash((super().__hash__(), self.time_step))


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


class Head(monoidal.Bubble, Box):
    """
    The head of a feedback diagram, interpreted as the first element followed
    by the identity stream on the empty type.
    """
    def __init__(self, arg: Diagram, time_step=0, _attr="head"):
        dom, cod = (
            getattr(x, _attr).delay(time_step) for x in [arg.dom, arg.cod])
        monoidal.Bubble.__init__(self, arg, dom=dom, cod=cod)
        Box.__init__(self, f"({arg}).{_attr}", self.dom, self.cod, time_step)

    delay, reset, __repr__ = HeadOb.delay, HeadOb.reset, HeadOb.__repr__
    __str__ = Box.__str__


class Tail(monoidal.Bubble, Box):
    """
    The tail of a feedback diagram, interpreted as the stream starting from the
    second time step with the identity on the empty type at the first step.
    """
    def __init__(self, arg: Diagram, time_step=0):
        Head.__init__(self, arg, time_step, _attr="tail")

    delay, reset, __repr__ = HeadOb.delay, HeadOb.reset, HeadOb.__repr__
    __str__ = Box.__str__


class Feedback(monoidal.Bubble, Box):
    """
    Feedback is a bubble that takes a diagram from `dom @ mem.delay()` to
    `cod @ mem` and returns a box from `dom` to `cod`.

    Examples
    --------
    >>> from discopy.drawing import Equation
    >>> x, y, z = map(Ty, "xyz")
    >>> f = Box('f', x @ y.delay(), z @ y)
    >>> fb = f.feedback()
    >>> Equation(f, fb, symbol="$\\\\mapsto$").draw(
    ...     path="docs/_static/feedback/feedback-operator.png")

    .. image:: /_static/feedback/feedback-operator.png
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
        monoidal.Bubble.__init__(self, arg, dom=dom, cod=cod)
        Box.__init__(self, self.name, dom, cod)
        mem_name = "" if len(mem) == 1 else f"mem={mem}"
        self.name = f"({self.arg}).feedback({mem_name})"
        self.use_hypergraph_equality = False

    def delay(self, n_steps=1):
        return type(self)(self.arg.delay(n_steps), mem=self.mem.delay(n_steps))

    def __repr__(self):
        arg, mem = map(repr, (self.arg, self.mem))
        return factory_name(type(self)) + f"({arg}, mem={mem})"

    __str__ = Box.__str__
    _get_structure = markov.Trace._get_structure
    __eq__ = markov.Trace.__eq__


class FollowedBy(Box):
    """
    The isomorphism between `x.head @ x.tail.delay()` and `x`.

    In the category of streams, this is just the identity.

    Example
    -------
    >>> from discopy import stream
    >>> x = Ty(Ob('x', is_constant=False))
    >>> FollowedBy(x).draw(path="docs/_static/feedback/followed-by.png")

    .. image:: /_static/feedback/followed-by.png
        :align: center

    >>> F = Functor({x: stream.Ty.sequence('x')}, cod=stream.Category())
    >>> X, Xh, Xtd = map(F, (x, x.head, x.tail.delay()))
    >>> for xh, xtd in [(Xh.now, Xtd.now),
    ...                 (Xh.later.now, Xtd.later.now),
    ...                 (Xh.later.later.now, Xtd.later.later.now)]:
    ...     print(f"({xh}, {xtd})")
    (x0, Ty())
    (Ty(), x1)
    (Ty(), x2)
    >>> eq_up_to_F = lambda f, g: F(f).unroll(2).now == F(g).unroll(2).now
    >>> assert eq_up_to_F(FollowedBy(x), Id(x))
    """
    def __init__(self, arg: Ty, is_dagger=False, time_step=0):
        self.arg = arg
        dagger_name = ", is_dagger=True" if is_dagger else ""
        name = f"FollowedBy({arg}{dagger_name})"
        dom, cod = arg.head @ arg.tail.delay(), arg
        dom, cod = (cod, dom) if is_dagger else (dom, cod)
        dom, cod = [x.delay(time_step) for x in (dom, cod)]
        super().__init__(name, dom, cod, time_step, is_dagger=is_dagger)

    def __repr__(self):
        is_dagger = ", is_dagger=True" if self.is_dagger else ""
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return f"FollowedBy({repr(self.arg)}{is_dagger}{time_step})"

    def delay(self, n_steps=1):
        arg = self.dom if self.is_dagger else self.cod
        return type(self)(arg, self.is_dagger, self.time_step + n_steps)

    def reset(self):
        return type(self)(self.arg, self.is_dagger)


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
    >>> x, y, m = [Ty(Ob(n, is_constant=False)) for n in "xym"]
    >>> f = Box('f', x @ m.d, y @ m)
    >>> g = Box('g', y.d @ m.d.d, x.d @ m.d)
    >>> F = Functor({x: y.d, y: x.d, m: m.d}, {f: g})

    >>> assert F(f.delay()) == F(f).delay()
    >>> assert F(f.feedback()) == F(f).feedback()
    >>> assert F(x.head) == F(x).head and F(x.tail) == F(x).tail
    >>> assert F(FollowedBy(x)) == FollowedBy(F(x))
    >>> assert F(f.head) == F(f).head and F(f.tail) == F(f).tail
    """
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, (Ob, Box)) and other.time_step:
            cod = self.cod.ob if isinstance(other, Ob) else self.cod.ar
            if hasattr(cod, "delay"):
                result = self(other.reset())
                for _ in range(other.time_step):
                    result = result.delay()
                return result
        if isinstance(other, (HeadOb, TailOb, Head, Tail)):
            cod = self.cod.ar if isinstance(
                other, (Head, Tail)) else self.cod.ob
            attr = "head" if isinstance(other, (HeadOb, Head)) else "tail"
            if hasattr(cod, attr):
                return getattr(self(other.arg), attr)
        if isinstance(
                other, FollowedBy) and hasattr(self.cod.ar, "followed_by"):
            arg = other.dom if other.is_dagger else other.cod
            return self.cod.ar.followed_by(self(arg))
        if isinstance(other, Feedback) and hasattr(self.cod.ar, "feedback"):
            return self(other.arg).feedback(*map(self, (
                other.dom, other.cod, other.mem)))
        return super().__call__(other)


class Hypergraph(markov.Hypergraph):
    category, functor = Category, Functor


Diagram.hypergraph_factory = Hypergraph
Diagram.braid_factory = Swap
Diagram.copy_factory, Diagram.merge_factory = Copy, Merge
Diagram.feedback_factory, Diagram.followed_by = Feedback, FollowedBy
Id = Diagram.id
