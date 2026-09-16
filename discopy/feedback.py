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

    Wire
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
    Functor

Axioms
------
A feedback category is a symmetric monoidal category with a monoidal
endofunctor, the delay property :attr:`Diagram.d`, and a method
:meth:`Diagram.feedback` of the following shape:


>>> x, y, m = map(Ty, "xym")
>>> f = Box('f', x @ m.d, y @ m)
>>> Equation(f, f.feedback(), symbol="$\\\\mapsto$").draw(
...     doctest="docs/_static/feedback/feedback-operator.svg")

.. image:: /_static/feedback/feedback-operator.svg
    :align: center

such that the following equations are satisfied:

Vanishing
=========

>>> assert Box('f', x, y).feedback(mem=Ty()) == Box('f', x, y)

Joining
=======

>>> f = Box('f', x @ (m @ m).d, y @ m @ m)
>>> assert f.feedback(mem=m @ m) == f.feedback().feedback()

Strength
========
This can only be checked up to a functor into streams.

>>> from discopy import stream
>>> F0 = Functor(
...     lambda x: stream.Ty.sequence(x.generator.name), cod=stream.Stream)
>>> F = Functor(
...     F0, lambda f: stream.Stream.sequence(f.name, F0(f.dom), F0(f.cod)),
...     cod=stream.Stream)
>>> all_eq = lambda xs: len(set(xs)) == 1
>>> eq_up_to_F = lambda *fs, n=2: all_eq(F(f).unroll(2).now for f in fs)

>>> f, g = Box('f', x @ m.d, y @ m), Box('g', x, y)
>>> strength = Equation(g @ f.feedback(), (g @ f).feedback())
>>> assert eq_up_to_F(*strength.terms)
>>> strength.draw(
...     doctest='docs/_static/feedback/strength.svg', wire_labels=False)

.. image:: /_static/feedback/strength.svg
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
...     doctest='docs/_static/feedback/sliding.svg', wire_labels=False)

.. image:: /_static/feedback/sliding.svg
    :align: center

>>> LHS, RHS = sliding.terms
>>> assert F(LHS).unroll(2).now.dom == symmetric.Ty("x0", "x1", "x2")
>>> eq = Equation(*map(lambda f: F(f).unroll(2).now, sliding.terms),
...     symbol="$\\\\sim$").draw(
...         doctest='docs/_static/feedback/slide-unroll.svg')
>>> assert symmetric.Equation(
...     F(LHS).unroll(2).now,
...     F(RHS).unroll(2).now
...         >> F(y).unroll(2).now @ F(h).later.later.now)

.. image:: /_static/feedback/slide-unroll.svg
    :align: center

Note
----
Every traced category is a feedback category with a trivial delay, see
:class:`discopy.abc.TracedCategory` and its free case :mod:`discopy.traced`:

>>> from discopy import traced
>>> F0 = Functor(
...     ob_map=lambda x: traced.Ty(x.generator.name), ar_map={},
...     cod=traced.Diagram)
>>> assert F0(x.d) == F0(x)

>>> F = Functor(
...     ob_map=F0,
...     ar_map=lambda f: traced.Box(f.name, F0(f.dom), F0(f.cod)),
...     cod=traced.Diagram)
>>> f = Box('f', x @ m.d, y @ m)
>>> assert F(f.d) == F(f) and F(f.feedback()) == F(f).trace()

Note
----
We also implement endofunctors :class:`Head` and :class:`Tail` together with an
isomorphism :class:`FollowedBy` between `x` and `x.head @ x.tail.d`.

This satisfies the following equations:

>>> assert x.head.head == x.head
>>> assert x.head.tail == Ty()
>>> assert x.d.head == Ty()
>>> assert x.d.tail == x

In the category of streams, this is just the identity.
"""

from __future__ import annotations

from discopy import monoidal, braided, symmetric, hypergraph
from discopy.abc import DelayedMonoid, FeedbackCategory
from discopy.utils import (
    deprecated_alias,
    factory, factory_name, assert_isinstance, AxiomError,
)


def str_delayed(time_step: int):
    return time_step * ".d"


class Wire(braided.Wire):
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

    @property
    def d(self) -> Wire:
        """ The delay of a feedback object by one time step. """
        return Wire(self.name, self.time_step + 1, self.is_constant)

    @property
    def head(self) -> HeadOb | None:
        """ Syntactic sugar for :class:`HeadOb` or `None` if delayed. """
        return None if self.time_step else HeadOb(self)

    @property
    def tail(self) -> Wire | None:
        """ Syntactic sugar for :class:`TailOb` or `self` if `is_constant`. """
        if self.time_step > 0:
            return Wire(self.name, self.time_step - 1, self.is_constant)
        return self if self.is_constant else TailOb(self)

    def reset(self) -> Wire:
        """ Reset an object to time step zero, used in :class:`Functor`. """
        return Wire(self.name, time_step=0, is_constant=self.is_constant)

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

    def to_tree(self):
        tree = {'factory': factory_name(type(self)), 'name': self.name}
        if self.time_step:
            tree['time_step'] = self.time_step
        if not self.is_constant:
            tree['is_constant'] = False
        return tree

    @classmethod
    def from_tree(cls, tree):
        return cls(
            tree['name'], tree.get('time_step', 0),
            tree.get('is_constant', True))


class HeadOb(Wire):
    """
    The head of a feedback object, interpreted as the first element of a stream
    followed by the constant stream on the empty type.

    Note the object `arg: Wire` cannot be itself a `HeadOb` or be delayed.
    """
    def __init__(self, arg: Wire, time_step: int = 0):
        assert_isinstance(arg, Wire)
        if isinstance(arg, HeadOb) or arg.time_step:
            raise ValueError
        self.arg = arg
        super().__init__(f"{arg}.head", time_step, is_constant=False)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return factory_name(type(self)) + f"({repr(self.arg)}{time_step})"

    @property
    def d(self) -> HeadOb:
        return type(self)(self.arg, self.time_step + 1)

    def reset(self) -> HeadOb:
        return type(self)(self.arg)

    @property
    def head(self):
        return None if self.time_step else self

    @property
    def tail(self):
        if self.time_step:
            return type(self)(self.arg, self.time_step - 1)
        return None


class TailOb(Wire):
    """
    The tail of a non-constant feedback object, interpreted as the stream
    starting from the second time step.

    Example
    -------
    >>> x = Wire('x', is_constant=False)
    >>> assert x.tail == TailOb(x)
    """
    def __init__(self, arg: Wire, time_step: int = 0):
        assert_isinstance(arg, Wire)
        if isinstance(arg, HeadOb) or arg.is_constant or arg.time_step > 0:
            raise ValueError
        self.arg = arg
        super().__init__(f"{arg}.tail", time_step, is_constant=False)

    d, reset, __repr__ = HeadOb.d, HeadOb.reset, HeadOb.__repr__


@factory
class Ty(monoidal.Ty, DelayedMonoid):
    """ A feedback type is a monoidal type with `d`, `head` and `tail`. """
    generator_factory = Wire

    @property
    def d(self) -> Ty:
        """ The delay of a feedback type by one time step. """
        return type(self)(*(x.d for x in self.inside))

    @property
    def head(self):
        """ The head of a feedback type, see :class:`HeadOb`. """
        return type(self)(*(x.head for x in self.inside if x.head))

    @property
    def tail(self):
        """ The tail of a feedback type, see :class:`TailOb`. """
        return type(self)(*(x.tail for x in self.inside if x.tail))


class Layer(symmetric.Layer):
    """ A feedback layer is a symmetric layer with a delay `d`. """
    @property
    def d(self) -> Layer:
        return type(self)(*(x.d for x in self.boxes_or_types),
                          normalise=False)


@factory
class Diagram(symmetric.Diagram, FeedbackCategory):
    """
    A feedback diagram is a symmetric diagram with a delay endofunctor
    :attr:`d` and a :meth:`feedback` operator.

    Parameters:
        inside(monoidal.Layer) : The layers inside the diagram.
        dom (Ty) : The domain of the diagram, i.e. its input.
        cod (Ty) : The codomain of the diagram, i.e. its output.

    Example
    -------
    >>> x, y, m = map(Ty, "xym")
    >>> f = Box('f', x @ m.d, y @ m)
    >>> assert f.feedback().dom == x and f.feedback().cod == y

    See :mod:`discopy.cartesian_feedback` for feedback diagrams with a
    supply of copy, e.g. to output a stream and feed it back at once.
    """
    ob = Ty
    layer_factory = Layer

    @property
    def d(self) -> Diagram:
        """ The delay of a feedback diagram by one time step. """
        inside = tuple(layer.d for layer in self.inside)
        return type(self)(inside, self.dom.d, self.cod.d, _scan=False)

    def feedback(self, dom=None, cod=None, mem=None):
        """ Syntactic sugar for :class:`Feedback`. """
        if mem is None or len(mem) == 1:
            return self.feedback_factory(self, dom=dom, cod=cod, mem=mem)
        return self if not mem else self.feedback(mem=mem[:-1]).feedback()

    @classmethod
    def wait(cls, dom: Ty) -> Diagram:
        """
        Wait one time step, i.e. `Swap(x, x.d).feedback()`.

        Example
        -------
        >>> x = Ty('x')
        >>> assert Diagram.wait(x) == Swap(x, x.d).feedback()
        >>> Diagram.wait(x).draw(doctest="docs/_static/feedback/wait.svg")

        .. image:: /_static/feedback/wait.svg
            :align: center
        """
        return cls.swap(dom, dom.d).feedback()

    @property
    def time_step(self) -> int:
        """
        The time step of a diagram is defined only if it is in fact a box.

        This is used for checking equality between boxes and diagrams.

        Example
        -------
        >>> f = Box('f', 'x', 'y')
        >>> assert f.d.d.time_step == 2
        """
        if len(self) != 1 or self != self.boxes[0]:
            raise ValueError
        return self.boxes[0].time_step

    @property
    def head(self):
        """ Syntactic sugar for :class:`Head`. """
        return self.head_factory(self)

    @property
    def tail(self):
        """ Syntactic sugar for :class:`Tail`. """
        return self.tail_factory(self)


class Box(symmetric.Box, Diagram):
    """
    A feedback box is a symmetric box in a feedback diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
        _time_step (int) : The number of times the box has been delayed.
    """

    _time_step = 0
    time_step = property(lambda self: self._time_step)

    def __init__(self, name, dom, cod, time_step: int = 0, **params):
        self._time_step, self._params = time_step, params
        symmetric.Box.__init__(self, name, dom, cod, **params)
        Diagram.__init__(self, self.inside, self.dom, self.cod)

    def to_drawing(self):
        result = monoidal.Box.to_drawing(self)
        if result.box.drawing_name:
            result.box.drawing_name += str_delayed(self.time_step)
        return result

    @property
    def d(self) -> Box:
        dom, cod, time_step = self.dom.d, self.cod.d, self._time_step + 1
        return type(self)(self.name, dom, cod, time_step, **self._params)

    def reset(self):
        """ Reset a box to time step zero, used in :class:`Functor`. """
        dom, cod = self.dom, self.cod
        for _ in range(self.time_step):
            dom, cod = dom.tail, cod.tail
        return type(self)(self.name, dom, cod, **self._params)

    def __str__(self):
        return super().__str__() + str_delayed(self.time_step)

    def __repr__(self):
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return super().__repr__()[:-1] + time_step + ")"

    def setoid(self):
        return symmetric.Box.setoid(self) + (self.time_step, )


class Permutation(symmetric.Permutation, Box):
    "A permutation in a feedback diagram."

    @property
    def d(self) -> Permutation:
        return type(self)(self.dom.d, self.perm)


class Swap(Permutation, symmetric.Swap, Box):
    """
    The swap of feedback types :code:`left` and :code:`right`.

    Parameters:
        left : The type on the top left and bottom right.
        right : The type on the top right and bottom left.
    """
    def __init__(self, left, right):
        symmetric.Swap.__init__(self, left, right)
        Box.__init__(self, self.name, self.dom, self.cod)

    @property
    def d(self) -> Swap:
        return type(self)(self.left.d, self.right.d)


class Head(monoidal.Bubble, Box):
    """
    The head of a feedback diagram, interpreted as the first element followed
    by the identity stream on the empty type.
    """
    def __init__(self, arg: Diagram, time_step=0, _attr="head"):
        dom, cod = (getattr(x, _attr) for x in [arg.dom, arg.cod])
        for _ in range(time_step):
            dom, cod = dom.d, cod.d
        monoidal.Bubble.__init__(self, arg, dom=dom, cod=cod)
        Box.__init__(self, f"({arg}).{_attr}", self.dom, self.cod, time_step)

    d, reset, __repr__ = HeadOb.d, HeadOb.reset, HeadOb.__repr__
    __str__ = Box.__str__


class Tail(monoidal.Bubble, Box):
    """
    The tail of a feedback diagram, interpreted as the stream starting from the
    second time step with the identity on the empty type at the first step.
    """
    def __init__(self, arg: Diagram, time_step=0):
        Head.__init__(self, arg, time_step, _attr="tail")

    d, reset, __repr__ = HeadOb.d, HeadOb.reset, HeadOb.__repr__
    __str__ = Box.__str__


class Feedback(monoidal.Bubble, Box):
    """
    Feedback is a bubble that takes a diagram from `dom @ mem.d` to
    `cod @ mem` and returns a box from `dom` to `cod`.

    Examples
    --------
    >>> x, y, z = map(Ty, "xyz")
    >>> f = Box('f', x @ y.d, z @ y)
    >>> fb = f.feedback()
    >>> Equation(f, fb, symbol="$\\\\mapsto$").draw(
    ...     doctest="docs/_static/feedback/feedback-bubble.svg")

    .. image:: /_static/feedback/feedback-bubble.svg
        :align: center
    """
    def __init__(self, arg: Diagram, dom=None, cod=None, mem=None, left=False):
        if left:
            raise NotImplementedError
        mem = arg.cod[-1:] if mem is None else mem
        dom = arg.dom[:-len(mem)] if dom is None else dom
        cod = arg.cod[:-len(mem)] if cod is None else cod
        if arg.dom != dom @ mem.d:
            raise AxiomError
        if arg.cod != cod @ mem:
            raise AxiomError
        self.mem, self.left = mem, left
        monoidal.Bubble.__init__(self, arg, dom=dom, cod=cod)
        Box.__init__(self, self.name, dom, cod)

    @property
    def d(self) -> Feedback:
        return type(self)(self.arg.d, mem=self.mem.d)

    def __str__(self):
        mem_name = "" if len(self.mem) == 1 else f"mem={self.mem}"
        return f"({self.arg}).feedback({mem_name})"

    def __repr__(self):
        arg, mem = map(repr, (self.arg, self.mem))
        return factory_name(type(self)) + f"({arg}, mem={mem})"

    def to_drawing(self):
        return self.arg.to_drawing().trace()


class FollowedBy(Box):
    """
    The isomorphism between `x.head @ x.tail.d` and `x`.

    In the category of streams, this is just the identity.

    Example
    -------
    >>> from discopy import stream
    >>> x = Ty(Wire('x', is_constant=False))
    >>> FollowedBy(x).draw(doctest="docs/_static/feedback/followed-by.svg")

    .. image:: /_static/feedback/followed-by.svg
        :align: center

    >>> F = Functor({x: stream.Ty.sequence('x')}, cod=stream.Stream)
    >>> X, Xh, Xtd = map(F, (x, x.head, x.tail.d))
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
        dom, cod = arg.head @ arg.tail.d, arg
        dom, cod = (cod, dom) if is_dagger else (dom, cod)
        for _ in range(time_step):
            dom, cod = dom.d, cod.d
        super().__init__(name, dom, cod, time_step, is_dagger=is_dagger)

    def __repr__(self):
        is_dagger = ", is_dagger=True" if self.is_dagger else ""
        time_step = f", time_step={self.time_step}" if self.time_step else ""
        return f"FollowedBy({repr(self.arg)}{is_dagger}{time_step})"

    @property
    def d(self) -> FollowedBy:
        return type(self)(self.arg, self.is_dagger, self.time_step + 1)

    def reset(self):
        return type(self)(self.arg, self.is_dagger)


class Functor(symmetric.Functor):
    """
    A feedback functor is a symmetric one that preserves delay and feedback.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Example
    -------
    >>> x, y, m = [Ty(Wire(n, is_constant=False)) for n in "xym"]
    >>> f = Box('f', x @ m.d, y @ m)
    >>> g = Box('g', y.d @ m.d.d, x.d @ m.d)
    >>> F = Functor({x: y.d, y: x.d, m: m.d}, {f: g})

    >>> assert F(f.d) == F(f).d
    >>> assert F(f.feedback()) == F(f).feedback()
    >>> assert F(x.head) == F(x).head and F(x.tail) == F(x).tail
    >>> assert F(FollowedBy(x)) == FollowedBy(F(x))
    >>> assert F(f.head) == F(f).head and F(f.tail) == F(f).tail
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, (Wire, Box)) and other.time_step:
            cod = self.cod.ob if isinstance(other, Wire) else self.cod
            if hasattr(cod, "d"):
                result = self(other.reset())
                for _ in range(other.time_step):
                    result = result.d
                return result
        if isinstance(other, (HeadOb, TailOb, Head, Tail)):
            cod = self.cod if isinstance(
                other, (Head, Tail)) else self.cod.ob
            attr = "head" if isinstance(other, (HeadOb, Head)) else "tail"
            if hasattr(cod, attr):
                return getattr(self(other.arg), attr)
        if isinstance(
                other, FollowedBy) and hasattr(self.cod, "followed_by"):
            arg = other.dom if other.is_dagger else other.cod
            return self.cod.followed_by(self(arg))
        if isinstance(other, Feedback) and hasattr(self.cod, "feedback"):
            return self(other.arg).feedback(*map(self, (
                other.dom, other.cod, other.mem)))
        return super().__call__(other)


Diagram.functor_factory = Functor
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.head_factory, Diagram.tail_factory = Head, Tail
Diagram.feedback_factory, Diagram.followed_by = Feedback, FollowedBy
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of feedback diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)


__getattr__ = deprecated_alias(__name__, {"Ob": "Wire"})
