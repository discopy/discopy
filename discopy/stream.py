"""
The feedback category of monoidal streams over a symmetric monoidal category.

We adapted the definition of intensional streams from :cite:t:`DiLavoreEtAl22`.

## Summary

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Stream
    Category

## Note

Monoidal streams form a feedback category as follows:

>>> from discopy import feedback, drawing
>>> x, y, m = map(feedback.Ty, "xym")
>>> f = feedback.Box('f', x @ m.delay(), y @ m)
>>> fb = f.feedback()

>>> X, Y, M = [Ty.sequence(symmetric.Ty(n)) for n in "xym"]
>>> Ff = Stream.sequence("f", X @ M.delay(), Y @ M)

>>> F = feedback.Functor(ob={x: X, y: Y, m: M}, ar={f: Ff},
...                      cod=feedback.Category(Ty, Stream))

>>> drawing.Equation(fb, F(fb).unroll(2).now, symbol="$\\\\mapsto$").draw(
...     path="docs/_static/stream/feedback-to-stream.png")

.. image:: /_static/stream/feedback-to-stream.png
    :align: center

## Examples

### Fibonacci

We can define the Fibonacci sequence as a feedback diagram interpreted in the
category of streams of python types and functions.

>>> from discopy import *
>>> from discopy.feedback import *

>>> X = Ty('X')
>>> fby, wait = FollowedBy(X), Swap(X, X.d).feedback()
>>> zero, one = Box('zero', Ty(), X.head), Box('one', Ty(), X.head)
>>> copy, plus = Copy(X), Box('plus', X @ X, X)

>>> @Diagram.feedback
... @Diagram.from_callable(X.d, X @ X)
... def fib(x):
...     y = fby(zero(), plus.d(fby.d(one.d(), wait.d(x)), x))
...     return (y, y)

>>> fib_ = (copy.d >> one.d @ wait.d @ X.d
...                >> fby.d @ X.d
...                >> plus.d
...                >> zero @ X.d
...                >> fby >> copy).feedback()
>>> with Diagram.hypergraph_equality:
...     assert fib == fib_
>>> fib_.draw(wire_labels=False, figsize=(5, 5),
...           path="docs/_static/stream/fibonacci-feedback.png")

.. image:: /_static/stream/fibonacci-feedback.png
    :align: center

>>> cod = stream.Category(python.Ty, python.Function)
>>> F = feedback.Functor(
...     ob={X: int},
...     ar={zero: cod.ar.singleton(python.Function(lambda: 0, (), int)),
...         one: cod.ar.singleton(python.Function(lambda: 1, (), int)),
...         plus: lambda x, y: x + y}, cod=cod)
>>> assert F(fib).unroll(9).now()[:10] == (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)

### Random walk

We can define a simple random walk as a feedback diagram interpreted in the
category of streams of python types and probabilistic functions.

>>> from random import choice, seed; seed(420)
>>> rand = Box('rand', Ty(), X)
>>> F.ar[rand] = lambda: choice([-1, +1])

>>> @Diagram.feedback
... @Diagram.from_callable(X.d, X @ X)
... def walk(x):
...     x = plus.d(rand.d(), x)
...     x = fby(zero(), x)
...     return (x, x)

>>> walk.draw(wire_labels=False, figsize=(5, 5),
...           path="docs/_static/stream/random-walk-feedback.png")

.. image:: /_static/stream/random-walk-feedback.png
    :align: center

>>> assert F(walk).unroll(9).now()[:10] == (0, -1, 0, 1, 2, 1, 0, -1, 0, 1)
>>> assert F(walk).unroll(9).now()[:10] == (0, -1, -2, -1, 0, 1, 0, 1, 2, 1)
>>> assert F(walk).unroll(9).now()[:10] == (0, -1, 0, 1, 0, 1, 0, -1, 0, -1)

## Axioms

Note that we can only check equality of streams up to a finite number of steps.

>>> from discopy.stream import *
>>> all_eq = lambda xs: len(set(xs)) == 1
>>> eq_up_to_n = lambda *xs, n=3: all_eq(x.unroll(n).now for x in xs)

>>> x, y, z, w, m, n, o = map(Ty.sequence, "xyzwmno")
>>> f = Stream.sequence('f', x, y, m)
>>> g = Stream.sequence('g', y, z, n)
>>> h = Stream.sequence('h', z, w, o)

* Unitality and associativity hold on the nose:

>>> _id = Stream.id
>>> assert eq_up_to_n(f @ _id(), f, _id() @ f)
>>> assert eq_up_to_n(f >> _id(f.cod), f, _id(f.dom) >> f)
>>> assert eq_up_to_n((f >> g) >> h), (f >> (g >> h))
>>> ((f >> g) >> h).now.draw(
...     path="docs/_static/stream/feedback-associativity.png")

.. image:: /_static/stream/feedback-associativity.png
    :align: center

* Associativity of tensor holds up to interchanger:

>>> from discopy.drawing import Equation
>>> drawing.Equation(*map(lambda x: x.now, ((f @ g) @ h, f @ (g @ h)))).draw(
...     path="docs/_static/stream/feedback-tensor-associativity.png")

.. image:: /_static/stream/feedback-tensor-associativity.png
    :align: center

>>> eq_up_to_interchanger = lambda *xs: all_eq(
...     monoidal.Diagram.normal_form(x.now) for x in xs)
>>> assert eq_up_to_interchanger((f @ g) @ h, f @ (g @ h))

* Interchanger holds up to permutation of the memories:

>>> x_, y_, z_, m_, n_ = [
...     Ty.sequence(symmetric.Ty(name + "'")) for name in "xyzmn"]
>>> f_ = Stream.sequence("f'", x_, y_, m_)
>>> g_ = Stream.sequence("g'", y_, z_, n_)

>>> LHS, RHS = f @ f_ >> g @ g_, (f >> g) @ (f_ >> g_)
>>> drawing.Equation(LHS.now, RHS.now, symbol="$\\\\sim$").draw(
...     path="docs/_static/stream/feedback-interchanger.png", figsize=(8, 6))

.. image:: /_static/stream/feedback-interchanger.png
    :align: center

>>> pi, id_dom = (0, 1, 2, 4, 3, 5), symmetric.Id(LHS.now.dom)
>>> with symmetric.Diagram.hypergraph_equality:
...     assert LHS.now == id_dom.permute(*pi) >> RHS.now.permute(*pi)

See :mod:`discopy.feedback` for the other axioms for feedback categories.
"""
from __future__ import annotations

from typing import Callable, Optional
from dataclasses import dataclass

from discopy import symmetric
from discopy.utils import (
    AxiomError, Composable, Whiskerable, NamedGeneric, get_origin, is_tuple,
    assert_isinstance, unbiased, inductive, classproperty, factory_name)


@dataclass
class Ty(NamedGeneric['base']):
    """
    A stream of types from some underlying class `base`.

    Parameters:
        now (base) : The value of the stream at time step zero.
        _later (Optional[Callable[[], Ty[base]]]) :
            A thunk for the tail of the stream, constant by default.
    """
    base = symmetric.Ty  # The underlying class of types.

    now: base = None
    _later: Callable[[], Ty[base]] = None

    factory = classproperty(lambda cls: cls)

    def __init__(
            self, now: base = None, _later: Callable[[], Ty[base]] = None):
        if is_tuple(self.base) and not isinstance(now, (tuple, type(None))):
            now = (now, )
        now = now if isinstance(now, get_origin(self.base)) else (
            self.base() if now is None else self.base(now))
        self.now, self._later = now, _later

    def __repr__(self):
        _later = "" if self.is_constant else f", _later={repr(self._later)}"
        return factory_name(type(self)) + f"({repr(self.now)}{_later})"

    @property
    def later(self) -> Ty:
        """ The tail of a stream, or `self` if :meth:`is_constant`. """
        return self if self.is_constant else self._later()

    @property
    def head(self) -> Ty:
        """ The :meth:`singleton` over the first time step. """
        return self.singleton(self.now)

    tail = later

    @property
    def is_constant(self) -> bool:
        """ Whether a stream of type is constant. """
        return self._later is None

    @classmethod
    def singleton(cls, x: base) -> Ty:
        """
        Constructs the stream with `x` now and the empty stream later.

        >>> XY = Ty.singleton(symmetric.Ty('x', 'y'))
        >>> for x in [XY.now, XY.later.now, XY.later.later.now]: print(x)
        x @ y
        Ty()
        Ty()
        """
        return cls(now=x, _later=lambda: cls())

    @inductive
    def delay(self) -> Ty:
        """
        Delays a stream of types by pre-pending with the unit.

        >>> XY = Ty(symmetric.Ty('x', 'y')).delay()
        >>> for x in [XY.now, XY.later.now, XY.later.later.now]: print(x)
        Ty()
        x @ y
        x @ y
        """
        return type(self)(self.base(), lambda: self)

    d = property(lambda self: self.delay())

    @classmethod
    def sequence(cls, x: base, n_steps: int = 0) -> Ty:
        """
        Constructs the stream `x0`, `x1`, etc.

        >>> XY = Ty.sequence(symmetric.Ty('x', 'y'))
        >>> for x in [XY.now, XY.later.now, XY.later.later.now]: print(x)
        x0 @ y0
        x1 @ y1
        x2 @ y2
        """
        now = sum([cls.base(f"{obj}{n_steps}") for obj in x], cls.base())
        return cls(now, _later=lambda: cls.sequence(x, n_steps + 1))

    @inductive
    def unroll(self) -> Ty:
        """
        Unroll a stream `x0, x1, x2, x3, ...` to `x0 @ x1, x2, x3, ...`.

        >>> U = Ty.sequence('x').unroll()
        >>> for x in [U.now, U.later.now, U.later.later.now]: print(x)
        x0 @ x1
        x2
        x3
        """
        return type(self)(self.now + self.later.now, lambda: self.later.later)

    @unbiased
    def tensor(self, other: Ty) -> Ty:
        """
        The tensor of streams of types is computed pointwise.

        >>> X, Y = map(Ty.sequence, "xy")
        >>> XY = X @ Y
        >>> for x in [XY.now, XY.later.now, XY.later.later.now]: print(x)
        x0 @ y0
        x1 @ y1
        x2 @ y2
        """
        if not isinstance(other, Ty):
            return NotImplemented
        _later = None if self.is_constant and other.is_constant else (
            lambda: self.later.tensor(other.later))
        return type(self)(self.now + other.now, _later)

    __add__ = __matmul__ = symmetric.Ty.__matmul__
    __pow__ = symmetric.Ty.__pow__


@dataclass
class Stream(Composable, Whiskerable, NamedGeneric['category']):
    """
    Monoidal streams over an underlying `category`.

    Parameters:
        now (category.ar) : The value of the stream at time step zero.
        dom (Optional[Ty[category.ob]]) :
            The domain of the stream, constant `now.dom` if `_later is None`.
        cod (Optional[Ty[category.ob]]) :
            The codomain of the stream, constant `now.dom` if `_later is None`.
        mem (Optional[Ty[category.ob]]) :
            The memory of the stream, the constant empty type by default.
        _later (Optional[Callable[[], Stream[category]]]) :
            A thunk for the tail of the stream, constant by default.

    Example
    -------
    >>> from discopy import python
    >>> T, S = Ty[python.Ty], Stream[python.Category]
    >>> x, y, m = int, bool, str
    >>> now = python.Function(lambda n: (bool(n % 2), str(n)), x, (y, m))
    >>> dom, cod, mem = T(x), T(y), T(m).delay()
    >>> later = S(lambda n, s: (bool(n % 2), f"{s} {n}"), dom, cod, mem.later)
    >>> f = S(now, dom, cod, mem, lambda: later)
    >>> f.unroll(2).now(1, 2, 3)
    (True, False, True, '1 2 3')

    Note
    ----
    The parameters should satisfy the following conditions:

    >>> assert now.dom == dom.now + mem.now
    >>> assert now.cod == cod.now + mem.later.now

    >>> assert dom.later.now == later.dom.now
    >>> assert cod.later.now == later.cod.now
    >>> assert mem.later.now == later.mem.now
    """
    category = symmetric.Category
    ty_factory = Ty[category.ob]

    now: category.ar
    dom: ty_factory = None
    cod: ty_factory = None
    mem: ty_factory = None
    _later: Callable[[], Stream[category]] = None

    later, is_constant = Ty.later, Ty.is_constant
    head, tail = Ty.head, Ty.tail

    def __init__(
            self, now: category.ar,
            dom: ty_factory = None,
            cod: ty_factory = None,
            mem: ty_factory = None,
            _later: Callable[[], Stream[category]] = None):
        if dom is None or cod is None:
            if mem is not None or _later is not None:
                raise ValueError(
                    "Cannot have mem or _later if dom or cod is None.")
        dom = Ty[self.category.ob](now.dom) if dom is None else dom
        cod = Ty[self.category.ob](now.cod) if cod is None else cod
        mem = Ty[self.category.ob]() if mem is None else mem
        for typ in (dom, cod, mem):
            assert_isinstance(typ, Ty)
        if not isinstance(now, self.category.ar):
            now = self.category.ar(
                now, dom.now + mem.now, cod.now + mem.later.now)
        if now.dom != dom.now + mem.now:
            raise AxiomError(f"{dom.now + mem.now} != {now.dom}")
        if now.cod != cod.now + mem.later.now:
            raise AxiomError(f"{dom.now + mem.later.now} != {now.dom}")
        if _later is None:
            if not all(x.is_constant for x in [dom, cod, mem]):
                raise ValueError(
                    "Constant streams should have constant dom, cod and mem")
        self.dom, self.cod, self.mem = dom, cod, mem
        self.now, self._later = now, _later

    def check_later(self):
        """ Check that later has consistent domain, codomain and memory. """
        later = self.later
        assert_isinstance(later, type(self))
        assert self.dom.later.now == later.dom.now
        assert self.cod.later.now == later.cod.now
        assert self.mem.later.now == later.mem.now

    mem_dom = property(lambda self: self.mem.now)
    mem_cod = property(lambda self: self.mem.later.now)

    @classmethod
    def singleton(cls, arg: category.ar) -> Stream:
        """
        Construct the stream with a given arrow now and the empty stream later.
        """
        dom, cod = map(Ty[cls.category.ob].singleton, (arg.dom, arg.cod))
        return cls(arg, dom, cod, _later=lambda: cls.id())

    @classmethod
    def sequence(
            cls, name: str, dom: Ty, cod: Ty, mem: Ty = None, n_steps: int = 0,
            box_factory=symmetric.Box) -> Stream:
        """
        Produce a stream of boxes indexed by a time step.

        Example
        -------
        >>> x, y, m = [Ty.sequence(symmetric.Ty(n)) for n in "xym"]
        >>> f = Stream.sequence("f", x @ m.delay(), y @ m)
        >>> for fi in [f.now, f.later.now, f.later.later.now]:
        ...     print(fi, ":", fi.dom, "->", fi.cod)
        f0 : x0 -> y0 @ m0
        f1 : x1 @ m0 -> y1 @ m1
        f2 : x2 @ m1 -> y2 @ m2
        """
        mem = Ty[cls.category.ob]() if mem is None else mem
        now = box_factory(
            f"{name}{n_steps}", dom.now @ mem.now, cod.now @ mem.later.now)
        return cls(now, dom, cod, mem, _later=lambda: cls.sequence(
            name, dom.later, cod.later, mem.later, n_steps + 1, box_factory))

    @inductive
    def delay(self) -> Stream:
        """ Delay a stream by one time step, shortened to `self.d`. """
        dom, cod, mem = [x.delay() for x in (self.dom, self.cod, self.mem)]
        now, _later = self.category.ar.id(self.mem.now), lambda: self
        return type(self)(now, dom, cod, mem, _later)

    d = property(lambda self: self.delay())

    @inductive
    def unroll(self) -> Stream:
        """
        Unrolling a stream for `n_steps`.

        Example
        -------

        >>> from discopy.drawing import Equation
        >>> f = Stream.sequence("f", *map(Ty.sequence, "xym"))
        >>> Equation(f.now, f.unroll().now, f.unroll(2).now, symbol=',').draw(
        ...     figsize=(8, 4), path="docs/_static/stream/unroll.png")

        .. image:: /_static/stream/unroll.png
            :align: center
        """
        later = self.later
        dom, cod = self.dom.unroll(), self.cod.unroll()
        mem = Ty[self.category.ob](self.mem.now, lambda: self.mem.later.later)
        now = self.dom.now @ self.category.ar.swap(later.dom.now, self.mem_dom)
        now >>= self.now @ later.dom.now
        now >>= self.cod.now @ self.category.ar.swap(
            self.mem_cod, later.dom.now) >> self.cod.now @ later.now
        return type(self)(now, dom, cod, mem, _later=lambda: later.later)

    @classmethod
    def id(cls, x: Optional[Ty] = None) -> Stream:
        """
        Construct a stream of identity arrows.

        Example
        -------
        >>> id_x = Stream.id(Ty.sequence('x'))
        >>> print(id_x.now, id_x.later.now, id_x.later.later.now)
        Id(x0) Id(x1) Id(x2)
        """
        x = Ty[cls.category.ob]() if x is None else x
        assert_isinstance(x, Ty)
        now, dom, cod = cls.category.ar.id(x.now), x, x
        _later = None if x.is_constant else lambda: cls.id(x.later)
        return cls(now, dom, cod, _later=_later)

    @unbiased
    def then(self, other: Stream) -> Stream:
        """
        Composition of streams is given by swapping the memories as follows:

        Example
        -------
        >>> x, y, z, m, n = map(Ty.sequence, "xyzmn")
        >>> f = Stream.sequence("f", x, y, m)
        >>> g = Stream.sequence("g", y, z, n)
        >>> (f >> g).now.draw(path="docs/_static/stream/stream-then.png")

        .. image:: /_static/stream/stream-then.png
            :align: center
        """
        swap = self.category.ar.swap
        now = self.now @ other.mem_dom
        now >>= self.cod.now @ swap(self.mem_cod, other.mem_dom)
        now >>= other.now @ self.mem_cod
        now >>= other.cod.now @ swap(other.mem_cod, self.mem_cod)
        dom, cod, mem = self.dom, other.cod, self.mem @ other.mem
        _later = None if self.is_constant and other.is_constant else (
            lambda: self.later >> other.later)
        return type(self)(now, dom, cod, mem, _later)

    @unbiased
    def tensor(self, other: Stream) -> Stream:
        """
        Tensor of streams is given by swapping the memories as follows:

        Example
        -------
        >>> x, y, z, w, m, n = map(Ty.sequence, "xyzwmn")
        >>> f = Stream.sequence("f", x, y, m)
        >>> g = Stream.sequence("g", z, w, n)
        >>> (f @ g).now.draw(path="docs/_static/stream/stream-tensor.png")

        .. image:: /_static/stream/stream-tensor.png
            :align: center
        """
        assert_isinstance(other, Stream)
        swap = self.category.ar.swap
        now = self.dom.now @ swap(other.dom.now, self.mem_dom) @ other.mem_dom
        now >>= self.now @ other.now
        now >>= self.cod.now @ swap(
            self.mem_cod, other.cod.now) @ other.mem_cod
        dom = self.dom @ other.dom
        cod = self.cod @ other.cod
        mem = self.mem @ other.mem
        _later = None if self.is_constant and other.is_constant else (
            lambda: self.later.tensor(other.later))
        return type(self)(now, dom, cod, mem, _later)

    @classmethod
    def swap(cls, left: Ty, right: Ty) -> Stream:
        """ Construct a stream of swaps. """
        now = cls.category.ar.swap(left.now, right.now)
        dom, cod = left @ right, right @ left
        _later = None if left.is_constant and right.is_constant else (
            lambda: cls.swap(left.later, right.later))
        return cls(now, dom, cod, _later=_later)

    @classmethod
    def copy(cls, dom: Ty, n: int = 2) -> Stream:
        """ Construct a stream of diagonal morphisms. """
        now, cod = cls.category.ar.copy(dom.now, n), dom ** n
        _later = None if dom.is_constant else lambda: cls.copy(dom.later, n)
        return cls(now, dom, cod, _later=_later)

    def feedback(
        self, dom: Ty = None, cod: Ty = None, mem: Ty = None, _first_call=True
    ) -> Stream:
        """
        The delayed feedback of a monoidal stream.

        Parameters:
            dom (Ty) : The domain of the result.
            cod (Ty) : The domain of the result.
            mem (Ty) : The memory over which we are taking a feedback.

        Example
        -------
        >>> x, y, m = [Ty.sequence(symmetric.Ty(n)) for n in "xym"]
        >>> f = Stream.sequence("f", x @ m.delay(), y @ m)
        >>> fb = f.feedback(x, y, m)

        >>> from discopy.drawing import Equation
        >>> Equation(f.unroll(2).now, fb.unroll(2).now, symbol="$\\\\mapsto$"
        ...     ).draw(path="docs/_static/stream/feedback-unrolling.png")

        .. image:: /_static/stream/feedback-unrolling.png
            :align: center
        """
        if mem is None or dom is None or cod is None:
            if not self.is_constant or dom is not None or cod is not None:
                raise NotImplementedError

        assert self.dom.now == dom.now if _first_call else (
            self.dom.now == dom.now + mem.now)
        assert self.cod.now == cod.now + mem.now if _first_call else (
            self.cod.now == cod.now + mem.later.now)

        def _later():
            return self.later.feedback(dom.later, cod.later, mem.later, False)
        mem = mem.delay() if _first_call else mem
        return type(self)(self.now, dom, cod, mem @ self.mem, _later)

    followed_by = id


@dataclass
class Category(symmetric.Category):
    """ Syntactic sugar for `Category(Ty[category.ob], Stream[category])`. """
    def __init__(self, ob: type = None, ar: type = None):
        ar = Stream if ar is None else Stream[symmetric.Category(ob, ar)]
        ob = Ty if ob is None else Ty[ob]
        super().__init__(ob, ar)
