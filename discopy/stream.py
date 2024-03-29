"""
The category of monoidal streams over a monoidal category.

We adapted the definition of intensional streams from :cite:t:`DiLavoreEtAl22`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Stream
    Category
"""
from __future__ import annotations

from typing import Callable, Optional
from dataclasses import dataclass

from discopy import symmetric
from discopy.python import is_tuple
from discopy.utils import (
    AxiomError, Composable, Whiskerable, NamedGeneric,
    assert_isinstance, unbiased, inductive, classproperty, factory_name)


@dataclass
class Ty(NamedGeneric['base']):
    """
    A `stream.Ty[base]` is a `base` for now and an optional function from the
    empty tuple to `stream.Ty[base]` for later, the constant stream by default.
    """
    base = symmetric.Ty

    now: base = None
    _later: Callable[[], Ty[base]] = None

    factory = classproperty(lambda cls: cls)

    def __init__(
            self, now: base = None, _later: Callable[[], Ty[base]] = None):
        if is_tuple(self.base) and not isinstance(now, (tuple, type(None))):
            now = (now, )
        origin = getattr(self.base, "__origin__", self.base)
        now = now if isinstance(now, origin) else (
            self.base() if now is None else self.base(now))
        self.now, self._later = now, _later

    def __repr__(self):
        factory = f"{factory_name(type(self))}[{factory_name(self.base)}]"
        _later = "" if self.is_constant else f", _later={repr(self._later)}"
        return factory + f"({repr(self.now)}{_later})"

    @property
    def later(self):
        return self if self.is_constant else self._later()
    
    head, tail = property(lambda self: self.singleton(self.now)), later
    
    @property
    def is_constant(self):
        return self._later is None

    @classmethod
    def constant(cls, x: base):
        """ Constructs the constant stream for a given base type `x`. """
        return cls(now=x, _later=None)

    @classmethod
    def singleton(cls, x: base):
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
    
    @classmethod
    def sequence(cls, x: base, n_steps: int = 0):
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

    def unroll(self) -> Ty:
        return type(self)(self.now @ self.later.now, self.later._later)

    def map(self, func: Callable[[base], base]) -> Ty:
        return type(self)(func(self.now), lambda: self.later.map(func))

    @unbiased
    def tensor(self, other: Ty) -> Ty:
        if not isinstance(other, Ty):
            return NotImplemented
        return type(self)(
            self.now + other.now, lambda: self.later + other.later)

    __add__ = __matmul__ = symmetric.Ty.__matmul__
    __pow__ = symmetric.Ty.__pow__


@dataclass
class Stream(Composable, Whiskerable, NamedGeneric['category']):
    """
    A `Stream[category]` is given by a triple of `stream.Ty[category.ob]` for
    domain, codomain and memory, a `category.ar` for `now` and an optional
    function from the empty tuple to `Stream[category]` for `_later`.

    If `_later is None` then a constant stream is initialised.
    In that case, the domain, codomain and memory are computed from `now`.
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
        if not isinstance(now, self.category.ar):
            now = self.category.ar(now, dom.now, cod.now)
        dom = Ty[self.category.ob](now.dom) if dom is None else dom
        cod = Ty[self.category.ob](now.cod) if cod is None else cod
        mem = Ty[self.category.ob]() if mem is None else mem
        self.dom, self.cod, self.mem = dom, cod, mem
        self.now, self._later = now, _later
        if now.dom != dom.now + self.mem_dom:
            raise AxiomError
        if now.cod != cod.now + self.mem_cod:
            raise AxiomError
    
    @property
    def mem_dom(self):
        return self.mem.now
    
    @property
    def mem_cod(self):
        return self.mem.later.now

    @classmethod
    def constant(cls, diagram: category.ar) -> Stream:
        return cls(diagram)

    @classmethod
    def singleton(cls, arg: category.ar) -> Stream:
        dom, cod = map(Ty[cls.category.ob].singleton, (arg.dom, arg.cod))
        return cls(arg, dom, cod, _later=lambda: cls.id())
    
    @classmethod
    def sequence(cls, box: symmetric.Box, n_steps: int = 0) -> Stream:
        ty_factory = Ty[cls.category.ob]
        dom, cod = ((ty_factory.sequence(x, n_steps))
                    for x in [box.dom, box.cod])
        now = type(box)(f"{box.name}{n_steps}", dom.now, cod.now)
        _later = lambda: cls.sequence(box, n_steps + 1)
        return cls(now, dom, cod, _later=_later)

    @inductive
    def delay(self) -> Stream:
        dom, cod, mem = [x.delay() for x in (self.dom, self.cod, self.mem)]
        now, _later = self.category.ar.id(self.mem.now), lambda: self
        return type(self)(now, dom, cod, mem, _later)

    @inductive
    def unroll(self) -> Stream:
        """
        Unrolling a stream for `n_steps`.

        Example
        -------

        >>> from discopy.drawing import Equation
        >>> x, y = map(symmetric.Ty, "xy")
        >>> now = symmetric.Box('f', x @ y, x @ y)
        >>> f = Stream(now, dom=Ty(x), cod=Ty(x), mem=Ty(y))
        >>> Equation(f.now, f.unroll().now, f.unroll(2).now, symbol=',').draw(
        ...     figsize=(8, 4), path="docs/_static/stream/unroll.png")

        .. image:: /_static/stream/unroll.png
            :align: center
        """
        later = self.later
        dom, cod = self.dom.unroll(), self.cod.unroll()
        mem_dom, mem_cod = self.mem.now, later.mem_cod
        mem = Ty[self.category.ob](mem_dom, self.mem.later._later)
        now = self.dom.now @ self.category.ar.swap(later.dom.now, self.mem_dom)
        now >>= self.now @ later.dom.now
        now >>= self.cod.now @ self.category.ar.swap(
            self.mem_cod, later.dom.now) >> self.cod.now @ later.now
        return type(self)(now, dom, cod, mem, _later=later._later)

    @classmethod
    def id(cls, x: Optional[Ty] = None) -> Stream:
        x = Ty[cls.category.ob]() if x is None else x
        assert_isinstance(x, Ty)
        now, dom, cod = cls.category.ar.id(x.now), x, x
        _later = None if x.is_constant else lambda: cls.id(x.later)
        return cls(now, dom, cod, _later=_later)

    @unbiased
    def then(self, other: Stream) -> Stream:
        swap = self.category.ar.swap
        now = self.now @ other.mem_dom
        now >>= self.cod.now @ swap(self.mem_cod, other.mem_dom)
        now >>= other.now @ self.mem_cod
        now >>= other.cod.now @ swap(other.mem_dom, self.mem_cod)
        dom, cod, mem = self.dom, other.cod, self.mem @ other.mem
        _later = None if self._later is None and other._later is None else (
            lambda: self.later >> other.later)
        return type(self)(now, self.dom, other.cod, mem, _later)

    @unbiased
    def tensor(self, other: Stream) -> Stream:
        assert_isinstance(other, Stream)
        swap = self.category.ar.swap
        now = self.dom.now @ swap(other.dom.now, self.mem_dom) @ other.mem_dom
        now >>= self.now @ other.now
        now >>= self.cod.now @ swap(
            self.mem_cod, other.cod.now) @ other.mem_cod
        dom = self.dom @ other.dom
        cod = self.cod @ other.cod
        mem = self.mem @ other.mem
        def _later():
            return self.later.tensor(other.later)
        return type(self)(now, dom, cod, mem, _later)

    @classmethod
    def swap(cls, left: Ty, right: Ty) -> Stream:
        now = cls.category.ar.swap(left.now, right.now)
        dom, cod = left @ right, right @ left
        _later = None if left.is_constant and right.is_constant else (
            lambda: cls.swap(left.later, right.later))
        return cls(now, dom, cod, _later=_later)

    @classmethod
    def copy(cls, dom: Ty, n: int = 2) -> Stream:
        now, cod = cls.category.ar.copy(dom.now, n), dom ** n
        _later = None if dom.is_constant else lambda: cls.copy(dom.later, n)
        return cls(now, dom, cod, _later=_later)

    def feedback(
            self, dom: Ty = None, cod: Ty = None, mem: Ty = None) -> Stream:
        """
        The delayed feedback of a monoidal stream.

        Example
        -------

        >>> def feedback_example(n=0):
        ...     x, y, z = [Ty.sequence(x, n) for x in "xyz"]
        ...     dom, cod = x.now @ z.now, y.now @ z.later.now
        ...     now = symmetric.Box(f"f{n}", dom, cod)
        ...     _later = lambda: feedback_example(n + 1)
        ...     return Stream(now, x @ z, y @ z.later, Ty(), _later)
        >>> f, fb = feedback_example(), feedback_example().feedback()
        >>> from discopy.drawing import Equation
        >>> Equation(f.unroll(3).now, fb.unroll(3).now, symbol="$\\\\mapsto$"
        ...     ).draw(path="docs/_static/stream/feedback.png")

        .. image:: /_static/stream/feedback.png
            :align: center
        """
        if mem is None:
            dom, cod = [X.map(lambda x: x[:-1]) for X in (self.dom, self.cod)]
            mem = self.dom.map(lambda x: x[-1:])
        if self.now.dom != dom.now + mem.now:
            raise AxiomError
        if self.cod.now != cod.now + mem.later.now:
            raise AxiomError
        def _later():
            return self.later.feedback(dom.later, cod.later, mem.later)
        return type(self)(self.now, dom, cod, self.mem @ mem, _later)


@dataclass
class Category(symmetric.Category):
    """ Syntactic sugar for `Category(Ty[category.ob], Stream[category])`. """
    def __init__(self, ob: type, ar: type):
        super().__init__(Ty[ob], Stream[symmetric.Category(ob, ar)])


Stream.followed_by = classmethod(Stream.id.__func__)
