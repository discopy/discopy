"""
The category of monoidal streams over a monoidal category.

We follow the definition of intensional streams from  :cite:t:`DiLavoreEtAl22`.

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
    assert_isinstance, unbiased, factory_name)


@dataclass
class Ty(NamedGeneric['base']):
    """
    A `stream.Ty[base]` is a `base` for now and an optional function from the
    empty tuple to `stream.Ty[base]` for later, the constant stream by default.
    """
    base = symmetric.Ty

    now: base = None
    _later: Callable[[], Ty[base]] = None

    def __init__(
            self, now: base = None, _later: Callable[[], Ty[base]] = None):
        if is_tuple(self.base) and not isinstance(now, tuple):
            now = (now, )
        origin = getattr(self.base, "__origin__", self.base)
        now = now if isinstance(now, origin) else (
            self.base() if now is None else self.base(now))
        self.now, self._later = now, _later

    def __repr__(self):
        factory = f"{factory_name(type(self))}[{factory_name(self.base)}]"
        _later = "" if self._later is None else f", _later={repr(self._later)}"
        return factory + f"({repr(self.now)}{_later})"

    @property
    def later(self):
        return self._later or (lambda: self)

    @classmethod
    def constant(cls, x: base):
        """ Constructs the constant stream for a given base type `x`. """
        return cls(now=x, _later=None)

    @classmethod
    def singleton(cls, x: base):
        """
        Constructs the stream with `x` now and the empty stream later.
        
        >>> x = Ty.singleton(symmetric.Ty('x'))
        >>> print(x.now, x.later().now, x.later().later().now)
        x Ty() Ty()
        """
        return cls(now=x, _later=lambda: cls())
    
    @classmethod
    def sequence(cls, name: str, n_steps: int = 0):
        """
        Constructs the stream `x0`, `x1`, etc.
        
        >>> x = Ty.sequence('x')
        >>> print(x.now, x.later().now, x.later().later().now)
        x0 x1 x2
        """
        now = cls.base(f"{name}{n_steps}")
        _later = lambda: cls.sequence(name, n_steps + 1)
        return cls(now, _later)

    def delay(self, n_steps: int = 1) -> Ty:
        """ Delays a stream of types by pre-pending with the unit. """
        assert_isinstance(n_steps, int)
        if n_steps < 0:
            raise ValueError
        if n_steps == 1:
            return type(self)(self.base(), lambda: self)
        return self if n_steps == 0 else self.delay().delay(n_steps - 1)

    def unroll(self) -> Ty:
        return type(self)(self.now @ self.later().now, self.later().later)

    def map(self, func: Callable[[base], base]) -> Ty:
        return type(self)(func(self.now), lambda: self.later().map(func))

    def __getitem__(self, key) -> Ty:
        return self.map(lambda x: x[key])

    @unbiased
    def tensor(self, other: Ty) -> Ty:
        return type(self)(
            self.now + other.now, lambda: self.later() + other.later())

    __add__ = __matmul__ = symmetric.Ty.__matmul__


@dataclass
class Stream(Composable, Whiskerable, NamedGeneric['category']):
    """
    A `Stream[category]` is given by a triple of `stream.Ty[category.ob]` for
    domain, codomain and memory, a `category.ar` for `now` and an optional
    function from the empty tuple to `Stream[category]` for `later`.

    If `later is None` then a constant stream is initialised.
    In that case, the domain, codomain and memory are computed from `now`.
    """
    category = symmetric.Category
    ty_factory = Ty[category.ob]

    now: category.ar
    dom: ty_factory = None
    cod: ty_factory = None
    mem: ty_factory = None
    _later: Callable[[], Stream[category]] = None

    def __init__(
            self, now: category.ar,
            dom: ty_factory = None,
            cod: ty_factory = None,
            mem: ty_factory = None,
            _later: Callable[[], Stream[category]] = None,
            _nested_check=True):
        assert_isinstance(now, self.category.ar)
        dom = Ty[self.category.ob](now.dom) if dom is None else dom
        cod = Ty[self.category.ob](now.cod) if cod is None else cod
        mem = Ty[self.category.ob]() if mem is None else mem
        self.dom, self.cod, self.mem = dom, cod, mem
        self.now, self._later = now, _later
        if now.dom != dom.now + self.mem_dom:
            raise AxiomError
        if now.cod != cod.now + self.mem_cod:
            raise AxiomError
        if _nested_check:
            later = self.later()
            assert later.mem_dom == self.mem_cod
            assert later.dom.now == self.dom.later().now
            assert later.cod.now == self.cod.later().now
    
    @property
    def mem_dom(self):
        return self.mem.now
    
    @property
    def mem_cod(self):
        return self.mem.later().now

    @property
    def later(self):
        return self._later or (lambda: self)

    @classmethod
    def constant(cls, diagram: category.ar) -> Stream:
        return cls(diagram)

    @classmethod
    def singleton(cls, diagram: category.ar) -> Stream:
        dom, cod = map(Ty[self.category.ob].singleton, diagram.dom)
        return cls(diagram, dom, cod, _later=lambda : cls.id())
    
    @classmethod
    def sequence(cls, box: symmetric.Box, n_steps: int = 0) -> Stream:
        dom, cod = (Ty[cls.category.ob].sequence(name, n_steps)
                    for name in [box.dom.name, box.cod.name])
        now = type(box)(f"{box.name}{n_steps}", dom.now, cod.now)
        _later = lambda: cls.sequence(box, n_steps + 1)
        return cls(now, dom, cod, _later=_later, _nested_check=False)

    def delay(self, n_steps=1) -> Stream:
        if n_steps != 1:
            return Ty.delay(self, n_steps)
        dom, cod, mem = [x.delay() for x in (self.dom, self.cod, self.mem)]
        now, later = self.category.ar.id(self.mem.now), lambda: self
        return type(self)(now, dom, cod, mem, _later=later, _nested_check=False)

    def unroll(self, n_steps=1) -> Stream:
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
        assert_isinstance(n_steps, int)
        if n_steps < 0:
            raise ValueError
        if n_steps == 0:
            return self
        if n_steps > 1:
            return self.unroll().unroll(n_steps - 1)
        later = self.later()
        dom, cod = self.dom.unroll(), self.cod.unroll()
        mem_dom, mem_cod = self.mem.now, later.mem_cod
        mem = Ty[self.category.ob](mem_dom, self.mem.later().later)
        now = self.dom.now @ self.category.ar.swap(later.dom.now, self.mem_dom)
        now >>= self.now @ later.dom.now
        now >>= self.cod.now @ self.category.ar.swap(
            self.mem_cod, later.dom.now) >> self.cod.now @ later.now
        return type(self)(now, dom, cod, mem, later.later)

    @classmethod
    def id(cls, x: Optional[Ty] = None) -> Stream:
        x = Ty[cls.category.ob]() if x is None else x
        assert_isinstance(x, Ty)
        now, dom, cod = cls.category.ar.id(x.now), x, x
        _later = None if x._later is None else lambda: cls.id(x.later())
        return cls(now, dom, cod, _later=_later)

    def then(self, *others: Stream) -> Stream:
        if not others:
            return self

        other = others[0]
        swap = self.category.ar.swap

        now = self.now @ other.mem_dom
        now >>= self.dom.now @ swap(self.mem_cod, other.mem_dom)
        now >>= other.now @ self.mem_cod
        now >>= other.cod.now @ swap(other.mem_dom, self.mem_cod)

        mem_dom = self.mem_dom @ other.mem_dom
        mem_cod = self.mem_dom @ other.mem_cod

        def _later():
            return self.later() >> other.later()

        combined = type(self)(
            now, self.dom, other.cod, mem_dom, mem_cod, _later)

        return combined.then(*others[1:])

    def tensor(self, *others: Stream) -> Stream:
        if not others:
            return self

        other = others[0]
        swap = self.category.ar.swap
        now = self.dom.now @ swap(other.dom.now, self.mem_dom) @ other.mem_dom
        now >>= self.now @ other.now
        now >>= self.cod.now @ swap(
            self.mem_cod, other.cod.now) @ other.mem_cod

        mem_dom = self.mem_dom @ other.mem_dom
        mem_cod = self.mem_cod @ other.mem_dom

        dom = self.dom @ other.dom
        cod = self.cod @ other.cod

        def _later():
            return self.later().tensor(other.later())

        combined = type(self)(now, dom, cod, mem_dom, mem_cod, _later)

        return combined.tensor(*others[1:])

    @classmethod
    def swap(cls, left: Ty, right: Ty) -> Stream:
        return cls.constant(cls.category.ar.swap(left.now, right.now))

    @classmethod
    def copy(cls, dom: Ty, n: int = 2) -> Stream:
        return cls.constant(cls.category.ar.copy(dom.now, n))

    def feedback(
            self, dom: Ty = None, cod: Ty = None, mem: Ty = None, _nested_check=True) -> Stream:
        """
        The delayed feedback of a monoidal stream.

        Example
        -------
        
        >>> x, y, z = map(symmetric.Ty, "xyz")
        >>> f0 = symmetric.Box('f0', x, y @ z)
        >>> f1 = symmetric.Box('f1', x @ z, y @ z)
        >>> dom, cod = Ty(x) @ Ty(z).delay(), Ty(y @ z)
        >>> f = Stream(f0, dom, cod, _later=lambda: Stream(f1))
        >>> f.feedback(dom=Ty(x), cod=Ty(y), mem=Ty(z)).unroll().now.draw()
        """
        if mem is None or dom is None or cod is None:
            raise NotImplementedError
        if self.now.dom != dom.now + mem.now:
            raise AxiomError
        if self.cod.now != cod.now + mem.later().now:
            raise AxiomError
        def _later():
            return self.later().feedback(
                dom.later(), cod.later(), mem.later(), _nested_check=False)
        return type(self)(self.now, dom, cod, self.mem @ mem, _later, _nested_check=_nested_check)


@dataclass
class Category(symmetric.Category):
    """ Syntactic sugar for `Category(Ty[category.ob], Stream[category])`. """
    def __init__(self, ob: type, ar: type):
        super().__init__(Ty[ob], Stream[symmetric.Category(ob, ar)])


def generic_feedback_example(n_steps = 0, _nested_check=2):
    name = f"f{n_steps}"
    x, y, z = [Ty.sequence(x, n_steps) for x in "xyz"]
    now = symmetric.Box(name, x.now @ z.now, y.now @ z.later().now)
    _later = lambda: generic_feedback_example(n_steps + 1, _nested_check - 1)
    return Stream(now, x @ z, y @ z.later(), _later=_later, _nested_check=False)
