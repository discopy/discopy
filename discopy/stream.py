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

Example
-------

>>> x = Ty('x')
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
        """ Constructs the stream with `x` now and the empty stream later. """
        return cls(now=x, _later=lambda: cls())

    def delay(self, n_steps: int = 1) -> Ty:
        """ Delays a stream of types by pre-pending with the unit. """
        assert_isinstance(n_steps, int)
        if n_steps < 0:
            raise ValueError
        if n_steps == 1:
            return type(self)(type(self)(), lambda: self)
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
    mem_dom: category.ob = None
    mem_cod: category.ob = None
    _later: Callable[[], Stream[category]] = None

    def __init__(
            self, now: category.ar,
            dom: ty_factory = None, cod: ty_factory = None,
            mem_dom: category.ob = None, mem_cod: category.ob = None,
            _later: Callable[[], Stream[category]] = None):
        assert_isinstance(now, self.category.ar)
        dom = Ty[self.category.ob](now.dom) if dom is None else dom
        cod = Ty[self.category.ob](now.cod) if cod is None else cod
        mem_dom = self.category.ob() if mem_dom is None else mem_dom
        mem_cod = self.category.ob() if mem_cod is None else mem_cod
        if now.dom != dom.now + mem_dom:
            raise AxiomError
        if now.cod != cod.now + mem_cod:
            raise AxiomError
        self.dom, self.cod = dom, cod
        self.mem_dom, self.mem_cod = mem_dom, mem_cod
        self.now, self._later = now, _later

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

    def delay(self, n_steps=1) -> Stream:
        if n_steps != 1:
            return Ty.delay(self, n_steps)
        dom, cod = [x.delay() for x in (self.dom, self.cod)]
        mem_dom = mem_cod = self.mem_dom
        now, later = self.category.ar.id(self.mem_dom), lambda: self
        return type(self)(now, dom, cod, mem_dom, mem_cod, _later=later)

    def unroll(self, n_steps=1) -> Stream:
        """
        Unrolling a stream for `n_steps`.

        Example
        -------
        >>> from discopy.drawing import Equation
        >>> x, y = map(symmetric.Ty, "xy")
        >>> now = symmetric.Box('f', x @ y, x @ y)
        >>> f = Stream(now, dom=Ty(x), cod=Ty(x), mem_dom=y, mem_cod=y)
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
        if later.mem_dom != self.mem_cod:
            raise AxiomError
        dom, cod = self.dom.unroll(), self.cod.unroll()
        mem_dom, mem_cod = later.mem_dom, later.mem_cod
        now = self.dom.now @ self.category.ar.swap(later.dom.now, self.mem_dom)
        now >>= self.now @ later.dom.now
        now >>= self.cod.now @ self.category.ar.swap(
            self.mem_cod, later.dom.now) >> self.cod.now @ later.now
        return type(self)(now, dom, cod, mem_dom, mem_cod, later.later)

    @classmethod
    def id(cls, x: Optional[Ty] = None) -> Stream:
        x = Ty[cls.category.ob]() if x is None else x
        assert_isinstance(x, Ty)
        now, dom, cod = cls.category.ar.id(x.now), x, x
        _later = None if x._later is None else lambda : cls.id(x.later())
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

        later = lambda: self.later() >> other.later()

        mem_dom = self.mem_dom @ other.mem_dom
        mem_cod = self.mem_dom @ other.mem_cod

        combined = type(self)(
            now, self.dom, other.cod, mem_dom, mem_cod, later)

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

        later = lambda: self.later() @ other.later()

        mem_dom = self.mem_dom @ other.mem_dom
        mem_cod = self.mem_cod @ other.mem_dom

        dom = self.dom @ other.dom
        cod = self.cod @ other.cod

        combined = type(self)(now, dom, cod, mem_dom, mem_cod, later)

        return combined.tensor(*others[1:])



    @classmethod
    def swap(cls, left: Ty, right: Ty) -> Stream:
        return cls.constant(cls.category.ar.swap(left.now, right.now))

    @classmethod
    def copy(cls, dom: Ty, n: int = 2) -> Stream:
        return cls.constant(cls.category.ar.copy(dom.now, n))

    def set_mem_dom(self, mem_dom: category.ob) -> Stream:
        assert_isinstance(mem_dom, self.category.ob)
        if not issubclass(self.category.ob, (symmetric.Ty, tuple)):
            raise NotImplementedError
        dom = self.dom[:-len(mem_dom) or len(dom)]
        return type(self)(
            self.now, dom, self.cod, mem_dom, self.mem_cod, self._later)

    def feedback(
            self, dom: Ty = None, cod: Ty = None, mem: Ty = None) -> Stream:
        """
        The delayed feedback of a monoidal stream.

        Example
        -------
        >>> x, y = map(symmetric.Ty, "xy")
        >>> now = symmetric.Box('f', x @ x @ y, x @ y)
        >>> f = Stream.constant(now)
        >>> dom, cod, mem = Ty(x @ x) @ Ty.singleton(y), Ty(x), Ty(y)
        >>> f.feedback(dom, cod, mem).unroll()
        """
        if mem is None or dom is None or cod is None:
            raise NotImplementedError
        if self.now.dom != dom.now + self.mem_dom:
            raise AxiomError
        if self.cod.now != cod.now + self.mem_cod + mem.now:
            raise AxiomError
        mem_dom, mem_cod = self.mem_dom, self.mem_cod + mem.now
        def _later():
            return self.later().set_mem_dom(mem_cod).feedback(
                dom.later(), cod.later(), mem.later())
        return type(self)(self.now, dom, cod, mem_dom, mem_cod, _later)

@dataclass
class Category(symmetric.Category):
    def __init__(self, ob: type, ar: type):
        super().__init__(Ty[ob], Stream[symmetric.Category(ob, ar)])
