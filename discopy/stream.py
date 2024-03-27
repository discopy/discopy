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

from typing import Callable
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
        return cls(now=x, later=None)

    def delay(self) -> Ty:
        """ Delays a stream of types by pre-pending with the unit. """
        return type(self)(type(self)(), lambda: self)

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
    dom: ty_factory
    cod: ty_factory
    mem: ty_factory
    _later: Callable[[], Stream[category]] = None

    def __init__(
            self, now: category.ar, dom: ty_factory, cod: ty_factory,
            mem: ty_factory, _later: Callable[[], Stream[category]] = None):
        now = now if isinstance(
            now, self.category.ar) else self.category.ar(now, dom, cod)
        assert_isinstance(now, self.category.ar)
        if _later is None:
            dom, cod = map(Ty[self.category.ob], (now.dom, now.cod))
            mem = Ty[self.category.ob]()
        assert_isinstance(dom, Ty[self.category.ob])
        assert_isinstance(cod, Ty[self.category.ob])
        assert_isinstance(mem, Ty[self.category.ob])
        if now.dom != dom.now:
            raise AxiomError
        if now.cod != cod.now + mem.now:
            raise AxiomError
        self.dom, self.cod, self.mem = dom, cod, mem
        self.now, self._later = now, _later

    @property
    def later(self):
        return self._later or (lambda: self)

    @classmethod
    def constant(cls, diagram: category.ar) -> Stream:
        return cls(dom=None, cod=None, mem=None, now=diagram, _later=None)

    def delay(self) -> Stream:
        dom, cod, mem = [x.delay() for x in (self.dom, self.cod, self.mem)]
        now, later = self.category.ar.id(), lambda: self
        return type(self)(now, dom, cod, mem, later)

    def unroll(self, n_steps=1) -> Stream:
        assert_isinstance(n_steps, int)
        if n_steps < 0:
            raise ValueError
        if n_steps == 0:
            return self
        if n_steps > 1:
            return self.unroll().unroll(n_steps - 1)
        later = self.later()
        dom, cod, mem = self.dom.unroll(), self.cod.unroll(), later.mem
        now = self.now @ later.dom.now >> self.cod.now @ self.category.ar.swap(
            self.mem.now, later.dom.now) >> self.cod.now @ later.now
        return type(self)(now, dom, cod, mem, later.later)

    @classmethod
    def id(cls, x: Ty) -> Stream:
        raise NotImplementedError

    def then(self, *others: Stream) -> Stream:
        raise NotImplementedError

    def tensor(self, *others: Stream) -> Stream:
        raise NotImplementedError

    @classmethod
    def swap(cls, left: Ty, right: Ty) -> Stream:
        raise NotImplementedError

    @classmethod
    def copy(cls, dom: Ty, n: int = 2) -> Stream:
        raise NotImplementedError

    def feedback(
            self, dom: Ty = None, cod: Ty = None, mem: Ty = None) -> Stream:
        if mem is None and hasattr(self.category.ob, "__getitem__"):
            mem = self.dom[-1:]
            dom = self.dom[:-1]
            cod = self.cod[:-1]
        elif mem is None:
            raise NotImplementedError
        if self.dom.now != dom.now:
            raise AxiomError
        if self.cod.now != cod.now + mem.now:
            raise AxiomError
        type(self)(dom, cod, self.mem @ mem, self.now, self.later)


@dataclass
class Category(symmetric.Category):
    def __init__(self, ob: type, ar: type):
        super().__init__(Ty[ob], Stream[ar])
