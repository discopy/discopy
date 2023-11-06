from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

from discopy import symmetric
from discopy.utils import (
    AxiomError, Composable, Whiskerable, NamedGeneric, assert_isinstance)


@dataclass
class Ty(NamedGeneric['base']):
    """
    A `stream.Ty[base]` is a `base` for now and an optional function from the
    empty tuple to `stream.Ty[base]` for later, the constant stream by default.
    """
    base = symmetric.Ty

    now: base = None
    later: Callable[[], Ty[base]] = None

    def __init__(self, now: base = None, later: Callable[[], Ty[base]] = None):
        now = now if isinstance(now, self.base) else (
            self.base() if now is None else self.base(now))
        later = (lambda: self) if later is None else later
        self.now, self.later = now, later

    @classmethod
    def constant(cls, x: base):
        """ Constructs the constant stream for a given base type `x`. """
        return cls(now=x, later=None)


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

    dom: ty_factory
    cod: ty_factory
    mem: ty_factory
    now: category.ar
    later: Callable[[], Stream[category]] = None

    def __init__(
            self, dom: ty_factory, cod: ty_factory, mem: ty_factory,
            now: category.ar, later: Callable[[], Stream[category]] = None):
        assert_isinstance(now, self.category.ar)
        if later is None:
            dom, cod = map(Ty[self.category.ob], (now.dom, now.cod))
            later, mem = (lambda: self), Ty[self.category.ob]()
        assert_isinstance(dom, Ty[self.category.ob])
        assert_isinstance(cod, Ty[self.category.ob])
        assert_isinstance(mem, Ty[self.category.ob])
        if now.dom != dom.now:
            raise AxiomError
        if now.cod != cod.now + mem.now:
            raise AxiomError
        self.dom, self.cod, self.mem = dom, cod, mem
        self.now, self.later = now, later

    @classmethod
    def constant(cls, diagram: category.ar) -> Stream:
        return cls(dom=None, cod=None, mem=None, now=diagram, later=None)

    def id(): ...
    def then(): ...
    def tensor(): ...
    def swap(): ...
    def feedback(): ...

