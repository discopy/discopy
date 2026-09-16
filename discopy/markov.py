# -*- coding: utf-8 -*-

"""
The free Markov category, i.e. a semicartesian category with a supply of
commutative comonoid, see :cite:t:`FritzLiang23`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Permutation
    Swap
    Trace
    Copy
    Merge
    Discard
    Sum
    Bubble
    Functor
    Context
    TermBase
    Constant
    Variable
    Application


Axioms
------

>>> x = Ty('x')

>>> copy, merge = Copy(x), Merge(x)
>>> unit, delete = Merge(x, n=0), Copy(x, n=0)

Commutative monoid
==================

>>> unitality = Equation(unit @ x >> merge, Id(x), x @ unit >> merge)
>>> associativity = Equation(merge @ x >> merge, x @ merge >> merge)
>>> commutativity = Equation(Swap(x, x) >> merge, merge)
>>> assert unitality and associativity and commutativity
>>> Equation(unitality, associativity, commutativity, symbol='').draw(
...     doctest="docs/_static/frobenius/monoid.svg")

.. image:: /_static/frobenius/monoid.svg
    :align: center

Cocommutative comonoid
======================

>>> counitality = Equation(copy >> delete @ x, Id(x), copy >> x @ delete)
>>> coassociativity = Equation(copy >> copy @ x, copy >> x @ copy)
>>> cocommutativity = Equation(copy >> Swap(x, x), copy)
>>> assert counitality and coassociativity and cocommutativity
>>> Equation(counitality, coassociativity, cocommutativity, symbol='').draw(
...     doctest="docs/_static/frobenius/comonoid.svg")

.. image:: /_static/frobenius/comonoid.svg
    :align: center

Coherence
=========

>>> assert Equation(Diagram.copy(x @ x, n=0), delete @ delete)
>>> assert Equation(Diagram.copy(x @ x),
...     copy @ copy >> x @ Swap(x, x) @ x)
>>> assert Equation(Diagram.merge(x @ x, n=0), unit @ unit)
>>> assert Equation(Diagram.merge(x @ x),
...     x @ Swap(x, x) @ x >> merge @ merge)

Note
----
Equality of Markov diagrams is computed by translation to hypergraph.
Both copy and merge boxes are translated to spiders, thus when they appear
in the same diagram they automatically satisfy the :mod:`frobenius` axioms.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from discopy import symmetric, monoidal, cmap, hypergraph
from discopy.abc import MarkovCategory
from discopy.cat import factory, Generator
from discopy.monoidal import Ty  # noqa: F401
from discopy.utils import assert_isatomic, assert_isinstance, factory_name


Layer = symmetric.Layer


@factory
class Diagram(symmetric.Diagram, MarkovCategory):
    """
    A Markov diagram is a symmetric diagram with :class:`Copy` boxes.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.

    Note
    ----
    We can create arbitrary Markov diagrams with the standard notation for
    Python functions.

    >>> x = Ty('x')
    >>> f = Box('f', x, x)

    >>> copy_then_apply = Diagram.from_callable(x, x @ x)(
    ...     lambda x: (f(x), f(x)))

    >>> @Diagram.from_callable(x, x @ x)
    ... def apply_then_copy(x):
    ...     y = f(x)
    ...     return y, y

    >>> Equation(copy_then_apply, apply_then_copy, symbol="$\\\\neq$").draw(
    ...     doctest="docs/_static/markov/copy_and_apply.svg")

    .. image:: /_static/markov/copy_and_apply.svg
    """
    @classmethod
    def spider_factory(cls, n_legs_in, n_legs_out, typ, phase=None):
        if phase is not None or 1 not in (n_legs_in, n_legs_out):
            raise ValueError
        return cls.copy_factory(typ, n_legs_out) if n_legs_in == 1\
            else cls.merge_factory(typ, n_legs_in)

    @classmethod
    def copy(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Make :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        from discopy import frobenius
        return frobenius.Diagram.spiders.__func__(cls, 1, n, x)

    @classmethod
    def merge(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        Merge :code:`n` copies of a given type :code:`x`.

        Parameters:
            x : The type to copy.
            n : The number of copies.
        """
        return cls.copy(x, n).dagger()

    @classmethod
    def discard(cls, x: monoidal.Ty, n=2) -> Diagram:
        """
        The discard of an atomic type :code:`x`.

        Parameters:
            x : The type to discard.
        """
        return cls.copy(x, 0)

    @Generator()
    def copy_factory(cls):
        return Copy

    @Generator()
    def merge_factory(cls):
        return Merge

    @Generator("copy_factory")
    def discard_factory(cls):
        return Discard


Box, Permutation, Swap = (
    Diagram.generator_factory, Diagram.permutation_factory,
    Diagram.swap_factory)


class Copy(Box):
    """
    The copy of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type to copy.
        n : The number of copies.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        name = f"Copy({x}" + ("" if n == 2 else f", {n}") + ")"
        self.generator_factory.__init__(
            self, name, dom=x, cod=x ** n,
            draw_as_spider=True, color="black", drawing_name="")

    def __new__(cls, x: monoidal.Ty, n: int = 2):
        return super().__new__(cls) if n else\
            cls.discard_factory.__new__(cls.discard_factory, x)

    def dagger(self) -> Merge:
        return self.merge_factory(self.dom, len(self.cod))

    def __repr__(self):
        return (
            factory_name(type(self)) + f"({repr(self.dom)}, {len(self.cod)})")


class Merge(Box):
    """
    The merge of an atomic type :code:`x` some :code:`n` number of times.

    Parameters:
        x : The type of wires to merge.
        n : The number of wires to merge.
    """
    def __init__(self, x: monoidal.Ty, n: int = 2):
        assert_isatomic(x, monoidal.Ty)
        name = f"Merge({x}" + ("" if n == 2 else f", {n}") + ")"
        self.generator_factory.__init__(
            self, name, dom=x ** n, cod=x,
            draw_as_spider=True, color="black", drawing_name="")

    def dagger(self) -> Copy:
        return self.copy_factory(self.cod, len(self.dom))

    def __repr__(self):
        return (
            factory_name(type(self)) + f"({repr(self.cod)}, {len(self.dom)})")


class Discard(Copy):
    """
    The discard of an atomic type :code:`x`.

    Parameters:
        x : The type to discard.
    """
    def __init__(self, x: monoidal.Ty, *args, **kwargs):
        super().__init__(x, 0)


Sum, Bubble = Diagram.sum_factory, Diagram.bubble_factory


class Functor(symmetric.Functor):
    """
    A Markov functor is a symmetric functor that preserves copies.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.

    Example
    -------

    We build a functor into python functions.

    >>> x = Ty('x')
    >>> add = Box('add', x @ x, x)
    >>> from discopy import python
    >>> F = Functor({x: int}, {add: lambda a, b: a + b},
    ...             cod=python.Function)
    >>> copy = Copy(x)
    >>> bialgebra_l = copy @ copy >> Id(x) @ Swap(x, x) @ Id(x) >> add @ add
    >>> bialgebra_r = add >> copy
    >>> assert F(bialgebra_l)(54, 46) == F(bialgebra_r)(54, 46)

    >>> Equation(bialgebra_l, bialgebra_r, symbol="=").draw(
    ...     doctest="docs/_static/markov/bialgebra.svg")

    .. image:: /_static/markov/bialgebra.svg
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Copy) and hasattr(self.cod, "copy"):
            return self.cod.copy(self(other.dom), len(other.cod))
        if isinstance(other, Merge) and hasattr(self.cod, "merge"):
            return self.cod.merge(self(other.cod), len(other.dom))
        return super().__call__(other)


CMap = cmap.CMap[Diagram]

Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Id = Diagram.id


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of Markov diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)


class TermBase(Box):
    """
    A term in the internal language of a Markov category: a
    :class:`Variable` can be copied and discarded, a :class:`Constant` is
    applied to terms and there is no abstraction since there are no
    exponentials — see :mod:`discopy.closed` for the linear lambda calculus.

    The ``dom`` of a term is the tensor of the types of its free variables,
    in order of first occurrence, and its ``cod`` is the type of the term.
    """
    functor: ClassVar[Functor] = None

    @abstractmethod
    def eval(self, functor: Functor = None, context: Context = None
             ) -> MarkovCategory:
        """
        The evaluation of a :class:`Functor` on a term gives a morphism in
        its codomain, from a ``context`` — the free variables of the term
        by default, of which the unused ones are discarded.
        """

    def draw(self, **kwargs):
        "Drawing a term by evaluating it in the free Markov category."
        return self.eval().draw(**kwargs)


type Term = Constant | Variable | Application


class Constant(TermBase):
    """
    A function symbol with a ``dom`` and a ``cod``, applied to terms with
    ``__call__``; a constant with an empty ``dom`` is itself a term.

    Example
    -------
    >>> X, Y = Ty('X'), Ty('Y')
    >>> f, x = Constant('f', X @ X, Y), Variable('x', X)
    >>> assert f(x, x).eval() == Copy(X) >> f
    """
    def __init__(self, name: str, dom: monoidal.Ty, cod: monoidal.Ty):
        super().__init__(name, dom, cod)
        self.freevars = []

    def __call__(self, *terms: Term) -> Application:
        return self.application_factory(self, terms)

    def eval(self, functor=None, context=None):
        if len(self.dom):
            raise ValueError("A constant with a domain must be applied.")
        functor = functor or self.functor
        if not context:
            return functor(self)
        return functor.cod.discard(functor(context.dom)) >> functor(self)

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.name!r}, {self.dom!r}, {self.cod!r})"


class Variable(TermBase):
    """
    A variable with a name and a type ``cod``.

    Example
    -------
    >>> x = Variable('x', Ty('X'))
    >>> assert x.eval() == Id(Ty('X'))
    """
    def __init__(self, name: str, cod: monoidal.Ty):
        super().__init__(name, cod, cod)
        self.freevars = [self]

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        if not context:
            return functor.cod.id(functor(self.cod))
        return functor.cod.tensor(*[
            functor.cod.id(functor(x.cod)) if x == self
            else functor.cod.discard(functor(x.cod))
            for x in context.inside])

    def __repr__(self):
        return factory_name(type(self)) + f"({self.name!r}, {self.cod!r})"


class Application(TermBase):
    """
    A constant applied to terms: the free variables are listed in order of
    first occurrence, a shared variable is copied and, in a context, an
    unused one is discarded.

    Example
    -------
    >>> X, Y, Z = Ty('X'), Ty('Y'), Ty('Z')
    >>> x, y = Variable('x', X), Variable('y', Y)
    >>> f, g = Constant('f', X @ X, Z), Constant('g', X @ Y, Z)
    >>> assert f(x, x).eval() == Copy(X) >> f
    >>> assert g(x, y).eval() == g
    >>> assert Equation(
    ...     g(x, y).eval(context=Context([y, x])), Swap(Y, X) >> g)
    """
    def __init__(self, symbol: Constant, args: tuple[Term, ...]):
        assert_isinstance(symbol, Constant)
        args = tuple(args)
        for arg in args:
            assert_isinstance(arg, TermBase)
        args_cod = self.ob().tensor(*[t.cod for t in args])
        if symbol.dom != args_cod:
            raise ValueError(f"Expected {symbol.dom}, got {args_cod}.")
        self.symbol, self.args = symbol, args
        self.freevars = list(dict.fromkeys(
            x for t in args for x in t.freevars))
        name = f"{symbol.name}({', '.join(map(str, args))})"
        dom = self.ob().tensor(*[x.cod for x in self.freevars])
        super().__init__(name, dom, symbol.cod)

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        if context is None and len(self.freevars)\
                == sum(len(t.freevars) for t in self.args):
            wiring = functor.cod.id(functor(self.ob()))
            for term in self.args:
                wiring = wiring @ term.eval(functor=functor)
            return wiring >> functor(self.symbol)
        context = Context(self.freevars) if context is None else context
        copies = functor.cod.id(functor(context.dom))\
            if len(self.args) == 1\
            else functor.cod.copy(functor(context.dom), len(self.args))
        wiring = functor.cod.id(functor(self.ob()))
        for term in self.args:
            wiring = wiring @ term.eval(functor=functor, context=context)
        return copies >> wiring >> functor(self.symbol)

    def __repr__(self):
        return f"{self.symbol!r}({', '.join(map(repr, self.args))})"


@dataclass
class Context:
    """
    A context is a list of variables, whose ``dom`` is the tensor of their
    types.

    Example
    -------
    >>> X = Ty('X')
    >>> assert Context([]).dom == Ty()
    >>> assert Context([Variable('x', X)]).dom == X
    """
    inside: list[Variable]
    category: ClassVar[type[MarkovCategory]] = Diagram

    @property
    def dom(self):
        return self.category.ob().tensor(*[x.cod for x in self.inside])


TermBase.functor = Functor.id(Diagram)
TermBase.application_factory = Application
