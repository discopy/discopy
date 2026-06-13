
"""
The free closed markov category, i.e. with copy, discard and exponentials.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Exp
    Diagram
    Box
    Eval
    Coeval
    Curry
    Sum
    Category
    Functor

Axioms
------

:meth:`Diagram.curry` and :meth:`Diagram.uncurry` are inverses.

>>> x, y, z = map(Ty, "xyz")
>>> f, g = Box('f', x, z << y), Box('g', x @ y, z)

>>> from discopy.drawing import Equation
>>> Equation(f.uncurry().curry(), f).draw(
...     path='docs/_static/closed/curry-left.png', margins=(0.1, 0.05))

.. image:: /_static/closed/curry-left.png
    :align: center

>>> Equation(g.curry().uncurry(), g).draw(
...     path='docs/_static/closed/uncurry.png')

.. image:: /_static/closed/uncurry.png
    :align: center
"""

from __future__ import annotations
from dataclasses import dataclass
from abc import abstractproperty
from typing import Dict, Callable, Union
from inspect import signature

from discopy import cat, biclosed, markov
from discopy.cat import factory
from discopy.utils import assert_isinstance


@factory
class Ty(biclosed.Ty):
    """
    A closed type is a biclosed type in a symmetric category where left and
    right exponentials coincide, i.e. `X << Y == X ** Y == Y >> X`.

    Applying a closed type to a function yields an :class:`Term` e.g.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> t = X(lambda x: (X >> Y)(lambda f: f(x)))
    >>> t.to_diagram().draw(
    ...     path='docs/_static/closed/diagram.png',
    ...     aspect="auto", figsize=(8, 8), margins=(0.2, 0))

    .. image:: /_static/closed/diagram.png
        :align: center

    Applying a closed type to a string yields a :class:`Constant` e.g.

    >>> N, S = Ty("N"), Ty("S")
    >>> Alice, loves, Bob = N("Alice"), (N >> (N >> S))("loves"), N("Bob")
    >>> loves(Alice)(Bob).to_diagram().draw(
    ...     path='docs/_static/closed/alice-loves-bob.png',
    ...     margins=(.3, 0), figsize=(5, 4))

    .. image:: /_static/closed/alice-loves-bob.png
        :align: center
    """
    def __pow__(self, other: Ty) -> Ty:
        return Exp(self, other) if isinstance(other, Ty)\
            else super().__pow__(other)

    def __lshift__(self, other):
        return Exp(self, other)

    def __rshift__(self, other):
        return Exp(other, self)

    def __call__(self, arg):
        if isinstance(arg, Callable):
            varnames = list(signature(arg).parameters.keys())
            if len(varnames) != 1:
                raise NotImplementedError
            var = Variable(self, varnames[0])
            return Abstraction(var, arg(var))
        if isinstance(arg, str):
            return Constant(self, arg)
        raise ValueError


class Exp(Ty, biclosed.Exp):
    "An exponential object in a markov category."
    __ambiguous_inheritance__ = (biclosed.Exp, )

    def __str__(self):
        return f"({self.exponent} >> {self.base})"


class TermBase:
    cod: Ty

    def __call__(self, other):
        return Application(self, other)

    @abstractproperty
    def freevars(self) -> list[Variable]: ...


Term = Union["Constant", "Variable", "Application", "Abstraction"]


@dataclass(frozen=True)
class Constant(TermBase):
    cod: Ty
    name: str

    def __str__(self):
        return self.name

    @property
    def freevars(self):
        return []

    def to_diagram(self, category=None, box_factory=None):
        category, box_factory = category or Category, box_factory or Box
        return box_factory(self.name, category.ar.ty_factory(), self.cod)


@dataclass(frozen=True)
class Variable(TermBase):
    cod: Ty
    name: str

    def __str__(self):
        return self.name

    @property
    def freevars(self):
        return [self]

    def to_diagram(self, category=None):
        return (category or Category).ar.id(self.cod)


@dataclass(frozen=True)
class Application(TermBase):
    func: Term
    args: Term

    def __post_init__(self):
        assert_isinstance(self.func.cod, Exp)
        if self.func.cod.exponent != self.args.cod:
            raise ValueError(
                f"Expected {self.func.cod.exponent}, got {self.args.cod}")

    @property
    def cod(self):
        return self.func.cod.base

    def __str__(self):
        return f"{self.func}({self.args})"

    @property
    def freevars(self, bound=None):
        return self.func.freevars + self.args.freevars

    def to_diagram(self, category=None):
        if set(self.func.freevars).intersection(self.args.freevars):
            raise NotImplementedError
        return self.func.to_diagram(category) @ self.args.to_diagram(
            category) >> Eval(self.func.cod, left=True)


@dataclass(frozen=True)
class Abstraction(TermBase):
    var: Variable
    body: Term

    @property
    def cod(self):
        return self.var.cod >> self.body.cod

    def __str__(self):
        return f"{self.var.cod}(lambda {self.var.name}: {self.body})"

    @property
    def freevars(self):
        return list(filter(lambda x: x != self.var, self.body.freevars))

    def to_diagram(self, category=None):
        i, n = self.body.freevars.index(self.var), len(self.body.freevars)
        body = self.body.to_diagram(category)
        p = body.permutation(
            [i] + [j for j in range(n) if j != i], body.dom)
        return (p >> body).curry()


@dataclass
class Substitution:
    inside: Dict[Variable, Term]

    def __call__(self, term: Term) -> Term:
        if isinstance(term, Variable):
            return self.inside.get(term, term)
        elif isinstance(term, Application):
            return self(term.func)(self(term.args))
        elif isinstance(term, Abstraction):
            other = Substitution(
                {k: v for k, v in self.inside.items() if k != term.var})
            return other(term)


@factory
class Diagram(markov.Diagram, biclosed.Diagram):
    """
    A closed diagram is both a markov and a biclosed diagram.

    A diagram applied to another post-composes their tensor with an `Eval`.
    """
    ty_factory = Ty

    @property
    def is_linear(self):
        return all(box.is_linear for box in self.boxes)


class Box(markov.Box, biclosed.Box, Diagram):
    "A closed box is a markov and biclosed box in a closed diagram."
    __ambiguous_inheritance__ = (markov.Box, biclosed.Box)

    is_linear = True


class Eval(biclosed.Eval, Box):
    "The evaluation of an exponential type."
    __ambiguous_inheritance__ = (biclosed.Eval, )
    drawing_name = "Eval"


class Coeval(biclosed.Coeval, Box):
    "The coevaluation of an exponential type, i.e. the dagger of an Eval."
    __ambiguous_inheritance__ = (biclosed.Coeval, )
    drawing_name = "$\\lambda$"


class Curry(biclosed.Curry, Box):
    "The currying of a closed diagram."
    __ambiguous_inheritance__ = (markov.Swap, )

    def to_drawing(self):
        if self.left:
            f, e = self.arg, Coeval(self.cod, left=True)
            return (f >> e).trace().to_drawing()
        f, e = self.arg, Coeval(self.cod)
        return (f >> e).trace(left=True).to_drawing()


class Swap(markov.Swap, Box):
    "Symmetric swap in a closed diagram."
    __ambiguous_inheritance__ = (markov.Swap, )


class Trace(markov.Trace, Box):
    "A trace in a closed category."
    __ambiguous_inheritance__ = (markov.Trace, )


class Copy(markov.Copy, Box):
    "A markov copy in a closed category"
    __ambiguous_inheritance__ = (markov.Copy, )

    is_linear = False


class Sum(markov.Sum, biclosed.Sum, Box):
    """
    A markov sum is a symmetric sum and a markov box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """
    __ambiguous_inheritance__ = (markov.Sum, biclosed.Sum)


class Category(markov.Category, biclosed.Category):
    """
    A Markov category is a markov category with a method :code:`curry`.

    Parameters:
        ob : The type of objects.
        ar : The type of arrows.
    """
    ob, ar = Ty, Diagram


class Functor(markov.Functor, biclosed.Functor):
    "A Markov functor is a markov functor that preserves currying."
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, (
                cat.Ob, biclosed.Eval, biclosed.Coeval, biclosed.Curry)):
            return biclosed.Functor.__call__(self, other)
        return super().__call__(other)


class Hypergraph(markov.Hypergraph):
    category, functor = Category, Functor


Diagram.hypergraph_factory = Hypergraph
Diagram.copy_factory = Copy
Diagram.braid_factory = Swap
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.trace_factory = Trace
Diagram.discard_factory = lambda X: Copy(X, 0)
Diagram.sum_factory = Sum

Id = Diagram.id
