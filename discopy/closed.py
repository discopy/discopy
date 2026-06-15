
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
    TermBase
    Constant
    Variable
    Application
    Abstraction
    Diagram
    Box
    Eval
    Coeval
    Curry
    Sum
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
from abc import abstractproperty
from dataclasses import dataclass
from typing import Dict, ClassVar

from discopy import cat, biclosed, markov
from discopy.abc import ClosedCategory
from discopy.cat import ob_factory, ar_factory
from discopy.utils import assert_isinstance


@ob_factory
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
    """
    def __pow__(self, other: Ty) -> Ty:
        return Exp(self, other) if isinstance(other, Ty)\
            else super().__pow__(other)

    def __lshift__(self, other):
        return Exp(self, other)

    def __rshift__(self, other):
        return Exp(other, self)


class Exp(Ty, biclosed.Exp):
    "An exponential object in a markov category."

    def __str__(self):
        return f"({self.exponent} >> {self.base})"


@ar_factory
class Diagram(markov.Diagram, biclosed.Diagram, ClosedCategory):
    """
    A closed diagram is both a markov and a biclosed diagram.

    A diagram applied to another post-composes their tensor with an `Eval`.
    """
    ob = Ty

    @property
    def is_linear(self):
        return all(box.is_linear for box in self.boxes)


class Box(markov.Box, biclosed.Box, Diagram):
    "A closed box is a markov and biclosed box in a closed diagram."

    is_linear = True


class Eval(biclosed.Eval, Box):
    "The evaluation of an exponential type."
    drawing_name = "__call__"


class Coeval(biclosed.Coeval, Box):
    "The coevaluation of an exponential type, i.e. the dagger of an Eval."


class Curry(biclosed.Curry, Box):
    "The currying of a closed diagram."

    def to_drawing(self):
        if self.left:
            f, e = self.arg, Coeval(self.cod, left=True)
            return (f >> e).trace().to_drawing()
        f, e = self.arg, Coeval(self.cod)
        return (f >> e).trace(left=True).to_drawing()


class Swap(markov.Swap, Box):
    "Symmetric swap in a closed diagram."


class Trace(markov.Trace, Box):
    "A trace in a closed category."


class Copy(markov.Copy, Box):
    "A markov copy in a closed category"

    is_linear = False


class Sum(markov.Sum, biclosed.Sum, Box):
    """
    A markov sum is a symmetric sum and a markov box.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


Diagram.over, Diagram.under, Diagram.exp\
    = map(staticmethod, (biclosed.Over, biclosed.Under, Exp))
Diagram.sum_factory = Sum

Id = Diagram.id


class Functor(biclosed.Functor, markov.Functor):
    """
    A closed functor is a markov functor
    that preserves evaluation and currying.

    Parameters:
        ob (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, (
                cat.Ob, biclosed.Eval, biclosed.Coeval, biclosed.Curry)):
            return biclosed.Functor.__call__(self, other)
        return super().__call__(other)


class Hypergraph(markov.Hypergraph):
    functor = Functor


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


class TermBase(biclosed.TermBase):
    """
    A term in the internal language of a closed category.
    """
    def __call__(self, other):
        return Application(self, other)

    def __lshift__(self, other):
        return Application(self, other)

    def __rshift__(self, other):
        return Application(other, self)

    @abstractproperty
    def freevars(self) -> list[Variable]: ...


type Term = Constant | Variable | Application | Abstraction


class Constant(biclosed.Constant, TermBase):
    def to_diagram(self, category=Diagram, context=None):
        if not context:
            return super().to_diagram(category)
        d = category.discard()
        return d >> super().to_diagram(category)


class Variable(biclosed.Variable, TermBase):
    typ: Ty
    name: str

    def to_diagram(self, category=Diagram, context=None):
        if not context:
            return category.id(self.typ)
        return category.tensor(*[
            category.id(x) if x == self else category.discard(x)
            for x in context])


@dataclass(frozen=True)
class Application(biclosed.Application, TermBase):
    func: Term
    args: Term

    def __post_init__(self):
        assert_isinstance(self.func.typ, Exp)
        if self.func.typ.exponent != self.args.typ:
            raise ValueError(
                f"Expected {self.func.typ.exponent}, got {self.args.typ}")

    @property
    def typ(self):
        return self.func.typ.base

    def __str__(self):
        return f"{self.func}({self.args})"

    def to_diagram(self, category=Diagram, context=None):
        evaluate = Eval(self.func.typ, left=True)
        if context is None:
            overlap = set(self.func.freevars).intersection(self.args.freevars)
            if not overlap:
                func = self.func.to_diagram(category)
                args = self.args.to_diagram(category)
                return func @ args >> evaluate
            context = Context(list(set(self.freevars)), category)
        func = self.func.to_diagram(category=category, context=context)
        args = self.args.to_diagram(category=category, context=context)
        return category.copy(context.dom) >> func @ args >> evaluate


@dataclass(frozen=True)
class Abstraction(biclosed.Abstraction, TermBase):
    var: Variable
    body: Term

    def __post_init__(self):
        pass  # No need to check for planarity or linearity.

    @property
    def typ(self):
        return self.var.typ >> self.body.typ

    def __str__(self):
        return f"{self.var.typ}(lambda {self.var.name}: {self.body})"

    def to_diagram(self, category=Diagram, context=None):
        if context:
            new_context = self.var + context.inside
            body = self.body.to_diagram(category=category, context=new_context)
            return body.curry()
        i, n = self.body.freevars.index(self.var), len(self.body.freevars)
        body = self.body.to_diagram(category)
        p = body.permutation(
            [i] + [j for j in range(n) if j != i], body.dom)
        return (p >> body).curry()


@dataclass
class Context:
    inside: list[Variable]
    category: ClassVar[type[ClosedCategory]] = Diagram

    @property
    def dom(self):
        return self.category.ob.tensor(*[x.typ for x in self.inside])


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


Ty.variable_factory = Variable
Ty.abstraction_factory = Abstraction
