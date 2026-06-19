
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
from dataclasses import dataclass
from typing import Dict, ClassVar

from discopy import cat, monoidal, biclosed, markov
from discopy.abc import ClosedCategory
from discopy.cat import arrow_factory


@arrow_factory
class Ty(biclosed.Ty):
    """
    A closed type is a biclosed type in a symmetric category where left and
    right exponentials coincide, i.e. `X << Y == X ** Y == Y >> X`.

    Applying a closed type to a function yields an :class:`Term` e.g.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> t = X(lambda x: (X >> Y)(lambda f: f(x)))
    >>> t.draw(
    ...     path='docs/_static/closed/diagram.png',
    ...     aspect="auto", figsize=(8, 8), margins=(0.2, 0))

    .. image:: /_static/closed/diagram.png
        :align: center
    """


class Exp(biclosed.Exp):
    "An exponential object in a markov category."

    ty_factory = Ty

    def __str__(self):
        return f"({self.exponent} >> {self.base})"


@arrow_factory
class Diagram(markov.Diagram, biclosed.Diagram, ClosedCategory):
    """
    A closed diagram is both a markov and a biclosed diagram.

    A diagram applied to another post-composes their tensor with an `Eval`.
    """
    ob = Ty

    @property
    def is_linear(self):
        return all(box.is_linear for box in self.boxes)

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left: bool = True):
        return cls.eval_factory(exponent >> base, left=left)

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)


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
Ty.exp_factory = Ty.under_factory = Ty.over_factory = staticmethod(Exp)

Id = Diagram.id


class TermBase(Box, biclosed.TermBase):
    """
    A term in the internal language of a closed category.
    """
    functor = Functor.id(Diagram)

    def __call__(self, other):
        return Application(self, other, left=False)


type Term = Constant | Variable | Application | Abstraction


class Constant(TermBase, biclosed.Constant):
    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        if not context:
            return super().eval(functor)
        return functor.cod.discard(functor(context.dom)) >> super().eval(
            functor)


class Variable(TermBase, biclosed.Variable):
    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        if not context:
            return functor.cod.id(functor(self.cod))
        return functor.cod.tensor(*[
            functor.cod.id(functor(x.cod)) if x == self
            else functor.cod.discard(functor(x.cod))
            for x in context.inside])


class Application(TermBase, biclosed.Application):
    def __check_dom__(self, func, args, left):
        self.overlap = set(func.freevars).intersection(args.freevars)
        self.freevars = list(set(func.freevars + args.freevars))\
            if self.overlap else func.freevars + args.freevars
        return self.ob.tensor(*[x.cod for x in self.freevars])

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        base, exponent = self.func.cod.base, self.func.cod.exponent
        evaluate = functor.cod.ev(functor(base), functor(exponent))
        if context is None:
            if not self.overlap:
                func = self.func.eval(functor=functor)
                args = self.args.eval(functor=functor)
                return func @ args >> evaluate
            context = Context(self.freevars)
        func = self.func.eval(functor=functor, context=context)
        args = self.args.eval(functor=functor, context=context)
        return functor.cod.copy(functor(context.dom))\
            >> func @ args >> evaluate


class Abstraction(TermBase, biclosed.Abstraction):
    def __check_dom__(self):
        self.freevars = [x for x in self.body.freevars if x != self.var]
        return self.ob().tensor(*[x.cod for x in self.freevars])

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        if context:
            new_context = Context([self.var] + context.inside)
            body = self.body.eval(functor=functor, context=new_context)
            return body.curry(left=True)
        i, n = self.body.freevars.index(self.var), len(self.body.freevars)
        body = self.body.eval(functor=functor)
        p = [0] + [j + 1 if j < i else j for j in range(n) if j != i]
        return (body.permutation(p, body.dom).dagger() >> body).curry()


@dataclass
class Context:
    inside: list[Variable]
    category: ClassVar[type[ClosedCategory]] = Diagram

    @property
    def dom(self):
        return self.category.ob.tensor(*[x.cod for x in self.inside])


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
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction
