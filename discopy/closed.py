
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
    Curry
    Sum
    Category
    Functor

Axioms
------

:meth:`Diagram.curry` and :meth:`Diagram.uncurry` are inverses.

>>> x, y, z = map(Ty, "xyz")
>>> f, g, h = Box('f', x, z << y), Box('g', x @ y, z), Box('h', y, x >> z)

>>> from discopy.drawing import Equation
>>> Equation(f.uncurry().curry(), f).draw(
...     path='docs/_static/biclosed/curry-left.png', margins=(0.1, 0.05))

.. image:: /_static/biclosed/curry-left.png
    :align: center

>>> Equation(h.uncurry(left=False).curry(left=False), h).draw(
...     path='docs/_static/biclosed/curry-right.png', margins=(0.1, 0.05))

.. image:: /_static/biclosed/curry-right.png
    :align: center

>>> Equation(
...     g.curry().uncurry(), g, g.curry(left=False).uncurry(left=False)).draw(
...         path='docs/_static/biclosed/uncurry.png')

.. image:: /_static/biclosed/uncurry.png
    :align: center
"""

from __future__ import annotations
from dataclasses import dataclass
from inspect import signature

from discopy import biclosed, markov
from discopy.cat import Category, factory
from discopy.utils import assert_isinstance


@factory
class Ty(biclosed.Ty):
    """
    A closed type is a biclosed type in a symmetric category where left and
    right exponentials coincide, i.e. `X << Y == X ** Y == Y >> X`.
    Applying a type to an anonymous function yields a diagram e.g.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> f = X(lambda x: (X >> Y)(lambda y: y(x)))
    >>> print(f)
    X(lambda x: (X >> Y)(lambda y: y(x)))
    """
    def __pow__(self, other: Ty) -> Ty:
        return Exp(self, other) if isinstance(other, Ty)\
            else super().__pow__(other)

    def __lshift__(self, other):
        return Exp(self, other)

    def __rshift__(self, other):
        return Exp(other, self)

    def __call__(self, func):
        varnames = list(signature(func).parameters.keys())
        if len(varnames) != 1:
            raise NotImplementedError
        var = Variable(self, varnames[0])
        return Abstraction(var, func(var))


class Exp(Ty, biclosed.Exp):
    "An exponential object in a markov category."
    __ambiguous_inheritance__ = (biclosed.Exp, )

    def __str__(self):
        return f"({self.exponent} >> {self.base})"


class TermBase:
    cod: Ty

    def __call__(self, other):
        return Application(self, other)


type Term = Variable | Application | Abstraction


@dataclass
class Variable(TermBase):
    cod: Ty
    name: str

    def __str__(self):
        return self.name


@dataclass
class Application(TermBase):
    func: Term
    args: Term

    def __init__(self, func, args):
        assert_isinstance(func.cod, Exp)
        assert func.cod.exponent == args.cod
        self.cod, self.func, self.args = func.cod.base, func, args

    def __str__(self):
        return f"{self.func}({self.args})"


@dataclass
class Abstraction(TermBase):
    var: Variable
    body: Term

    def __init__(self, var, body):
        self.cod = var.cod >> body.cod
        self.var, self.body = var, body

    def __str__(self):
        return f"{self.var.cod}(lambda {self.var.name}: {self.body})"


@factory
class Diagram(markov.Diagram, biclosed.Diagram):
    """
    A closed diagram is both a markov and a biclosed diagram.

    A diagram applied to another post-composes their tensor with an `Eval`.
    """
    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left=True) -> Eval:
        return cls.eval_factory(base << exponent, left)


class Box(markov.Box, biclosed.Box, Diagram):
    "A closed box is a markov and biclosed box in a closed diagram."
    __ambiguous_inheritance__ = (markov.Box, biclosed.Box)


class Eval(biclosed.Eval, Box):
    "The evaluation of an exponential type."
    __ambiguous_inheritance__ = (biclosed.Eval, )

    def __init__(self, x: Exp, left=True):
        self.base, self.exponent, self.left = x.base, x.exponent, left
        dom = x @ self.exponent if left else self.exponent @ x
        Box.__init__(self, "Eval" + str(x), dom, self.base)


class Curry(biclosed.Curry, Box):
    "The currying of a closed diagram."
    __ambiguous_inheritance__ = (markov.Swap, )


class Swap(markov.Swap, Box):
    "Symmetric swap in a closed diagram."
    __ambiguous_inheritance__ = (markov.Swap, )


class Trace(markov.Trace, Box):
    "A trace in a closed category."
    __ambiguous_inheritance__ = (markov.Trace, )


class Copy(markov.Copy, Box):
    "A markov copy in a closed category"
    __ambiguous_inheritance__ = (markov.Copy, )


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
        if isinstance(other, Eval):
            return self.cod.ar.copy(self(other.dom), len(other.cod))
        if isinstance(other, markov.Merge):
            return self.cod.ar.merge(self(other.cod), len(other.dom))
        return super().__call__(other)


class Hypergraph(markov.Hypergraph):
    category, functor = Category, Functor


Diagram.hypergraph_factory = Hypergraph
Diagram.copy_factory = Copy
Diagram.braid_factory = Swap
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.trace_factory = Trace
Diagram.discard_factory = lambda X: Copy(X, 0)
Diagram.sum_factory = Sum

Id = Diagram.id
