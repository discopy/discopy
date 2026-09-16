
"""
The free closed category, i.e. symmetric diagrams with exponentials,
whose terms are the linear lambda calculus: a variable is used exactly
once, see :mod:`discopy.markov` for terms with copy and discard.

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
    Permutation
    Swap
    Sum
    Bubble
    Functor
    CMap

Axioms
------

:meth:`Diagram.curry` and :meth:`Diagram.uncurry` are inverses.

>>> x, y, z = map(Ty, "xyz")
>>> f, g = Box('f', x, z << y), Box('g', x @ y, z)

>>> Equation(f.uncurry().curry(), f).draw(
...     doctest='docs/_static/closed/curry-left.svg', margins=(0.1, 0.05))

.. image:: /_static/closed/curry-left.svg
    :align: center

>>> Equation(g.curry().uncurry(), g).draw(
...     doctest='docs/_static/closed/uncurry.svg')

.. image:: /_static/closed/uncurry.svg
    :align: center
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from discopy import (
    cat, monoidal, biclosed, symmetric, cmap, hypergraph)
from discopy.abc import ClosedCategory
from discopy.cat import factory, Generator


@factory
class Ty(biclosed.Ty):
    """
    A closed type is a biclosed type in a symmetric category where left and
    right exponentials coincide, i.e. `X << Y == X ** Y == Y >> X`.

    Applying a closed type to a function yields an :class:`Term` e.g.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> t = X(lambda x: (X >> Y)(lambda f: f(x)))
    >>> t.draw(
    ...     doctest='docs/_static/closed/diagram.svg',
    ...     aspect="auto", figsize=(8, 8), margins=(0.2, 0))

    .. image:: /_static/closed/diagram.svg
        :align: center
    """


class Exp(biclosed.Exp):
    "An exponential object in a markov category."

    ob = Ty

    def __str__(self):
        return f"({self.exponent} >> {self.base})"


@factory
class Diagram(symmetric.Diagram, biclosed.Diagram, ClosedCategory):
    """
    A closed diagram is both a symmetric and a biclosed diagram: it is
    fully linear, with no copy nor discard, see :mod:`discopy.markov`.

    A diagram applied to another post-composes their tensor with an `Eval`.
    """
    ob = Ty

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left: bool = True):
        return cls.eval_factory(exponent >> base, left=left)

    def to_compact(self) -> "cmap.CMap":
        """
        Collapse the curry bubbles down to wiring structure, i.e. the
        combinatorial map where each curry becomes its argument followed
        by :class:`Coeval` with the curried wires fed back. A closed
        category is not traced, so the result is a :class:`cmap.CMap`,
        which is compact whatever category hosts it — the diagrams keep
        their currying as a bubble, and only collapse when read through
        the geometry of interaction.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> assert f.curry().to_compact() == f.curry().to_map().to_compact()
        """
        return self.to_map().to_compact()

    def to_drawing(self):
        return monoidal.Diagram.to_drawing(self, functor_factory=Functor)

    @Generator()
    def eval_factory(cls):
        return Eval


Box = Diagram.generator_factory


class Eval(biclosed.Eval, Box):
    "The evaluation of an exponential type."
    drawing_name = "__call__"


Coeval, Curry, Permutation, Swap, Sum, Bubble = (
    Diagram.coeval_factory, Diagram.curry_factory,
    Diagram.permutation_factory, Diagram.swap_factory,
    Diagram.sum_factory, Diagram.bubble_factory)


class Functor(biclosed.Functor, symmetric.Functor):
    """
    A closed functor is a symmetric functor
    that preserves evaluation and currying.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, (
                cat.Ob, biclosed.Eval, biclosed.Coeval, biclosed.Curry)):
            return biclosed.Functor.__call__(self, other)
        return super().__call__(other)


CMap = cmap.CMap[Diagram]


Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Ty.exp_factory = Ty.under_factory = Ty.over_factory = staticmethod(Exp)

Id = Diagram.id


class TermBase(Box, biclosed.TermBase):
    """
    A term in the internal language of a closed category, i.e. the linear
    lambda calculus: an application shares no free variable between its
    function and its arguments, an abstracted variable occurs exactly once
    in the body, at any position since the category is symmetric. See
    :class:`markov.TermBase` for terms with copy and discard.
    """
    functor = Functor.id(Diagram)

    def __call__(self, other):
        return Application(self, other, left=False)


type Term = Constant | Variable | Application | Abstraction


class Constant(TermBase, biclosed.Constant):
    "A constant term in a closed category."


class Variable(TermBase, biclosed.Variable):
    "A variable in a closed category, used exactly once."


class Application(TermBase, biclosed.Application):
    """
    A linear application: the free variables of the function and of the
    arguments are disjoint — a shared variable needs the copy of
    :class:`markov.Application`.
    """
    def __check_dom__(self, func, args, left):
        if set(func.freevars).intersection(args.freevars):
            raise ValueError("Expected disjoint free variables.")
        self.freevars = args.freevars + func.freevars if left\
            else func.freevars + args.freevars
        return self.ob().tensor(*[x.cod for x in self.freevars])


class Abstraction(TermBase, biclosed.Abstraction):
    """
    A linear abstraction: the variable occurs exactly once in the body,
    at any position — the symmetry permutes it into place.
    """
    def __check_dom__(self):
        if self.body.freevars.count(self.var) != 1:
            raise ValueError("Expected variable to occur exactly once.")
        self.freevars = [x for x in self.body.freevars if x != self.var]
        return self.ob().tensor(*[x.cod for x in self.freevars])

    def eval(self, functor=None):
        functor = functor or self.functor
        if self.left:
            return type(self)(self.var, self.body).eval(functor)
        body = self.body.eval(functor=functor)
        i, n = self.body.freevars.index(self.var), len(self.body.freevars)
        p = [i] + [j for j in range(n) if j != i]
        doms = [self.ob(wire) for wire in body.dom.inside]
        return (body.permutation(p, doms).dagger() >> body).curry(left=False)


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


class Equation(symmetric.Equation):
    """ The :class:`symmetric.Equation` of closed diagrams. """
