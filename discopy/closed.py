
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
    Substitution
    BohmTree
    Strategy
    LeftmostOutermost
    Diagram
    Box
    Eval
    Coeval
    Curry
    Sum
    Functor
    CMap

Axioms
------

:meth:`Diagram.curry` and :meth:`Diagram.uncurry` are inverses.

>>> x, y, z = map(Ty, "xyz")
>>> f, g = Box('f', x, z << y), Box('g', x @ y, z)

>>> Equation(f.uncurry().curry(), f).draw(
...     path='docs/_static/closed/curry-left.svg', margins=(0.1, 0.05))

.. image:: /_static/closed/curry-left.svg
    :align: center

>>> Equation(g.curry().uncurry(), g).draw(
...     path='docs/_static/closed/uncurry.svg')

.. image:: /_static/closed/uncurry.svg
    :align: center
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, ClassVar, Optional

from discopy import cat, monoidal, biclosed, markov, hypergraph
from discopy.abc import ClosedCategory
from discopy.cat import factory
from discopy.utils import assert_isinstance, factory_name


@factory
class Ty(biclosed.Ty):
    """
    A closed type is a biclosed type in a symmetric category where left and
    right exponentials coincide, i.e. `X << Y == X ** Y == Y >> X`.

    Applying a closed type to a function yields an :class:`Term` e.g.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> t = X(lambda x: (X >> Y)(lambda f: f(x)))
    >>> t.draw(
    ...     path='docs/_static/closed/diagram.svg',
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


class CMap(biclosed.CMap):
    category = Diagram
    require_planar = False


Diagram.functor_factory = Functor
Diagram.map_factory = CMap
Hypergraph = hypergraph.Hypergraph[Diagram]
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

    def reduce(self, budget: int = None, strategy: type = None
               ) -> Optional[BohmTree]:
        """
        The :class:`BohmTree` of the term, contracting at most ``budget``
        beta redexes, or ``None`` when the budget runs out before a head
        normal form is reached.

        Parameters:
            budget : The number of beta redexes to contract, no bound if
                ``None``.
            strategy : The subclass of :class:`Strategy` to follow,
                :class:`LeftmostOutermost` by default.

        Example
        -------
        >>> X = Ty('X')
        >>> f, x = Variable('f', X >> X), Variable('x', X)
        >>> two = (X >> X)(lambda f: X(lambda x: f(f(x))))
        >>> assert two.reduce() == BohmTree(X, (f, x), 0, (
        ...     BohmTree(X, (f, x), 0, (BohmTree(X, (f, x), 1, ()), )), ))
        >>> assert two(X(lambda x: x))(x).reduce(budget=1) is None
        """
        return (strategy or self.strategy_factory)(budget)(
            self, tuple(self.freevars))

    def normal_form(self, budget: int = None, strategy: type = None) -> Term:
        """
        The beta normal form of the term, i.e. the term of its Böhm tree,
        so that normalisation is idempotent.

        Parameters:
            budget : The number of beta redexes to contract, no bound if
                ``None``.
            strategy : The subclass of :class:`Strategy` to follow,
                :class:`LeftmostOutermost` by default.

        Example
        -------
        >>> X = Ty('X')
        >>> x = Variable('x', X)
        >>> term = X(lambda x: x)(x)
        >>> assert term.normal_form() == x == x.normal_form()
        >>> term.normal_form(budget=0)
        Traceback (most recent call last):
            ...
        ValueError: The budget of 0 beta redexes ran out reducing \
X(lambda x: x)(x).
        """
        tree = self.reduce(budget, strategy)
        if tree is None:
            raise ValueError(
                f"The budget of {budget} beta redexes ran out "
                f"reducing {self}.")
        return tree.to_term(len(self.freevars))


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
        return self.ob().tensor(*[x.cod for x in self.freevars])

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
    """
    The capture-avoiding substitution of terms for free variables.

    Parameters:
        inside : The mapping from variables to terms of the same type.

    A bound variable is renamed, appending primes to its name, only when a
    free variable of a substituted term would be captured.

    Example
    -------
    >>> X = Ty('X')
    >>> f = Variable('f', X >> X)
    >>> x, y = Variable('x', X), Variable('y', X)
    >>> print(Substitution({f: X(lambda y: y)})(X(lambda x: f(x))))
    X(lambda x: X(lambda y: y)(x))
    >>> g = Variable('g', X >> (X >> X))
    >>> print(Substitution({f: g(y)})(X(lambda y: f(y))))
    X(lambda y': g(y)(y'))
    """
    inside: Dict[Variable, Term]

    def __call__(self, term: Term) -> Term:
        if isinstance(term, Variable):
            return self.inside.get(term, term)
        if isinstance(term, Application):
            return type(term)(self(term.func), self(term.args), term.left)
        if isinstance(term, Abstraction):
            return self.abstract(term)
        return term

    def abstract(self, term: Abstraction) -> Abstraction:
        """
        Substitute under the binder of an abstraction, renaming it when a
        free variable of a substituted term would be captured.

        Parameters:
            term : The abstraction to substitute inside of.
        """
        inside = {x: v for x, v in self.inside.items()
                  if x != term.var and x != v and x in term.body.freevars}
        if not inside:
            return term
        var, body = term.var, term.body
        freevars = [x for v in inside.values() for x in v.freevars]
        if var in freevars:
            var = self.fresh(var, freevars + body.freevars)
            body = type(self)({term.var: var})(body)
        return type(term)(var, type(self)(inside)(body), term.left)

    def fresh(self, var: Variable, avoid: list[Variable]) -> Variable:
        """
        Rename a variable, appending primes until it avoids a list of
        variables.

        Parameters:
            var : The variable to rename.
            avoid : The variables whose names to avoid.
        """
        names, name = {x.name for x in avoid}, var.name
        while name in names:
            name += "'"
        return type(var)(name, var.cod)


@dataclass
class BohmTree:
    """
    A `Böhm tree <https://en.wikipedia.org/wiki/B%C3%B6hm_tree>`_ is the
    normal form of a :class:`Term`: each node abstracts the variables in
    scope beyond those of its parent, then applies a head variable to the
    trees of its arguments, with ``None`` for the arguments left unreduced
    when the budget of a :class:`Strategy` runs out.

    Parameters:
        cod : The type of the head variable applied to the arguments.
        variables : The variables in scope, those of the parent followed by
            the ones bound at this node.
        head : The index in ``variables`` of the head variable.
        args : The trees of the arguments of the head variable.

    Example
    -------
    >>> from discopy import closed
    >>> X = Ty('X')
    >>> f, x = Variable('f', X >> X), Variable('x', X)
    >>> tree = (X >> X)(lambda f: X(lambda x: f(x))).reduce()
    >>> assert tree == BohmTree(X, (f, x), 0, (BohmTree(X, (f, x), 1, ()), ))
    >>> assert tree == eval(repr(tree))
    >>> print(tree.to_term())
    (X >> X)(lambda f: X(lambda x: f(x)))
    """
    cod: Ty
    variables: tuple[Variable, ...]
    head: int
    args: tuple[Optional[BohmTree], ...]

    def __post_init__(self):
        assert 0 <= self.head < len(self.variables)
        cod = self.variables[self.head].cod
        for arg in self.args:
            assert cod.is_exp
            assert arg is None or arg.ty(len(self.variables)) == cod.exponent
            cod = cod.base
        assert cod == self.cod

    def ty(self, n: int = 0) -> Ty:
        """
        The type of the term of the tree, abstracting over ``variables[n:]``
        for ``n`` the number of variables of the parent.

        Parameters:
            n : The number of variables bound by the parent.
        """
        result = self.cod
        for var in reversed(self.variables[n:]):
            result = var.cod >> result
        return result

    def to_term(self, n: int = 0) -> Term:
        """
        The term of the tree in the standard syntax, abstracting over
        ``variables[n:]`` for ``n`` the number of variables of the parent.

        Parameters:
            n : The number of variables bound by the parent.
        """
        if None in self.args:
            raise ValueError(f"Missing arguments in {self}.")
        result = self.variables[self.head]
        for arg in self.args:
            result = result(arg.to_term(len(self.variables)))
        for var in reversed(self.variables[n:]):
            result = var.cod.abstraction_factory(var, result)
        return result

    def __repr__(self):
        return factory_name(type(self))\
            + f"({self.cod!r}, {self.variables!r}, {self.head}, {self.args!r})"


@dataclass
class Strategy:
    """
    A reduction strategy computes the :class:`BohmTree` of a term within a
    ``budget`` of beta redexes to contract, with no bound when ``None``.

    Contracting the head redex comes first in every strategy that reaches a
    head normal form; concrete strategies such as :class:`LeftmostOutermost`
    choose the order in which the arguments of the head variable are reduced
    by implementing :meth:`Strategy.arguments`.

    Parameters:
        budget : The number of beta redexes left to contract.
    """
    budget: Optional[int] = None

    def __call__(self, term: Term, variables=()) -> Optional[BohmTree]:
        """
        The Böhm tree of a term in a scope of variables, or ``None`` when
        the budget runs out before a head normal form is reached.

        Parameters:
            term : The term to reduce.
            variables : The variables in scope, i.e. the free variables of
                the term followed by the ones bound by its parents.
        """
        variables, spine = list(variables), []
        while isinstance(term, (Application, Abstraction)):
            if isinstance(term, Application):
                spine.append(term.args)
                term = term.func
            elif spine:
                if not self.spend():
                    return None
                term = Substitution({term.var: spine.pop()})(term.body)
            else:
                variables.append(term.var)
                term = term.body
        assert_isinstance(term, Variable)
        head = len(variables) - 1 - variables[::-1].index(term)
        cod = term.cod
        for _ in spine:
            cod = cod.base
        args = self.arguments(tuple(reversed(spine)), tuple(variables))
        return BohmTree(cod, tuple(variables), head, args)

    def spend(self) -> bool:
        "Take one beta redex from the budget, or return ``False``."
        if self.budget == 0:
            return False
        self.budget = None if self.budget is None else self.budget - 1
        return True

    def arguments(self, terms: tuple[Term, ...],
                  variables: tuple[Variable, ...]
                  ) -> tuple[Optional[BohmTree], ...]:
        """
        The trees of the arguments of a head variable, reduced in the order
        chosen by the concrete strategy.

        Parameters:
            terms : The arguments of the head variable.
            variables : The variables in scope.
        """
        raise NotImplementedError


class LeftmostOutermost(Strategy):
    """
    The default :class:`Strategy`: contract the head redex, then reduce the
    arguments of the head variable from left to right.

    Example
    -------
    >>> X = Ty('X')
    >>> f, x = Variable('f', X >> X), Variable('x', X)
    >>> term = (X >> X)(lambda f: X(lambda x: f(X(lambda x: x)(x))))
    >>> assert LeftmostOutermost(0)(term)\\
    ...     == BohmTree(X, (f, x), 0, (None, ))
    >>> assert LeftmostOutermost(1)(term) == term.reduce() == BohmTree(
    ...     X, (f, x), 0, (BohmTree(X, (f, x), 1, ()), ))
    """
    def arguments(self, terms, variables):
        return tuple(self(term, variables) for term in terms)


Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction
TermBase.strategy_factory = LeftmostOutermost


class Equation(markov.Equation):
    """ The :class:`markov.Equation` of closed diagrams. """
