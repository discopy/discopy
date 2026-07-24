
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
    Swap
    Trace
    Copy
    Discard
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
from typing import Dict, ClassVar

from discopy import cat, monoidal, biclosed, markov, hypergraph
from discopy.abc import ClosedCategory
from discopy.cat import factory
from discopy.drawing import Drawing


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
    @classmethod
    def from_biclosed(cls, old: biclosed.Ty) -> Ty:
        """
        Translate a biclosed type into a closed type, collapsing left and
        right exponentials into a single exponential.

        Parameters:
            old : The biclosed type to translate.

        Example
        -------
        >>> x, y = biclosed.Ty("x"), biclosed.Ty("y")
        >>> assert Ty.from_biclosed(x << y) == Ty.from_biclosed(y >> x)
        """
        return cls().tensor(*[
            cls.from_biclosed(ob.base) ** cls.from_biclosed(ob.exponent)
            if isinstance(ob, biclosed.Exp) else cls(ob.name)
            for ob in old.inside])


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

    @classmethod
    def fa(cls, left, right):
        """Forward application."""
        return cls.ev(left, right, left=True)

    @classmethod
    def ba(cls, left, right):
        """Backward application."""
        return cls.ev(right, left, left=False)

    @classmethod
    def fc(cls, left, middle, right):
        """Forward composition."""
        return (cls.id(left ** middle) @ cls.fa(middle, right)
                >> cls.fa(left, middle)).curry(
                    n=len(right), left=True)

    @classmethod
    def bc(cls, left, middle, right):
        """Backward composition."""
        return (cls.ba(left, middle) @ cls.id(middle >> right)
                >> cls.ba(middle, right)).curry(n=len(left))

    fx = fc
    bx = bc

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


class Discard(markov.Discard, Copy):
    "The discard of a closed type, i.e. a copy with zero legs."


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
        if self.cod is Drawing and isinstance(other, markov.Swap):
            return other.to_drawing()
        if isinstance(other, (
                cat.Ob, biclosed.Eval, biclosed.Coeval, biclosed.Curry,
                biclosed.TermBase)):
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
Diagram.discard_factory = Discard
Diagram.sum_factory = Sum
Ty.exp_factory = Ty.under_factory = Ty.over_factory = staticmethod(Exp)

Id = Diagram.id


class TermBase(Box, biclosed.TermBase):
    """
    A term in the internal language of a closed category.
    """
    functor = Functor.id(Diagram)

    def __call__(self, other, left=False):
        args = (other, self, left) if left else (self, other, left)
        return self.cod.application_factory(*args)

    def occurrences(self, variable: Variable) -> int:
        """Count the free occurrences of ``variable`` in the term."""
        if isinstance(self, Variable):
            return int(self == variable)
        if isinstance(self, Application):
            return self.func.occurrences(variable)\
                + self.args.occurrences(variable)
        if isinstance(self, Abstraction):
            return 0 if self.var == variable\
                else self.body.occurrences(variable)
        return 0

    @classmethod
    def from_biclosed(cls, term: biclosed.Term) -> Term:
        """
        Translate a biclosed term into a closed term, dropping planarity by
        collapsing left and right exponentials and applications.

        Parameters:
            term : The biclosed term to translate.

        Note
        ----
        This method is inherited by :class:`Constant`, :class:`Variable`,
        :class:`Application` and :class:`Abstraction`, i.e. every closed
        :class:`Term`.

        Example
        -------
        >>> X, Y = biclosed.Ty("X"), biclosed.Ty("Y")
        >>> g, x = (Y << X)("g"), X("x")
        >>> print(TermBase.from_biclosed(g(x)))
        (X >> Y)('g')(X('x'))
        """
        functor = biclosed.Functor(
            ob_map=lambda x: cls.ob(x.inside[0].name),
            ar_map=lambda c: cls.ob.constant_factory(c.name, functor(c.cod)),
            dom=biclosed.Diagram, cod=cls.functor.cod)
        return functor(term)

    def normal_form(self) -> Term:
        """
        The beta-normal form of a term, obtained by normal-order reduction.

        Raises
        ------
        ValueError
            If reduction changes the ordered free-variable context or would
            duplicate an argument in a non-cartesian category.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> f, x = (X >> Y)("f"), X("x")
        >>> assert X(lambda y: f(y))(x).normal_form() == f(x)
        """
        def normalize(term):
            if isinstance(term, Application):
                func = normalize(term.func)
                if isinstance(func, Abstraction):
                    if func.body.occurrences(func.var) > 1:
                        raise ValueError(
                            "Beta reduction would duplicate an argument.")
                    return normalize(
                        Substitution({func.var: term.args})(func.body))
                return type(term)(
                    func, normalize(term.args), term.left)
            if isinstance(term, Abstraction):
                return type(term)(
                    term.var, normalize(term.body), term.left)
            return term

        result = normalize(self)
        if result.freevars != self.freevars:
            raise ValueError(
                "Beta reduction changed the free-variable context.")
        return result


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
        self.overlap = any(
            variable in args.freevars for variable in func.freevars)
        freevars = args.freevars + func.freevars if left\
            else func.freevars + args.freevars
        self.freevars = list(dict.fromkeys(freevars))
        return self.ob().tensor(*[x.cod for x in self.freevars])

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        base, exponent = self.func.cod.base, self.func.cod.exponent
        evaluate = functor.cod.ev(
            functor(base), functor(exponent), left=not self.left)
        if context is None:
            if not self.overlap:
                func = self.func.eval(functor=functor)
                args = self.args.eval(functor=functor)
                return (args @ func if self.left else func @ args) >> evaluate
            context = Context(self.freevars)
        func = self.func.eval(functor=functor, context=context)
        args = self.args.eval(functor=functor, context=context)
        return functor.cod.copy(functor(context.dom))\
            >> (args @ func if self.left else func @ args) >> evaluate


class Abstraction(TermBase, biclosed.Abstraction):
    def __check_dom__(self):
        self.freevars = [x for x in self.body.freevars if x != self.var]
        return self.ob().tensor(*[x.cod for x in self.freevars])

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        mapped_var = functor(self.var.cod)
        n = len(mapped_var)
        if context is not None:
            inside = [self.var] + context.inside if self.left\
                else context.inside + [self.var]
            new_context = Context(inside)
            body = self.body.eval(functor=functor, context=new_context)
            return functor.cod.curry(body, n, not self.left)

        body = self.body.eval(functor=functor)
        if self.var not in self.body.freevars:
            discard = functor.cod.discard(mapped_var)
            body = discard @ body if self.left else body @ discard
            return functor.cod.curry(body, n, not self.left)

        i = self.body.freevars.index(self.var)
        start = sum(
            len(functor(variable.cod))
            for variable in self.body.freevars[:i])
        stop = start + n
        before, bound, after = (
            list(range(start)), list(range(start, stop)),
            list(range(stop, len(body.dom))))
        permutation = bound + before + after if self.left\
            else before + after + bound
        if permutation != list(range(len(body.dom))):
            permute = functor.cod.permutation(permutation, body.dom).dagger()
            body = permute >> body
        return functor.cod.curry(body, n, not self.left)


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
    The simultaneous substitution of terms for free variables.

    Attributes:
        inside : The mapping from variables to the terms substituted for them.

    Substitution is simultaneous and capture-avoiding.
    """
    inside: Dict[Variable, Term]

    def __post_init__(self):
        for variable, term in self.inside.items():
            if not isinstance(variable, Variable)\
                    or not isinstance(term, TermBase):
                raise TypeError
            if variable.cod != term.cod:
                raise ValueError(
                    f"Expected {variable.cod}, got {term.cod}")

    def __call__(self, term: Term) -> Term:
        if isinstance(term, Variable):
            return self.inside.get(term, term)
        if isinstance(term, Application):
            return type(term)(self(term.func), self(term.args), term.left)
        if isinstance(term, Abstraction):
            other = Substitution(
                {k: v for k, v in self.inside.items() if k != term.var})
            capture = any(
                key in term.body.freevars and term.var in value.freevars
                for key, value in other.inside.items())
            if not capture:
                return type(term)(term.var, other(term.body), term.left)
            var = type(term.var).fresh(
                term.var.name, term.var.cod, term.body,
                *other.inside, *other.inside.values())
            body = Substitution({term.var: var})(term.body)
            return type(term)(var, other(body), term.left)
        return term


Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction


class Equation(markov.Equation):
    """ The :class:`markov.Equation` of closed diagrams. """
