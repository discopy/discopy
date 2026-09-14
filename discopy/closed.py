
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
    Diagram
    Box
    Eval
    Coeval
    Curry
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
from functools import reduce
from typing import Dict

from discopy import monoidal, biclosed, markov, cmap, hypergraph
from discopy.abc import ClosedCategory
from discopy.cat import factory


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

    def to_compact(self) -> Diagram:
        """
        Open the curry bubbles into coevaluation and feedback, which stays
        a :class:`Diagram` as a closed category is traced: each curry
        becomes its argument followed by :class:`Coeval`, traced over the
        curried wires, and each term is evaluated first.

        Example
        -------
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> assert f.curry().to_compact() == (
        ...     f >> Coeval(z << y, left=True)).trace()
        """
        def image(box):
            if isinstance(box, Curry):
                return (box.arg.to_compact() >> Coeval(
                    box.cod, left=box.left)).trace(
                        len(box.cod.exponent), left=not box.left)
            if isinstance(box, (Application, Abstraction)):
                return box.eval(Functor.id(Diagram)).to_compact()
            return box
        result = self.id(self.dom)
        for layer in self.inside:
            for box, offset in layer.boxes_and_offsets:
                cod = result.cod
                result >>= cod[:offset] @ image(box)\
                    @ cod[offset + len(box.dom):]
        return result

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
    "The currying of a closed diagram, linear when its argument is."

    @property
    def is_linear(self):
        return self.arg.is_linear


class Permutation(markov.Permutation, Box):
    "A permutation in a closed diagram."


class Swap(Permutation, markov.Swap, Box):
    "Symmetric swap in a closed diagram."


class Trace(markov.Trace, Box):
    "A trace in a closed category, linear when its argument is."

    @property
    def is_linear(self):
        return self.arg.is_linear


class Copy(markov.Copy, Box):
    "A markov copy in a closed category"

    is_linear = False


class Discard(markov.Discard, Copy):
    "A markov discard in a closed category."


class Sum(markov.Sum, biclosed.Sum, Box):
    """
    A markov sum is a symmetric sum and a markov box,
    linear when every term is.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """
    @property
    def is_linear(self):
        return all(term.is_linear for term in self.terms)


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


CMap = cmap.CMap[Diagram]


Diagram.functor_factory = Functor
Hypergraph = hypergraph.Hypergraph[Diagram]
Diagram.copy_factory = Copy
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
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
    A term in the internal language of a closed category, i.e. a lambda term
    which need not be linear: a variable may occur any number of times, since
    a closed category is markov it can be copied and discarded.

    A term is evaluated in a context, a list of distinct variables containing
    its free ones: the variables that do not occur in the term are discarded,
    the others are permuted into the order of its :attr:`freevars`, see
    :meth:`weaken`. The context is :attr:`freevars` itself by default.

    Note
    ----
    Closed terms accept the ``left`` argument of their biclosed counterparts
    and ignore it: a closed category has one exponential, so an application
    ``x(f, left=True)`` is ``f(x)`` and an abstraction on the left is one on
    the right.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> f, x = (X >> Y)("f"), X("x")
    >>> assert x(f, left=True) == f(x)
    """
    functor = Functor.id(Diagram)

    def __call__(self, other, left=False):
        args = (other, self) if left else (self, other)
        return self.cod.application_factory(*args)

    def weaken(self, functor: Functor, context=None) -> Diagram:
        """
        The structural morphism from the image of a context to that of the
        free variables of the term: it discards the variables that do not
        occur in the term and permutes the others into the order of
        :attr:`freevars`.

        Parameters:
            functor : The functor to evaluate the types.
            context : A list of distinct variables containing the free ones,
                the free variables themselves by default.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> x, y = Variable("x", X), Variable("y", Y)
        >>> assert x.weaken(x.functor, [y, x])\\
        ...     == Diagram.swap(Y, X) >> X @ Diagram.discard(Y)
        """
        context = self.freevars if context is None else list(context)
        if context == self.freevars:
            return functor.cod.id(functor(self.dom))
        unused = [x for x in context if x not in self.freevars]
        permutation = functor.cod.permutation(
            [context.index(x) for x in self.freevars + unused],
            [functor(x.cod) for x in context])
        if not unused:
            return permutation
        discard = functor.cod.discard(functor(
            self.ob().tensor(*[x.cod for x in unused])))
        return permutation >> functor.cod.id(functor(self.dom)) @ discard

    def occurrences(self, variable: Variable) -> int:
        "The number of free occurrences of a variable in the term."
        return 0

    def substitute(self, substitution: Substitution) -> Term:
        "The term with the free variables of a substitution replaced."
        return self

    def normal_form(self) -> Term:
        """
        The beta-normal form of a term, obtained by normal-order reduction.

        Reduction may discard a free variable, since discarding is natural
        in a markov category, but never copies an argument, since copying is
        not: a redex whose variable occurs more than once in the body raises
        ``ValueError``.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> f, x = (X >> Y)("f"), X("x")
        >>> assert X(lambda y: f(y))(x).normal_form() == f(x)
        """
        return self

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


type Term = Constant | Variable | Application | Abstraction


class Constant(TermBase, biclosed.Constant):
    "A constant term, evaluated in a context by discarding it."
    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        return self.weaken(functor, context)\
            >> biclosed.Constant.eval(self, functor)


class Variable(TermBase, biclosed.Variable):
    "A variable, evaluated in a context by discarding the other variables."
    def eval(self, functor=None, context=None):
        return self.weaken(functor or self.functor, context)

    def occurrences(self, variable):
        return int(self == variable)

    def substitute(self, substitution):
        return substitution.inside.get(self, self)


class Application(TermBase, biclosed.Application):
    """
    The application ``func(args)`` of a term to another.

    Attributes:
        overlap : The variables free in both ``func`` and ``args``, which
            :meth:`eval` copies.
    """
    def __init__(self, func: Term, args: Term, left: bool = False):
        biclosed.Application.__init__(self, func, args)

    def __check_dom__(self, func, args, left):
        self.overlap = [x for x in func.freevars if x in args.freevars]
        self.freevars = list(dict.fromkeys(func.freevars + args.freevars))
        return self.ob().tensor(*[x.cod for x in self.freevars])

    @property
    def is_linear(self):
        return not self.overlap\
            and self.func.is_linear and self.args.is_linear

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        func, args = self.func, self.args
        source = [(x, i) for x in self.freevars
                  for i in range(2 if x in self.overlap else 1)]
        target = [(x, 0) for x in func.freevars]\
            + [(x, int(x in self.overlap)) for x in args.freevars]
        copy = reduce(lambda left, right: left @ right, [
            functor.cod.copy(functor(x.cod)) if x in self.overlap
            else functor.cod.id(functor(x.cod)) for x in self.freevars],
            functor.cod.id(functor.cod.ob()))
        permutation = functor.cod.permutation(
            [source.index(pair) for pair in target],
            [functor(x.cod) for x, _ in source])
        evaluate = functor.cod.ev(
            functor(func.cod.base), functor(func.cod.exponent))
        return self.weaken(functor, context) >> copy >> permutation\
            >> func.eval(functor) @ args.eval(functor) >> evaluate

    def occurrences(self, variable):
        return self.func.occurrences(variable)\
            + self.args.occurrences(variable)

    def substitute(self, substitution):
        return type(self)(
            self.func.substitute(substitution),
            self.args.substitute(substitution))

    def normal_form(self):
        func, args = self.func.normal_form(), self.args.normal_form()
        if not isinstance(func, Abstraction):
            return type(self)(func, args)
        if func.body.occurrences(func.var) > 1:
            raise ValueError(f"{self} copies its argument {args}.")
        return Substitution({func.var: args})(func.body).normal_form()


class Abstraction(TermBase, biclosed.Abstraction):
    """
    The abstraction ``var.cod(lambda var: body)`` of a variable in a term,
    which need not occur in it or may occur several times.
    """
    def __init__(self, var: Variable, body: Term, left: bool = False):
        biclosed.Abstraction.__init__(self, var, body)

    def __check_dom__(self):
        self.freevars = [x for x in self.body.freevars if x != self.var]
        return self.ob().tensor(*[x.cod for x in self.freevars])

    @property
    def is_linear(self):
        return self.body.is_linear and self.body.occurrences(self.var) == 1

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        body = self.body.eval(functor, [self.var] + self.freevars)
        return self.weaken(functor, context)\
            >> body.curry(len(functor(self.var.cod)), left=False)

    def occurrences(self, variable):
        return 0 if variable == self.var else self.body.occurrences(variable)

    def substitute(self, substitution):
        inside = {key: value for key, value in substitution.inside.items()
                  if key != self.var and key in self.body.freevars}
        var, body = self.var, self.body
        if any(var in value.freevars for value in inside.values()):
            var = type(var).fresh(var.name, var.cod, body, *inside.values())
            body = Substitution({self.var: var})(body)
        return type(self)(var, Substitution(inside)(body))

    def normal_form(self):
        return type(self)(self.var, self.body.normal_form())


@dataclass
class Substitution:
    """
    The simultaneous, capture-avoiding substitution of terms for variables.

    Parameters:
        inside : The term substituted for each variable, of the same type.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> f, x, y = (X >> Y)("f"), Variable("x", X), Variable("y", X)
    >>> assert Substitution({x: y})(X(lambda y: f(x))) == X(lambda y_: f(y))
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
        return term.substitute(self)


Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction


class Equation(markov.Equation):
    """ The :class:`markov.Equation` of closed diagrams. """
