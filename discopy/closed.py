
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
    Product
    Projection
    LetStatement
    Diagram
    Box
    Eval
    Coeval
    Curry
    Sum
    Functor
    CMap

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        let

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
from inspect import signature
from typing import Callable, Dict, ClassVar

from discopy import cat, monoidal, biclosed, markov
from discopy.abc import ClosedCategory
from discopy.cat import ob_factory, ar_factory
from discopy.drawing import Drawing
from discopy.utils import assert_isinstance, factory_name


@ob_factory
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

    ob = Ty

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


class Discard(markov.Discard, Copy):
    "The discard of an atomic type in a closed category."


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
        if self.cod is Drawing and isinstance(
                other, (markov.Copy, markov.Merge, markov.Swap)):
            return monoidal.Functor.__call__(self, other)
        return super().__call__(other)


class Hypergraph(markov.Hypergraph):
    functor = Functor


class CMap(biclosed.CMap):
    functor = Functor
    require_planar = False


Diagram.hypergraph_factory = Hypergraph
Diagram.map_factory = CMap
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

    Note
    ----
    Calling a term with several arguments (or none at all) applies it to
    the :class:`Product` of the arguments, e.g.

    >>> X, Y = Ty("X"), Ty("Y")
    >>> point, effect = Constant("point", Ty() >> X), (X @ Y >> Ty())("f")
    >>> assert point() == point(Product())
    >>> x, y = Variable("x", X), Variable("y", Y)
    >>> assert effect(x, y) == effect(Product(x, y))
    """
    functor = Functor.id(Diagram)

    def __call__(self, *others):
        other = others[0] if len(others) == 1 else Product(*others)
        return Application(self, other, left=False)

    def eval(self, functor=None, context=None):
        """
        Evaluate a term by calling :code:`__eval__` then simplifying the
        result by round-trip to hypergraph, whenever the codomain of the
        functor is a category of diagrams.
        """
        functor = functor or self.functor
        result = self.__eval__(functor, context)
        return result.simplify()\
            if isinstance(result, markov.Diagram) else result


type Term = Constant | Variable | Application | Abstraction\
    | Product | Projection | LetStatement


def unbiased_tensor(functor: Functor, diagrams: list) -> Diagram:
    """
    The tensor of a list of diagrams in the codomain of a functor.

    Parameters:
        functor : The functor in whose codomain to tensor.
        diagrams : The list of diagrams to tensor, possibly empty.
    """
    result = functor.cod.id(functor(functor.dom.ob()))
    for diagram in diagrams:
        result = result @ diagram
    return result


class Constant(TermBase, biclosed.Constant):
    def __eval__(self, functor, context):
        if not context:
            return biclosed.Constant.eval(self, functor)
        return functor.cod.discard(functor(context.dom))\
            >> biclosed.Constant.eval(self, functor)


class Variable(TermBase, biclosed.Variable):
    def __eval__(self, functor, context):
        if not context:
            return functor.cod.id(functor(self.cod))
        return unbiased_tensor(functor, [
            functor.cod.id(functor(x.cod)) if x == self
            else functor.cod.discard(functor(x.cod))
            for x in context.inside])


class Application(TermBase, biclosed.Application):
    def __check_dom__(self, func, args, left):
        self.overlap = set(func.freevars).intersection(args.freevars)
        self.freevars = list(dict.fromkeys(func.freevars + args.freevars))\
            if self.overlap else func.freevars + args.freevars
        return self.ob().tensor(*[x.cod for x in self.freevars])

    def __eval__(self, functor, context):
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

    def __eval__(self, functor, context):
        if context:
            new_context = Context([self.var] + context.inside)
            body = self.body.eval(functor=functor, context=new_context)
            return body.curry()
        i, n = self.body.freevars.index(self.var), len(self.body.freevars)
        body = self.body.eval(functor=functor)
        p = [0] + [j + 1 if j < i else j for j in range(n) if j != i]
        return (body.permutation(p, body.dom).dagger() >> body).curry()


class Product(TermBase):
    """
    The product of a tuple of terms, i.e. the tensor of their diagrams
    with a shared context.

    Parameters:
        terms : The terms inside the product.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> x, y = Variable("x", X), Variable("y", Y)
    >>> assert Product(x, y).cod == X @ Y
    >>> assert Product(x, y).eval() == Id(X @ Y)
    >>> assert Product().cod == Ty() and Product().eval() == Id(Ty())

    Note
    ----
    When the terms of a product share free variables, their context is copied:

    >>> assert Product(x, x).eval() == Diagram.copy(X)
    """
    def __init__(self, *terms: Term):
        for term in terms:
            assert_isinstance(term, TermBase)
        self.terms = terms
        name = f"({', '.join(map(str, terms))})"
        cod = self.ob().tensor(*[term.cod for term in terms])
        dom = self.__check_dom__()
        super().__init__(name, dom, cod)

    def __check_dom__(self):
        freevars = [x for term in self.terms for x in term.freevars]
        self.overlap = {x for x in freevars if freevars.count(x) > 1}
        self.freevars = list(dict.fromkeys(freevars)) if self.overlap\
            else freevars
        return self.ob().tensor(*[x.cod for x in self.freevars])

    @property
    def constants(self):
        return [c for term in self.terms for c in term.constants]

    def __eval__(self, functor, context):
        if context is None:
            if not self.overlap:
                return unbiased_tensor(functor, [
                    term.eval(functor=functor) for term in self.terms])
            context = Context(self.freevars)
        return functor.cod.copy(functor(context.dom), len(self.terms))\
            >> unbiased_tensor(functor, [
                term.eval(functor=functor, context=context)
                for term in self.terms])

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.terms))})"


class Projection(TermBase):
    """
    The projection of a term onto one of the atomic components of its type.

    Parameters:
        arg : The term to project.
        index : The index of the atomic component.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> x, y = Variable("x", X), Variable("y", Y)
    >>> assert Projection(Product(x, y), 1).cod == Y
    >>> assert Projection(Product(x, y), 1).eval()\\
    ...     == Diagram.discard(X) @ Y
    """
    def __init__(self, arg: Term, index: int):
        assert_isinstance(arg, TermBase)
        assert_isinstance(index, int)
        if not 0 <= index < len(arg.cod):
            raise IndexError(f"Expected index < {len(arg.cod)}")
        self.arg, self.index = arg, index
        self.freevars = arg.freevars
        name = f"{arg}[{index}]"
        super().__init__(name, arg.dom, arg.cod[index])

    @property
    def constants(self):
        return self.arg.constants

    def __eval__(self, functor, context):
        arg = self.arg.eval(functor=functor, context=context)
        return arg >> unbiased_tensor(functor, [
            functor.cod.id(functor(x)) if i == self.index
            else functor.cod.discard(functor(x))
            for i, x in enumerate(self.arg.cod)])

    def __repr__(self):
        return factory_name(type(self)) + f"({self.arg!r}, {self.index})"


class LetStatement(TermBase):
    """
    A let statement evaluates a term ``expression`` and binds its output to a
    tuple of ``variables`` which may then occur freely in a term ``body``.

    Parameters:
        expression : The term to evaluate.
        variables : The variables to which the output is bound.
        body : The term in which the variables may occur.

    Example
    -------
    A let statement evaluates its expression exactly once, e.g. the term
    ``once`` copies the output of one application of ``f`` while the term
    ``twice`` applies it two times:

    >>> X, Y = Ty("X"), Ty("Y")
    >>> f, x = Constant("f", X >> Y), Variable("x", X)
    >>> once = let(f(x), lambda y: Product(y, y))
    >>> twice = Product(f(x), f(x))
    >>> assert once.cod == twice.cod == Y @ Y

    >>> from discopy.drawing import Equation
    >>> Equation(once.eval(), twice.eval(), symbol="$\\\\neq$").draw(
    ...     path='docs/_static/closed/let-once-vs-twice.png')

    .. image:: /_static/closed/let-once-vs-twice.png
        :align: center
    """
    def __init__(self, expression: Term, variables: tuple[Variable, ...],
                 body: Term):
        assert_isinstance(expression, TermBase)
        assert_isinstance(body, TermBase)
        for variable in variables:
            assert_isinstance(variable, Variable)
        self.expression, self.variables = expression, tuple(variables)
        self.body = body
        cod = self.ob().tensor(*[x.cod for x in self.variables])
        if expression.cod != cod:
            raise ValueError(f"Expected {cod}, got {expression.cod}")
        varnames = " " + ", ".join(x.name for x in self.variables)\
            if self.variables else ""
        name = f"let({expression}, lambda{varnames}: {body})"
        body_freevars = [
            x for x in body.freevars if x not in self.variables]
        freevars = expression.freevars + body_freevars
        self.overlap = set(expression.freevars).intersection(body_freevars)
        self.freevars = list(dict.fromkeys(freevars)) if self.overlap\
            else freevars
        dom = self.ob().tensor(*[x.cod for x in self.freevars])
        super().__init__(name, dom, body.cod)

    @property
    def constants(self):
        return self.expression.constants + self.body.constants

    def __eval__(self, functor, context):
        if context is None:
            context = Context(self.freevars)
        new_context = Context(list(self.variables) + context.inside)
        expression = self.expression.eval(functor=functor, context=context)
        body = self.body.eval(functor=functor, context=new_context)
        dom = functor(context.dom)
        return functor.cod.copy(dom)\
            >> expression @ functor.cod.id(dom) >> body

    def __repr__(self):
        return factory_name(type(self)) + f"({self.expression!r}, "\
            f"{self.variables!r}, {self.body!r})"


def let(expression: Term, func: Callable) -> LetStatement:
    """
    Syntactic sugar for :class:`LetStatement`, with variable names extracted
    by introspection of ``func`` and their types from ``expression.cod``.

    Parameters:
        expression : The term to evaluate.
        func : A callable from variables to the body of the let statement.

    Example
    -------
    Here is the snake equation with both the cup and the cap implemented as
    let statements:

    >>> X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    >>> state = Constant("state", Ty() >> X @ Y)
    >>> effect = Constant("effect", Y @ Z >> Ty())
    >>> snake = Z(lambda z: let(
    ...     state(), lambda x, y: let(effect(y, z), lambda: x)))
    >>> assert snake.cod == Z >> X
    >>> snake.draw(path='docs/_static/closed/snake-let.png')

    .. image:: /_static/closed/snake-let.png
        :align: center
    """
    varnames = list(signature(func).parameters)
    if len(varnames) != len(expression.cod):
        raise ValueError(f"Expected {len(expression.cod)} variables, "
                         f"got {len(varnames)}")
    variables = tuple(
        Variable(name, typ) for name, typ in zip(varnames, expression.cod))
    return LetStatement(expression, variables, func(*variables))


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
        if isinstance(term, Application):
            return self(term.func)(self(term.args))
        if isinstance(term, Abstraction):
            other = Substitution(
                {k: v for k, v in self.inside.items() if k != term.var})
            return type(term)(term.var, other(term.body), term.left)
        if isinstance(term, Product):
            return type(term)(*map(self, term.terms))
        if isinstance(term, Projection):
            return type(term)(self(term.arg), term.index)
        if isinstance(term, LetStatement):
            other = Substitution({
                k: v for k, v in self.inside.items()
                if k not in term.variables})
            return type(term)(
                self(term.expression), term.variables, other(term.body))
        return term


Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction
