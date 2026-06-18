
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
from abc import ABC, abstractmethod
from typing import Dict, Callable
from inspect import signature
from functools import reduce

from discopy import cat, biclosed, markov, messages
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

    def __str__(self):
        return f"({self.exponent} >> {self.base})"


class TermBase(ABC):
    cod: Ty
    ob = Ty

    @property
    def dom(self):
        return self.ob().tensor(*(variable.cod for variable in self.freevars))

    def __eq__(self, other):
        return isinstance(other, TermBase)\
            and self.alpha_equal(
                other, Substitution(()), Substitution(()))

    def __hash__(self):
        return hash(self.alpha_key(Substitution(())))

    def __call__(self, *others: Term):
        return reduce(lambda fun, arg: Application(fun, arg), others, self)

    @classmethod
    def id(cls, dom):
        variables = tuple(
            Variable(obj, f"x{i}") for i, obj in enumerate(dom))
        return variables[0] if len(variables) == 1 else pack(*variables)

    @classmethod
    def ev(cls, base, exponent, left=True):
        exp = exponent >> base
        func, arg = Variable(exp, "f"), Variable(exponent, "x")
        return Application(func, arg)

    def then(self: Term, other: Term) -> Term:
        if self.cod != other.dom:
            raise ValueError
        return unpack(self, other.freevars, other)

    def tensor(self: Term, other=None, *others):
        if other is None:
            return self
        return pack(self, other, *others)

    def curry(self: Term, n=1, left=True):
        if n != 1 or not left:
            raise NotImplementedError
        if not self.freevars:
            raise ValueError
        return Abstraction(self.freevars[-1], self)

    @staticmethod
    def abs(*vars) -> Callable[[Term], Term]:
        def _inner(body):
            cont = body
            for var in vars:
                cont = Abstraction(var, cont)
            return cont
        return _inner

    @property
    @abstractmethod
    def freevars(self) -> list[Variable]: ...

    @abstractmethod
    def to_diagram(self, category=None) -> Diagram: ...

    @abstractmethod
    def to_map(self, category=None) -> CombinatorialMap: ...

    @abstractmethod
    def alpha_equal(self, other, left_sub, right_sub) -> bool: ...

    @abstractmethod
    def alpha_key(self, substitution): ...

    @staticmethod
    def alpha_bound(cod, index):
        return Variable(cod, f"__bound_{index}")


@dataclass(frozen=True, eq=False)
class Constant(TermBase):
    cod: Ty
    name: str

    def __str__(self):
        return self.name

    @property
    def freevars(self) -> list[Variable]:
        return []

    def to_diagram(self, category=None, box_factory=None):
        category, box_factory = category or Diagram, box_factory or Box
        return box_factory(self.name, category.ob(), self.cod)

    def to_map(self, category=None):
        raise ValueError("Constants are not pure linear lambda terms.")

    def alpha_equal(self, other, left_sub, right_sub) -> bool:
        return isinstance(other, Constant)\
            and (self.cod, self.name) == (other.cod, other.name)

    def alpha_key(self, substitution):
        return ("constant", self.cod, self.name)


@dataclass(frozen=True, eq=False)
class Variable(TermBase):
    ty: Ty
    name: str

    def __str__(self):
        return self.name

    # REVIEW: Can we avoid this?
    @property
    def cod(self) -> Ty:
        return self.ty.inside[0] if self.ty.is_exp else self.ty

    @property
    def freevars(self):
        return [self]

    def to_diagram(self, category=None):
        return (category or Diagram).id(self.cod)

    def to_map(self, category=None):
        category = category or CombinatorialMap
        cmap = category.id(self.cod)
        assert_term_map(cmap, self, category)
        return cmap

    def alpha_equal(self, other, left_sub, right_sub) -> bool:
        return isinstance(other, Variable)\
            and self.alpha_key(left_sub) == other.alpha_key(right_sub)

    def alpha_key(self, substitution):
        image = substitution(self)
        return ("free", self.cod, self.name) if image is self\
            else ("bound", image.cod, image.name)

    def same_variable(self, other):
        return isinstance(other, Variable)\
            and (self.cod, self.name) == (other.cod, other.name)


@dataclass(frozen=True, eq=False)
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

    def to_map(self, category=None):
        category = category or CombinatorialMap
        func_map = self.func.to_map(category)
        args_map = self.args.to_map(category)
        if common_vars := set(self.func.freevars).intersection(
                self.args.freevars):
            plural = "" if len(common_vars) == 1 else "s"
            names = ", ".join(var.name for var in common_vars)
            raise ValueError(messages.NON_LINEAR_TERM.format(
                suffix=plural, names=names))
        app = Eval(self.func.cod, left=True)
        cmap = (func_map @ args_map) >> category.from_box(app)
        assert_term_map(cmap, self, category)
        return cmap

    def alpha_equal(self, other, left_sub, right_sub) -> bool:
        return isinstance(other, Application)\
            and self.func.alpha_equal(other.func, left_sub, right_sub)\
            and self.args.alpha_equal(other.args, left_sub, right_sub)

    def alpha_key(self, substitution):
        return ("application", self.func.alpha_key(substitution),
                self.args.alpha_key(substitution))


@dataclass(frozen=True, eq=False)
class Abstraction(TermBase):
    var: Variable
    body: Term

    @property
    def cod(self) -> Ty:
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

    def to_map(self, category=None):
        category = category or CombinatorialMap
        body_map = self.body.to_map(category)
        free_vars = self.body.freevars
        matches = [
            index for index, variable in enumerate(free_vars)
            if variable == self.var]
        if len(matches) != 1:
            raise ValueError(messages.NON_LINEAR_TERM.format(
                suffix="s", names=[free_vars[i] for i in matches]))
        index, = matches
        lam = Coeval(self.cod, left=True)
        cmap = body_map.plug_input(index, lam, self.cod)
        assert_term_map(cmap, self, category)
        return cmap

    def alpha_equal(self, other, left_sub, right_sub) -> bool:
        if not isinstance(other, Abstraction) or self.var.cod != other.var.cod:
            return False
        variable = self.alpha_bound(self.var.cod, len(left_sub))
        return self.body.alpha_equal(
            other.body,
            left_sub.extend(((self.var, variable), )),
            right_sub.extend(((other.var, variable), )))

    def alpha_key(self, substitution):
        variable = self.alpha_bound(self.var.cod, len(substitution))
        return ("abstraction", self.var.cod, self.body.alpha_key(
            substitution.extend(((self.var, variable), ))))


@dataclass(frozen=True, eq=False)
class Pack(TermBase):
    terms: tuple[Term, ...]

    def __init__(self, *terms: Term):
        object.__setattr__(self, "terms", tuple(terms))

    @property
    def cod(self):
        return self.ob().tensor(*(term.cod for term in self.terms))

    def __str__(self):
        return "pack(" + ", ".join(map(str, self.terms)) + ")"

    @property
    def freevars(self):
        return sum([term.freevars for term in self.terms], [])

    def to_diagram(self, category=None):
        category = category or Diagram
        result = category.id(category.ob())
        for term in self.terms:
            result = result @ term.to_diagram(category)
        return result

    def to_map(self, category=None):
        category = category or CombinatorialMap
        result = category.id()
        for term in self.terms:
            result = result @ term.to_map(category)
        assert_term_map(result, self, category)
        return result

    def alpha_equal(self, other, left_sub, right_sub) -> bool:
        return isinstance(other, Pack) and len(self.terms) == len(other.terms)\
            and all(left.alpha_equal(right, left_sub, right_sub)
                    for left, right in zip(self.terms, other.terms))

    def alpha_key(self, substitution):
        return ("pack", tuple(
            term.alpha_key(substitution) for term in self.terms))


@dataclass(frozen=True, eq=False)
class Unpack(TermBase):
    package: Term
    variables: tuple[Variable, ...]
    body: Term

    def __init__(self, package: Term, variables, body: Term):
        variables = tuple(variables)
        if Ty().tensor(*(variable.cod for variable in variables))\
                != package.cod:
            raise ValueError
        object.__setattr__(self, "package", package)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "body", body)

    @property
    def cod(self):
        return self.body.cod

    def __str__(self):
        names = ", ".join(variable.name for variable in self.variables)
        return f"unpack({self.package}, lambda {names}: {self.body})"

    @property
    def freevars(self):
        return self.package.freevars + [
            variable for variable in self.body.freevars
            if variable not in self.variables]

    def _remaining(self):
        return [
            variable for variable in self.body.freevars
            if variable not in self.variables]

    def _permutation(self):
        remaining = self._remaining()
        result = []
        for variable in self.body.freevars:
            if variable in self.variables:
                result.append(self.variables.index(variable))
            else:
                result.append(len(self.variables) + remaining.index(variable))
        return result

    def to_diagram(self, category=None):
        category = category or Diagram
        remaining = self._remaining()
        remaining_dom = category.ob().tensor(
            *(variable.cod for variable in remaining))
        package = self.package.to_diagram(category)
        start = package @ category.id(remaining_dom)
        permutation = category.permutation(
            self._permutation(), self.package.cod @ remaining_dom)
        return start >> permutation >> self.body.to_diagram(category)

    def to_map(self, category=None):
        category = category or CombinatorialMap
        diagram = self.to_diagram(category.category)
        result = category.from_hypergraph(diagram.to_hypergraph())
        assert_term_map(result, self, category)
        return result

    def alpha_equal(self, other, left_sub, right_sub) -> bool:
        if not isinstance(other, Unpack)\
                or len(self.variables) != len(other.variables):
            return False
        if not self.package.alpha_equal(
                other.package, left_sub, right_sub):
            return False
        left_extension, right_extension = [], []
        for i, (left_var, right_var) in enumerate(
                zip(self.variables, other.variables)):
            if left_var.cod != right_var.cod:
                return False
            variable = self.alpha_bound(left_var.cod, len(left_sub) + i)
            left_extension.append((left_var, variable))
            right_extension.append((right_var, variable))
        return self.body.alpha_equal(
            other.body,
            left_sub.extend(left_extension),
            right_sub.extend(right_extension))

    def alpha_key(self, substitution):
        extension = tuple(
            (variable, self.alpha_bound(
                variable.cod, len(substitution) + i))
            for i, variable in enumerate(self.variables))
        return (
            "unpack",
            self.package.alpha_key(substitution),
            tuple(variable.cod for variable in self.variables),
            self.body.alpha_key(substitution.extend(extension)))


type Term = Constant | Variable | Application | Abstraction | Pack | Unpack


def pack(*terms: Term) -> Pack:
    return Pack(*terms)


def unpack(package: Term, variables=None, body: Term | None = None) -> Unpack:
    if callable(variables) and body is None:
        func = variables
        names = list(signature(func).parameters.keys())
        if len(names) != len(package.cod):
            raise ValueError
        variables = tuple(
            Variable(obj, name) for obj, name in zip(package.cod, names))
        body = func(*variables)
    elif body is None:
        raise ValueError
    return Unpack(package, variables, body)


def assert_term_map(
    cmap,
    term,
    category: type[CombinatorialMap] | None = None
):
    category = category or CombinatorialMap
    if cmap.dom != category.ob().tensor(
            *(variable.cod for variable in term.freevars)):
        raise ValueError
    if cmap.cod != term.cod:
        raise ValueError
    if any(len(cycle) != 3 for cycle in cmap.node_cycles):
        raise ValueError


@dataclass
class Substitution:
    inside: Dict[Variable, Term] | tuple[tuple[Variable, Term], ...]

    def __len__(self):
        return len(tuple(self.items()))

    def items(self):
        return self.inside.items() if hasattr(self.inside, "items")\
            else self.inside

    def extend(self, inside) -> Substitution:
        items = inside.items() if hasattr(inside, "items") else inside
        return type(self)(tuple(self.items()) + tuple(items))

    def without(self, variables) -> Substitution:
        return type(self)(tuple(
            (k, v) for k, v in self.items()
            if all(not k.same_variable(variable)
                   for variable in variables)))

    def __call__(self, term: Term) -> Term:
        if isinstance(term, Variable):
            for variable, image in reversed(tuple(self.items())):
                if variable.same_variable(term):
                    return image
            return term
        if isinstance(term, Application):
            return self(term.func)(self(term.args))
        if isinstance(term, Abstraction):
            other = self.without((term.var, ))
            return Abstraction(term.var, other(term.body))
        if isinstance(term, Pack):
            return pack(*(self(t) for t in term.terms))
        if isinstance(term, Unpack):
            package = self(term.package)
            other = self.without(term.variables)
            return unpack(package, term.variables, other(term.body))
        raise ValueError(f"not a term: {term!r}")


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
    drawing_name = "Eval"


class Coeval(biclosed.Coeval, Box):
    "The coevaluation of an exponential type, i.e. the dagger of an Eval."
    drawing_name = "$\\lambda$"


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


class CombinatorialMap(markov.CombinatorialMap):
    functor = Functor


Diagram.hypergraph_factory = Hypergraph
Diagram.map_factory = CombinatorialMap
Diagram.copy_factory = Copy
Diagram.braid_factory = Swap
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.trace_factory = Trace
Diagram.discard_factory = lambda X: Copy(X, 0)
Diagram.sum_factory = Sum

Id = Diagram.id
