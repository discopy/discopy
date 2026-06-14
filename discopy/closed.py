
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
        return _tensor_types(variable.cod for variable in self.freevars)

    def __eq__(self, other):
        return isinstance(other, TermBase) and _alpha_equal(self, other)

    def __hash__(self):
        return hash(_alpha_key(self))

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


@dataclass(frozen=True, eq=False)
class Variable(TermBase):
    cod: Ty
    name: str

    def __str__(self):
        return self.name

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


@dataclass(frozen=True, eq=False)
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


@dataclass(frozen=True, eq=False)
class Pack(TermBase):
    terms: tuple[Term, ...]

    def __init__(self, *terms: Term):
        object.__setattr__(self, "terms", tuple(terms))

    @property
    def cod(self):
        return _tensor_types(term.cod for term in self.terms)

    def __str__(self):
        return "pack(" + ", ".join(map(str, self.terms)) + ")"

    @property
    def freevars(self):
        return sum([term.freevars for term in self.terms], [])

    def to_diagram(self, category=None):
        category = category or Diagram
        result = category.id(category.ty_factory())
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


@dataclass(frozen=True, eq=False)
class Unpack(TermBase):
    package: Term
    variables: tuple[Variable, ...]
    body: Term

    def __init__(self, package: Term, variables, body: Term):
        variables = tuple(variables)
        if _tensor_types(variable.cod for variable in variables)\
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
        remaining_dom = _tensor_types(
            (variable.cod for variable in remaining), category)
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


def _alpha_equal(left, right, left_bound=None, right_bound=None):
    left_bound = () if left_bound is None else left_bound
    right_bound = () if right_bound is None else right_bound
    if isinstance(left, Constant) and isinstance(right, Constant):
        return (left.cod, left.name) == (right.cod, right.name)
    if isinstance(left, Variable) and isinstance(right, Variable):
        left_index = _lookup_bound(left_bound, left)
        right_index = _lookup_bound(right_bound, right)
        if left_index is not None or right_index is not None:
            return left.cod == right.cod and left_index == right_index
        return (left.cod, left.name) == (right.cod, right.name)
    if isinstance(left, Application) and isinstance(right, Application):
        return _alpha_equal(left.func, right.func, left_bound, right_bound)\
            and _alpha_equal(left.args, right.args, left_bound, right_bound)
    if isinstance(left, Abstraction) and isinstance(right, Abstraction):
        if left.var.cod != right.var.cod:
            return False
        index = len(left_bound)
        return _alpha_equal(
            left.body, right.body,
            left_bound + ((left.var, index), ),
            right_bound + ((right.var, index), ))
    if isinstance(left, Pack) and isinstance(right, Pack):
        return len(left.terms) == len(right.terms) and all(
            _alpha_equal(l, r, left_bound, right_bound)
            for l, r in zip(left.terms, right.terms))
    if isinstance(left, Unpack) and isinstance(right, Unpack):
        if len(left.variables) != len(right.variables):
            return False
        if not _alpha_equal(
                left.package, right.package, left_bound, right_bound):
            return False
        for lvar, rvar in zip(left.variables, right.variables):
            if lvar.cod != rvar.cod:
                return False
        shift = len(left_bound)
        return _alpha_equal(
            left.body, right.body,
            left_bound + tuple(
                (variable, shift + i)
                for i, variable in enumerate(left.variables)),
            right_bound + tuple(
                (variable, shift + i)
                for i, variable in enumerate(right.variables)))
    return False


def _alpha_key(term, bound=None):
    bound = () if bound is None else bound
    if isinstance(term, Constant):
        return ("constant", term.cod, term.name)
    if isinstance(term, Variable):
        index = _lookup_bound(bound, term)
        return ("bound", term.cod, index) if index is not None\
            else ("free", term.cod, term.name)
    if isinstance(term, Application):
        return ("application", _alpha_key(term.func, bound),
                _alpha_key(term.args, bound))
    if isinstance(term, Abstraction):
        index = len(bound)
        return ("abstraction", term.var.cod, _alpha_key(
            term.body, bound + ((term.var, index), )))
    if isinstance(term, Pack):
        return ("pack", tuple(_alpha_key(t, bound) for t in term.terms))
    if isinstance(term, Unpack):
        shift = len(bound)
        return (
            "unpack",
            _alpha_key(term.package, bound),
            tuple(variable.cod for variable in term.variables),
            _alpha_key(term.body, bound + tuple(
                (variable, shift + i)
                for i, variable in enumerate(term.variables))))
    raise TypeError


def _lookup_bound(bound, variable):
    for other, index in reversed(bound):
        if _same_variable(other, variable):
            return index
    return None


def _same_variable(left, right):
    return isinstance(left, Variable) and isinstance(right, Variable)\
        and (left.cod, left.name) == (right.cod, right.name)


def _tensor_types(types, category=None):
    factory = Ty if category is None else getattr(
        category, "ty_factory", None)
    if factory is None:
        factory = category.category.ty_factory
    result = factory()
    for typ in types:
        result = result @ typ
    return result


def assert_term_map(cmap, term, category: type[CombinatorialMap] | None = None):
    category = category or CombinatorialMap
    if cmap.dom != _tensor_types(
            (variable.cod for variable in term.freevars), category):
        raise ValueError
    if cmap.cod != term.cod:
        raise ValueError
    if any(len(cycle) != 3 for cycle in cmap.node_cycles):
        raise ValueError


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
        elif isinstance(term, Pack):
            return pack(*(self(t) for t in term.terms))
        elif isinstance(term, Unpack):
            package = self(term.package)
            other = Substitution({
                k: v for k, v in self.inside.items()
                if k not in term.variables})
            return unpack(package, term.variables, other(term.body))
        else:
            raise ValueError(f"not a term: {term!r}")


@ar_factory
class Diagram(markov.Diagram, biclosed.Diagram, ClosedCategory):
    """
    A closed diagram is both a markov and a biclosed diagram.

    A diagram applied to another post-composes their tensor with an `Eval`.
    """
    ob = ty_factory = Ty

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
