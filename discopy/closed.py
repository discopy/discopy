
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
from pyparsing import common
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Callable, Self
from inspect import signature
from functools import reduce

from discopy import cat, biclosed, markov
from discopy.abc import ClosedCategory
from discopy.cat import ob_factory, ar_factory
from discopy.combinatorial_map import Permutation
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

    def __call__(self, *others: Term):
        return reduce(lambda fun, arg: Application(fun, arg), others, self)

    @staticmethod
    def abs(*vars) -> Callable[[Term], Term]:
        def _inner(body):
            cont = body
            for var in vars:
                cont = Abstraction(var, cont)
            return cont
        return _inner

    def to_map(self, category=None, box_factory=None) -> CombinatorialMap:
        cmap, freevars = self._to_map_data(category, box_factory)
        if cmap.dom != _tensor_types(
                (variable.cod for variable in freevars), category):
            raise ValueError
        if cmap.cod != self.cod or len(cmap.cod) != 1:
            raise ValueError
        if any(len(cycle) != 3 for cycle in cmap.node_cycles):
            raise ValueError
        return cmap

    @property
    @abstractmethod
    def freevars(self) -> list[Variable]: ...

    @abstractmethod
    def to_diagram(self, category=None) -> Diagram: ...

    @abstractmethod
    def _to_map_data(self, category=None, box_factory=None): ...


type Term = Constant | Variable | Application | Abstraction


@dataclass(frozen=True)
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

    def _to_map_data(self, category=None, box_factory=None):
        raise ValueError("Constants are not pure linear lambda terms.")


@dataclass(frozen=True)
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

    def _to_map_data(self, category=None, box_factory=None):
        category = category or Category
        return _map_factory(category).id(self.cod), (self, )


@dataclass(frozen=True)
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

    def _to_map_data(self, category=None, box_factory=None):
        category, box_factory = category or Category, box_factory or Box
        func_map, func_vars = self.func._to_map_data(category, box_factory)
        args_map, args_vars = self.args._to_map_data(category, box_factory)
        if common_vars := set(func_vars).intersection(args_vars):
            raise ValueError(f"Non-linear term: variable{'' if len(common_vars) == 1 else 's'} {', '.join(var.name for var in common_vars)} used more than once")
        app = box_factory("@", self.func.cod @ self.args.cod, self.cod)
        return (func_map @ args_map) >> _map_factory(category).from_box(app),\
            func_vars + args_vars


@dataclass(frozen=True)
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

    def _to_map_data(self, category=None, box_factory=None):
        category, box_factory = category or Category, box_factory or Box
        body_map, body_vars = self.body._to_map_data(category, box_factory)
        matches = [
            index for index, variable in enumerate(body_vars)
            if variable == self.var]
        if len(matches) != 1:
            raise ValueError(
                "Non-linear term: bound variables must occur exactly once.")
        index, = matches
        lam = box_factory(
            "lambda", self.body.cod, self.cod @ self.var.cod)
        cmap = _abstract_map(body_map, index, lam, self.cod, category)
        return cmap, body_vars[:index] + body_vars[index + 1:]


def _map_factory(category=None):
    category = category or Category
    return getattr(category.ar, "map_factory", CombinatorialMap)


def _tensor_types(types, category=None):
    result = (category or Category).ob()
    for typ in types:
        result = result @ typ
    return result


def _abstract_map(body_map, var_index, lam, cod, category=None):
    map_factory = _map_factory(category)
    old_dom, old_cod = len(body_map.dom), len(body_map.cod)
    if old_cod != 1:
        raise ValueError

    old_input = var_index
    old_output = body_map.n_ports - 1
    new_dom = _tensor_types(
        (obj for i, obj in enumerate(body_map.dom) if i != var_index),
        category)
    boxes = body_map.boxes + (lam, )
    offsets = body_map.offsets + (None, )

    mapping, new_index = {}, 0
    for i in range(old_dom):
        if i != old_input:
            mapping[i] = new_index
            new_index += 1
    for i in range(old_dom, body_map.n_ports - old_cod):
        mapping[i] = new_index
        new_index += 1

    lambda_dom = new_index
    lambda_root = new_index + 1
    lambda_param = new_index + 2
    new_output = new_index + 3

    edge_pairs = []
    for i, j in enumerate(body_map.edge):
        if i < j and i not in [old_input, old_output]\
                and j not in [old_input, old_output]:
            edge_pairs.append((mapping[i], mapping[j]))

    input_partner = body_map.edge[old_input]
    output_partner = body_map.edge[old_output]
    if input_partner == old_output:
        edge_pairs.append((lambda_param, lambda_dom))
    else:
        edge_pairs.append((mapping[input_partner], lambda_param))
        edge_pairs.append((mapping[output_partner], lambda_dom))
    edge_pairs.append((lambda_root, new_output))
    edge = Permutation.from_transpositions(edge_pairs, new_output + 1)

    node = Permutation.from_cycles(
        [tuple(mapping[i] for i in cycle) for cycle in body_map.node_cycles]
        + [(lambda_dom, lambda_root, lambda_param)],
        new_output + 1)
    return map_factory(new_dom, cod, boxes, edge, node, offsets)


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
    category, functor = Category, Functor


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
