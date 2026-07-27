
"""
The free closed markov category, i.e. with copy, discard, exponentials and
products that are not strictly associative.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Exp
    Product
    TermBase
    Constant
    Variable
    Application
    Abstraction
    Tuple
    Projection
    Let
    Diagram
    Box
    Eval
    Coeval
    Curry
    Pack
    Unpack
    Copy
    Discard
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
from inspect import signature
from typing import Callable, Dict, ClassVar

from discopy import cat, monoidal, biclosed, markov, hypergraph
from discopy.abc import ClosedCategory
from discopy.cat import factory
from discopy.drawing import Drawing
from discopy.utils import assert_isinstance, factory_name, from_tree


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
    def __mul__(self, other: Ty) -> Ty:
        return self.product(other)

    def product(self, *others: Ty) -> Ty:
        "The :class:`Product` of a type with a tuple of ``others``."
        return self.ar(self.product_factory(self, *others))

    @property
    def is_product(self):
        """
        Whether the type is a :class:`Product` object.

        Example
        -------
        >>> x, y = Ty('x'), Ty('y')
        >>> assert (x * y).is_product and (x * y @ Ty()).is_product
        """
        return len(self) == 1 and isinstance(self.inside[0], Product)

    @property
    def factors(self):
        "The factors of a product type, assumes ``self.is_product``."
        assert self.is_product
        return self.inside[0].factors


class Exp(biclosed.Exp):
    "An exponential object in a markov category."

    ob = Ty

    def __str__(self):
        return f"({self.exponent} >> {self.base})"


class Product(biclosed.Ob):
    """
    The product of a tuple of types, which is not strictly associative,
    called with ``*`` in the binary case.

    Parameters:
        factors : The factors of the product.

    Example
    -------
    >>> X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
    >>> assert X * Y == Ty(Product(X, Y))
    >>> assert (X * Y) * Z != X * (Y * Z) != X.product(Y, Z)

    Evaluation strictifies a product to the tensor of its factors, see
    :class:`Pack`, so that the three types above are all interpreted as
    ``X @ Y @ Z``.
    """
    ob = Ty

    def __init__(self, *factors: Ty):
        for typ in factors:
            assert_isinstance(typ, self.ob)
        self.factors = factors
        super().__init__(str(self))

    def __eq__(self, other):
        return isinstance(other, type(self))\
            and self.factors == other.factors

    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return "(" + " * ".join(
            str(typ) if len(typ) == 1 else f"({typ})"
            for typ in self.factors) + ")"

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.factors))})"

    def to_tree(self):
        return {
            'factory': factory_name(type(self)),
            'factors': [typ.to_tree() for typ in self.factors]}

    @classmethod
    def from_tree(cls, tree):
        return cls(*map(from_tree, tree['factors']))


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

    def to_term(self) -> Term:
        """
        Read a causal diagram as a term in fine-grain call-by-value style:
        one let statement per box in topological order with a variable for
        each wire, going through :class:`Hypergraph` so that the copy,
        discard and swap structure simplifies away into the spiders.

        The free variables of the term are the inputs that the diagram
        actually uses; a diagram with no box is a tuple of variables.

        Example
        -------
        >>> X, Y = Ty("X"), Ty("Y")
        >>> f, g = Box("f", X, Y @ Y), Box("g", Y @ Y, Y)
        >>> diagram = Diagram.copy(X) >> f @ Diagram.discard(X) >> g
        >>> print(diagram.to_term())
        let(f(x0), lambda x1, x2: g(Tuple(x1, x2)))
        >>> assert Diagram.swap(X, Y).to_term()\\
        ...     == Tuple(Variable("x1", Y), Variable("x0", X))
        """
        hypergraph = Hypergraph.from_diagram(self)
        if not hypergraph.is_causal:
            raise ValueError(f"Expected a causal diagram, got {self}")
        variables = [self.ob.variable_factory(f"x{i}", typ)
                     for i, typ in enumerate(hypergraph.spider_types)]
        outputs = [variables[i] for i in hypergraph.cod_wires]
        result = outputs[0] if len(outputs) == 1 else Tuple(*outputs)
        for box, (dom_wires, cod_wires) in reversed(list(zip(
                hypergraph.boxes, hypergraph.box_wires))):
            expression = self.__box_to_term__(
                box, [variables[i] for i in dom_wires])
            bound = [variables[i] for i in cod_wires]
            result = expression if bound == [result]\
                else Let(expression, tuple(bound), result)
        return result

    @classmethod
    def __box_to_term__(cls, box, args):
        "The application of a box as a constant to variables for its inputs."
        if not box.dom:
            return cls.ob.constant_factory(box.name, box.cod)
        factors = [box.dom[i:i + 1] for i in range(len(box.dom))]
        exponent = box.dom if len(factors) == 1\
            else factors[0].product(*factors[1:])
        constant = cls.ob.constant_factory(box.name, exponent >> box.cod)
        return constant(args[0] if len(args) == 1 else Tuple(*args))


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


class Pack(Box):
    """
    The canonical isomorphism from the tensor of the factors of a
    :class:`Product` type to the product itself.

    Parameters:
        cod : The product type to pack into.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> assert Pack(X * Y).dom == X @ Y and Pack(X * Y).cod == X * Y
    >>> assert Pack(X * Y).dagger() == Unpack(X * Y)
    """
    def __init__(self, cod: Ty):
        if not cod.is_product:
            raise TypeError(f"Expected {Product}, got {cod!r}")
        dom = self.ob().tensor(*cod.factors)
        super().__init__(f"Pack({cod})", dom, cod)

    def dagger(self):
        return Unpack(self.cod)

    def __repr__(self):
        return factory_name(type(self)) + f"({self.cod!r})"


class Unpack(Box):
    """
    The canonical isomorphism from a :class:`Product` type to the tensor of
    its factors, i.e. the dagger of :class:`Pack`.

    Parameters:
        dom : The product type to unpack.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> assert Unpack(X * Y).dom == X * Y and Unpack(X * Y).cod == X @ Y
    >>> assert Unpack(X * Y).dagger() == Pack(X * Y)
    """
    def __init__(self, dom: Ty):
        if not dom.is_product:
            raise TypeError(f"Expected {Product}, got {dom!r}")
        cod = self.ob().tensor(*dom.factors)
        super().__init__(f"Unpack({dom})", dom, cod)

    def dagger(self):
        return Pack(self.dom)

    def __repr__(self):
        return factory_name(type(self)) + f"({self.dom!r})"


class Swap(markov.Swap, Box):
    "Symmetric swap in a closed diagram."


class Trace(markov.Trace, Box):
    "A trace in a closed category."


class Copy(markov.Copy, Box):
    "A markov copy in a closed category"

    is_linear = False


class Discard(markov.Discard, Copy):
    "A markov discard in a closed category."


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
    A closed functor is a markov functor that preserves evaluation, currying
    and packing. When the codomain has no products, i.e. its objects have no
    ``product`` method, the functor strictifies: a :class:`Product` is mapped
    to the tensor of its factors and :class:`Pack`, :class:`Unpack` to the
    identity. The exception is :class:`Drawing` where a product is drawn as
    a single wire and its packing as a box.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) : The codomain of the functor.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Product) and self.cod is not Drawing:
            if hasattr(self.cod.ob, "product"):
                return self.cod.ob(self.cod.ob.product_factory(
                    *map(self, other.factors)))
            return self(self.dom.ob().tensor(*other.factors))
        if isinstance(other, (Pack, Unpack)) and self.cod is not Drawing:
            typ = other.cod if isinstance(other, Pack) else other.dom
            if hasattr(self.cod.ob, "product"):
                return type(other)(self(typ))
            return self.cod.id(self(typ))
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
Diagram.discard_factory = Discard
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


type Term = Constant | Variable | Application | Abstraction\
    | Tuple | Projection | Let


class Constant(TermBase, biclosed.Constant):
    """
    A constant term prints as its bare name, so that terms read like
    textbook effectful lambda calculus and ``eval(str(term)) == term``
    under the obvious variable naming convention, e.g.
    ``query = (E >> E)("query")``.
    """
    def __str__(self):
        return self.name

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
        self.freevars = list(dict.fromkeys(func.freevars + args.freevars))
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
        if not self.func.freevars:
            func = self.func.eval(functor=functor)
            args = self.args.eval(functor=functor, context=context)
            return func @ args >> evaluate
        if not self.args.freevars:
            func = self.func.eval(functor=functor, context=context)
            args = self.args.eval(functor=functor)
            return func @ args >> evaluate
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


class Tuple(TermBase):
    """
    The tupling of terms, its codomain is the :class:`Product` of theirs.

    Parameters:
        terms : The terms inside the tuple.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> x, y = Variable("x", X), Variable("y", Y)
    >>> assert Tuple(x, y).cod == X * Y
    >>> assert Tuple(x, Tuple(y, x)).cod == X * (Y * X)
    """
    def __init__(self, *terms: Term):
        for term in terms:
            assert_isinstance(term, TermBase)
        self.terms = terms
        freevars = sum([term.freevars for term in terms], [])
        self.freevars = list(dict.fromkeys(freevars))
        self.overlap = len(freevars) != len(self.freevars)
        dom = self.ob().tensor(*[x.cod for x in self.freevars])
        cod = self.ob(self.ob.product_factory(*[t.cod for t in terms]))
        name = f"Tuple({', '.join(map(str, terms))})"
        super().__init__(name, dom, cod)

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        pack = functor(Pack(self.cod))
        identity = functor.cod.id(functor(self.ob()))
        splits = not self.overlap\
            and (context is None or self.freevars == context.inside)
        if splits:
            return identity.tensor(
                *[t.eval(functor=functor) for t in self.terms]) >> pack
        context = context or Context(self.freevars)
        terms = [t.eval(functor=functor, context=context)
                 for t in self.terms]
        return functor.cod.copy(functor(context.dom), len(terms))\
            >> identity.tensor(*terms) >> pack

    def __repr__(self):
        return factory_name(type(self))\
            + f"({', '.join(map(repr, self.terms))})"

    @property
    def constants(self):
        return sum([term.constants for term in self.terms], [])


class Projection(TermBase):
    """
    The projection onto one factor of a term with a :class:`Product` type,
    which evaluation interprets by discarding the other factors.

    Parameters:
        arg : The term to project from, with a product type as codomain.
        index : The index of the factor to project onto.

    Example
    -------
    >>> X, Y = Ty("X"), Ty("Y")
    >>> x, y = Variable("x", X), Variable("y", Y)
    >>> assert Projection(Tuple(x, y), 1).cod == Y
    """
    def __init__(self, arg: Term, index: int):
        assert_isinstance(arg, TermBase)
        assert_isinstance(index, int)
        if not arg.cod.is_product:
            raise TypeError(f"Expected {Product}, got {arg.cod!r}")
        if not 0 <= index < len(arg.cod.factors):
            raise IndexError(f"{arg.cod!r} has no factor {index}")
        self.arg, self.index = arg, index
        self.freevars = arg.freevars
        name = f"Projection({arg}, {index})"
        super().__init__(name, arg.dom, arg.cod.factors[index])

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        unpack = functor(Unpack(self.arg.cod))
        discards = functor.cod.id(functor(self.ob())).tensor(*[
            functor.cod.id(functor(typ)) if i == self.index
            else functor.cod.discard(functor(typ))
            for i, typ in enumerate(self.arg.cod.factors)])
        return self.arg.eval(functor=functor, context=context)\
            >> unpack >> discards

    def __repr__(self):
        return factory_name(type(self)) + f"({self.arg!r}, {self.index!r})"

    @property
    def constants(self):
        return self.arg.constants


class Let(TermBase):
    """
    The evaluation of an ``expression`` term, binding a tuple of
    ``variables`` to the factors of its result inside a ``body`` term,
    i.e. the statement ``let (x, ..., z) = expression in body``.

    Parameters:
        expression : The term that is evaluated.
        variables : The variables binding the factors of the result.
        body : The term in which the variables are bound.

    Note
    ----
    The codomain of ``expression`` unpacks either as the factors of its
    :class:`Product` type or as the tensor of the variables' types. Bound
    variables may be discarded or copied by the body, see :func:`let` for
    the introspection helper that builds the statement from a function.
    """
    def __init__(self, expression: Term, variables: tuple[Variable, ...],
                 body: Term):
        assert_isinstance(expression, TermBase)
        assert_isinstance(body, TermBase)
        variables = tuple(variables)
        for var in variables:
            assert_isinstance(var, Variable)
        if len(set(variables)) != len(variables):
            raise ValueError(f"Expected distinct variables, got {variables}")
        if set(variables).intersection(expression.freevars):
            raise ValueError(f"{variables} are free in {expression}")
        cods = [x.cod for x in variables]
        matched = list(expression.cod.factors) == cods\
            if expression.cod.is_product\
            else self.ob().tensor(*cods) == expression.cod
        if not matched:
            raise ValueError(
                f"Expected variables of type {expression.cod}, got {cods}")
        self.expression, self.variables, self.body\
            = expression, variables, body
        self.freevars = list(dict.fromkeys(expression.freevars + [
            x for x in body.freevars if x not in variables]))
        dom = self.ob().tensor(*[x.cod for x in self.freevars])
        params = ", ".join(x.name for x in variables)
        name = f"let({expression}, lambda {params}: {body})" if variables\
            else f"let({expression}, lambda: {body})"
        super().__init__(name, dom, body.cod)

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        unpack = functor(Unpack(self.expression.cod))\
            if self.expression.cod.is_product\
            else functor.cod.id(functor(self.expression.cod))
        shared = set(self.expression.freevars).intersection(
            self.body.freevars)
        if context is None and not shared:
            rest = [x for x in self.body.freevars
                    if x not in self.variables]
            expression = self.expression.eval(functor=functor)
            body = self.body.eval(functor=functor, context=Context(
                list(self.variables) + rest))
            identity = functor.cod.id(functor(
                self.ob().tensor(*[x.cod for x in rest])))
            return (expression >> unpack) @ identity >> body
        context = context or Context(self.freevars)
        expression = self.expression.eval(functor=functor, context=context)
        if not shared and all(x in self.expression.freevars
                              for x in context.inside):
            body = self.body.eval(
                functor=functor, context=Context(list(self.variables)))
            return expression >> unpack >> body
        body = self.body.eval(functor=functor, context=Context(
            list(self.variables) + context.inside))
        return functor.cod.copy(functor(context.dom))\
            >> (expression >> unpack) @ functor.cod.id(functor(context.dom))\
            >> body

    def __repr__(self):
        return factory_name(type(self)) + f"({self.expression!r}, "\
            + f"{self.variables!r}, {self.body!r})"

    @property
    def constants(self):
        return self.expression.constants + self.body.constants


def let(expression: Term, body: Callable) -> Let:
    """
    Bind the result of an ``expression`` term inside the ``body`` of a
    Python function, whose variable names are given by introspection and
    whose types are the factors of the codomain of ``expression``.

    Parameters:
        expression : The term that is evaluated.
        body : A function from the bound variables to a term.

    Example
    -------
    The term for the self-attention block of the CatGPT benchmark, where
    an embedded token is packed into a query, key and value before
    attention and a feed-forward layer are applied:

    >>> E = Ty("E")
    >>> query, key, value = [(E >> E)(name) for name in (
    ...     "query", "key", "value")]
    >>> attention = (E.product(E, E) >> E)("attention")
    >>> feed_forward = (E >> E)("feed_forward")
    >>> block = E(lambda x: let(Tuple(query(x), key(x), value(x)),
    ...     lambda q, k, v: let(attention(Tuple(q, k, v)),
    ...         lambda a: feed_forward(a))))
    >>> assert block.cod == E >> E
    >>> block.draw(doctest="docs/_static/closed/catgpt-block.svg",
    ...     aspect="auto", figsize=(6, 8), margins=(0.2, 0))

    .. image:: /_static/closed/catgpt-block.svg
        :align: center
    """
    varnames = list(signature(body).parameters)
    cod = expression.cod
    factors = list(cod.factors) if cod.is_product\
        else [cod[i:i + 1] for i in range(len(cod))]
    if len(varnames) != len(factors):
        raise ValueError(
            f"Expected {len(factors)} variables, got {len(varnames)}")
    variables = tuple(typ.variable_factory(name, typ)
                      for name, typ in zip(varnames, factors))
    return Let(expression, variables, body(*variables))


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

    def without(self, *variables: Variable) -> Substitution:
        "The restriction of a substitution away from bound ``variables``."
        return type(self)({k: v for k, v in self.inside.items()
                           if k not in variables})

    def __call__(self, term: Term) -> Term:
        if isinstance(term, Variable):
            return self.inside.get(term, term)
        if isinstance(term, Application):
            return type(term)(self(term.func), self(term.args), term.left)
        if isinstance(term, Abstraction):
            return type(term)(
                term.var, self.without(term.var)(term.body), term.left)
        if isinstance(term, Tuple):
            return type(term)(*map(self, term.terms))
        if isinstance(term, Projection):
            return type(term)(self(term.arg), term.index)
        if isinstance(term, Let):
            return type(term)(self(term.expression), term.variables,
                              self.without(*term.variables)(term.body))
        return term


Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction
Ty.product_factory = Product


class Equation(markov.Equation):
    """ The :class:`markov.Equation` of closed diagrams. """
