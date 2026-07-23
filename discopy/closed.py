
"""
The free closed markov category, i.e. with copy, discard and exponentials.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ty
    Unitype
    Exp
    TermBase
    Constant
    Variable
    Application
    Abstraction
    Substitution
    BohmTree
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
from typing import Callable, Dict, ClassVar

from discopy import cat, monoidal, biclosed, markov, symmetric, hypergraph
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


class Unitype(Ty):
    """
    A closed type which is its own exponential, used to type terms that are
    not simply typed, e.g. the exponentiation of Church numerals.

    Parameters:
        name : The name of the unitype, ``"o"`` by default.

    Note
    ----
    A unitype compares equal to the ordinary atomic type with the same name,
    but only the unitype short-circuits exponentiation: do not reuse the name
    of a unitype for an ordinary atom.

    Example
    -------
    >>> o = Unitype()
    >>> assert o >> o == o == o << o and o.base == o.exponent == o
    >>> two = o(lambda f: o(lambda x: f(f(x))))
    >>> assert two.cod == o and two(two).cod == o
    """
    def __init__(self, name: str = "o"):
        super().__init__(name)

    is_exp = property(lambda self: True)
    base = exponent = property(lambda self: self)

    def exp(self, other: Ty) -> Ty:
        return self if other == self else super().exp(other)

    over = under = exp


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


class Discard(markov.Discard, Copy):
    "The discard of an atomic type in a closed diagram."


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

    def to_map(self) -> symmetric.CMap:
        """
        Encode an almost-linear lambda term as a rooted map over a single
        generating object. On pure linear terms this is one direction of
        Zeilberger's isomorphism with rooted trivalent maps, and the inverse
        is :meth:`discopy.cmap.CMap.to_term`.

        Each application becomes a node with the function and argument
        subtrees as inputs and the result as output, each abstraction becomes
        a node plugging the root and the wire of the abstracted variable, see
        :meth:`discopy.cmap.CMap.plug_input`. A variable with several
        occurrences goes through one delta node with one port per occurrence,
        see :meth:`discopy.cmap.CMap.merge_inputs`, and a constant becomes a
        node with a single port; :meth:`discopy.cmap.CMap.to_term` does not
        support these two node kinds. The free variables of the term are the
        inputs of the map and the root is its output. The names of the
        variables are attached to the objects of their wires, see
        :func:`biclosed.annotate`, so that the round-trip from term to map
        and back is faithful on the nose.

        Example
        -------
        >>> a, b = Ty("a"), Ty("b")
        >>> term = (a >> b)(lambda f: a(lambda v: f(v)))
        >>> cmap = term.to_map()
        >>> len(cmap.boxes)
        3
        >>> assert cmap.to_term() == term

        >>> o = Unitype()
        >>> two = o(lambda f: o(lambda x: f(f(x))))
        >>> sorted(box.name for box in two.to_map().boxes)
        ['@', '@', 'δ', 'λ', 'λ']
        """
        x = symmetric.Ty("x")
        application_box = symmetric.Box("@", x @ x, x)

        def merge(cmap, variables, var):
            """ One delta node with a port for each occurrence of var. """
            indices = tuple(
                i for i, other in enumerate(variables) if other == var)
            if len(indices) == 1:
                return cmap, variables
            wire = biclosed.annotate(x, var.name)
            delta = symmetric.Box("δ", wire, wire ** len(indices))
            variables = [v for v in variables if v != var]
            variables.insert(indices[0], var)
            return cmap.merge_inputs(indices, delta), variables

        def go(term):
            if isinstance(term, Constant):
                return symmetric.CMap.from_box(
                    symmetric.Box(term.name, symmetric.Ty(), x)), []
            if isinstance(term, Variable):
                wire = biclosed.annotate(x, term.name)
                return symmetric.CMap.id(wire), [term]
            if isinstance(term, Application):
                func, func_vars = go(term.func)
                args, args_vars = go(term.args)
                return func @ args >> symmetric.CMap.from_box(
                    application_box), func_vars + args_vars
            if isinstance(term, Abstraction):
                body, body_vars = go(term.body)
                if term.var not in body_vars:
                    raise ValueError(
                        "Expected an almost-linear term where every "
                        f"abstracted variable occurs, got {self}.")
                body, body_vars = merge(body, body_vars, term.var)
                index = body_vars.index(term.var)
                abstraction_box = symmetric.Box(
                    "λ", x, x @ biclosed.annotate(x, term.var.name))
                remaining = body_vars[:index] + body_vars[index + 1:]
                return body.plug_input(index, abstraction_box, x), remaining
            raise NotImplementedError(
                f"Expected an almost-linear term, got {term}.")

        cmap, variables = go(self)
        for var in dict.fromkeys(variables):
            cmap, variables = merge(cmap, variables, var)
        return cmap


type Term = Constant | Variable | Application | Abstraction


class Constant(TermBase, biclosed.Constant):
    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        if not context:
            return super().eval(functor)
        return functor.cod.discard(context.image(functor)) >> super().eval(
            functor)


class Variable(TermBase, biclosed.Variable):
    def eval(self, functor=None, context=None):
        functor = functor or self.functor

        def annotated(x):
            typ = functor(x.cod)
            return biclosed.annotate(typ, x.name)\
                if isinstance(typ, monoidal.Ty) else typ

        if not context:
            return functor.cod.id(annotated(self))
        return functor.cod.tensor(*[
            functor.cod.id(annotated(x)) if x == self
            else functor.cod.discard(annotated(x))
            for x in context.inside])


class Application(TermBase, biclosed.Application):
    def __check_dom__(self, func, args, left):
        self.overlap = set(func.freevars).intersection(args.freevars)
        self.freevars = list(dict.fromkeys(func.freevars + args.freevars))\
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
        return functor.cod.copy(context.image(functor))\
            >> func @ args >> evaluate


class Abstraction(TermBase, biclosed.Abstraction):
    def __check_dom__(self):
        self.freevars = [x for x in self.body.freevars if x != self.var]
        return self.ob().tensor(*[x.cod for x in self.freevars])

    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        n = len(functor(self.var.cod))
        if context:
            new_context = Context([self.var] + context.inside)
            body = self.body.eval(functor=functor, context=new_context)
            return body.curry(n)
        i = self.body.freevars.index(self.var)
        offset = sum(
            len(functor(x.cod)) for x in self.body.freevars[:i])
        body = self.body.eval(functor=functor)
        p = list(range(offset, offset + n)) + [
            j for j in range(len(body.dom)) if not offset <= j < offset + n]
        return (body.permutation(p, body.dom).dagger() >> body).curry(n)


@dataclass
class Context:
    inside: list[Variable]
    category: ClassVar[type[ClosedCategory]] = Diagram

    @property
    def dom(self):
        return self.category.ob.tensor(*[x.cod for x in self.inside])

    def image(self, functor) -> monoidal.Ty:
        """
        The image of :attr:`dom` under a functor, with the name of each
        variable attached to the objects of its wires, see
        :func:`biclosed.annotate`.
        """
        segments = [functor(x.cod) for x in self.inside]
        if all(isinstance(seg, monoidal.Ty) for seg in segments):
            segments = [
                biclosed.annotate(seg, x.name)
                for seg, x in zip(segments, self.inside)]
            return segments[0].tensor(*segments[1:]) if segments\
                else functor(self.dom)
        return functor(self.dom)


@dataclass
class Substitution:
    """
    The simultaneous substitution of terms for free variables.

    Substitution is capture-avoiding: a binder is renamed to a fresh name
    only when a substituted term has its variable free, so that names are
    preserved whenever capture cannot happen.

    Parameters:
        inside : The mapping from variables to the substituted terms.

    Example
    -------
    >>> X = Ty("X")
    >>> u, v = Variable("u", X), Variable("v", X)
    >>> assert Substitution({u: v})(X(lambda w: u)) == X(lambda w: v)
    >>> renamed = Substitution({u: v})(X(lambda v: u))
    >>> assert renamed.body == v and renamed.var != v
    """
    inside: Dict[Variable, Term]

    def __call__(self, term: Term) -> Term:
        if isinstance(term, Variable):
            return self.inside.get(term, term)
        if isinstance(term, Application):
            return type(term)(self(term.func), self(term.args), term.left)
        if isinstance(term, Abstraction):
            inside = {
                key: value for key, value in self.inside.items()
                if key != term.var}
            if not inside:
                return term
            var = term.var
            if any(var in value.freevars for value in inside.values()):
                var = type(term.var)(biclosed.fresh_name(), term.var.cod)
                inside[term.var] = var
            return type(term)(var, Substitution(inside)(term.body), term.left)
        return term


@dataclass
class BohmTree:
    """
    The head normal form of a term: abstracted ``variables`` over a head
    variable applied to ``args``, with ``None`` for the unexpanded holes of
    an incomplete tree.

    Parameters:
        cod : The type of the subterm at this node.
        variables : The variables abstracted at this node.
        head : The index of the head variable in the scope, i.e. the free
               variables followed by the variables bound from the root down
               to and including this node, with shadowed names resolving to
               the innermost binder.
        args : The subtrees the head is applied to, ``None`` for holes.

    Note
    ----
    Heads are de Bruijn levels, so trees compare equal up to alpha
    equivalence: the names of the variables are preserved as much as
    possible by :meth:`to_term` but they do not affect equality.

    Example
    -------
    >>> o = Unitype()
    >>> two = o(lambda f: o(lambda x: f(f(x))))
    >>> tree = BohmTree.from_term(two(two))
    >>> assert tree == BohmTree.from_term(
    ...     o(lambda f: o(lambda x: f(f(f(f(x)))))))
    >>> assert BohmTree.from_term(tree.to_term()) == tree
    >>> assert BohmTree.from_term(two(two), budget=0) is None
    """
    cod: Ty
    variables: tuple[Variable, ...]
    head: int
    args: tuple[BohmTree | None, ...]

    def __eq__(self, other):
        return isinstance(other, BohmTree) and self.head == other.head\
            and self.cod == other.cod and self.args == other.args\
            and tuple(x.cod for x in self.variables)\
            == tuple(x.cod for x in other.variables)

    def __hash__(self):
        return hash((self.cod, len(self.variables), self.head, self.args))

    @staticmethod
    def step(term: Term) -> Term | None:
        """
        One step of leftmost-outermost head beta reduction, or ``None`` if
        the term is in head normal form. The extension point for other
        reduction strategies, see :meth:`from_term`.

        Parameters:
            term : The term to reduce.
        """
        if isinstance(term, Abstraction):
            body = BohmTree.step(term.body)
            return None if body is None\
                else type(term)(term.var, body, term.left)
        spine, head = [], term
        while isinstance(head, Application):
            spine.append(head)
            head = head.func
        if not isinstance(head, Abstraction) or not spine:
            return None
        redex = spine.pop()
        result = Substitution({head.var: redex.args})(head.body)
        for application in reversed(spine):
            result = type(application)(
                result, application.args, application.left)
        return result

    @classmethod
    def from_term(cls, term: Term, budget: int | None = None,
                  step: Callable = None, scope: tuple = ()) -> BohmTree | None:
        """
        The Böhm tree of a term, reduced with at most ``budget`` beta steps:
        the nodes that cannot be reached within the budget are ``None``.

        Parameters:
            term : The term to normalise.
            budget : The number of beta steps allowed, unbounded by default.
            step : The reduction strategy, :meth:`step` by default.
            scope : The free variables of the term.
        """
        counter = [budget]
        return cls._expand(term, tuple(scope), counter, step or cls.step)

    @classmethod
    def _expand(cls, term, scope, counter, step):
        """ Build one node, spending beta steps from the shared counter. """
        while (reduced := step(term)) is not None:
            if counter[0] is not None:
                if counter[0] == 0:
                    return None
                counter[0] -= 1
            term = reduced
        cod, variables = term.cod, []
        while isinstance(term, Abstraction):
            variables.append(term.var)
            term = term.body
        scope, spine = scope + tuple(variables), []
        while isinstance(term, Application):
            spine.append(term.args)
            term = term.func
        if not isinstance(term, Variable):
            raise NotImplementedError(
                f"Expected a variable head, got {term}.")
        head = len(scope) - 1 - scope[::-1].index(term)
        return cls(cod, tuple(variables), head, tuple(
            cls._expand(arg, scope, counter, step)
            for arg in reversed(spine)))

    def to_term(self, scope: tuple = ()) -> Term:
        """
        The term of a complete Böhm tree, so that normalisation is idempotent.

        Parameters:
            scope : The free variables of the term.

        Raises:
            ValueError : If the tree has a hole.
        """
        scope = tuple(scope) + self.variables
        term = scope[self.head]
        for arg in self.args:
            if arg is None:
                raise ValueError(f"{self} has a hole.")
            term = term(arg.to_term(scope))
        for var in reversed(self.variables):
            term = Abstraction(var, term)
        return term


Ty.variable_factory = Variable
Ty.constant_factory = Constant
Ty.application_factory = Application
Ty.abstraction_factory = Abstraction


class Equation(markov.Equation):
    """ The :class:`markov.Equation` of closed diagrams. """
