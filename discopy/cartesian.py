# -*- coding: utf-8 -*-

"""
The free cartesian closed category, i.e. closed diagrams with a supply of
copy where every morphism is deterministic.

The naturality of copy — ``f >> Diagram.copy(f.cod) == Diagram.copy(f.dom)
>> f @ f`` — is the axiom that distinguishes
:class:`discopy.abc.CartesianCategory` from a Markov category, to be
checked once property testing reaches this level of the hierarchy. The
main example is :class:`discopy.python.Function`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Box
    Swap
    Permutation
    Copy
    Merge
    Discard
    Eval
    Coeval
    Curry
    Functor
    Constant
    Variable
    Application

Example
-------

>>> X, Y = Ty('X'), Ty('Y')
>>> x, f = Variable('x', X), Constant('f', X @ X, Y)
>>> assert f(x, x).eval() == Copy(X) >> f
>>> assert isinstance(f(x, x).eval(), Diagram)
"""

from __future__ import annotations

from discopy import closed, markov, cmap, hypergraph
from discopy.abc import CartesianClosedCategory
from discopy.cat import factory
from discopy.closed import Ty  # noqa: F401


@factory
class Diagram(markov.Diagram, closed.Diagram, CartesianClosedCategory):
    """
    A cartesian diagram is a markov diagram which is also closed, whose
    boxes are read as deterministic morphisms.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (closed.Ty) : The domain of the diagram, i.e. its input.
        cod (closed.Ty) : The codomain of the diagram, i.e. its output.
    """
    ob = Ty


class Box(markov.Box, closed.Box, Diagram):
    """
    A cartesian box is a markov and closed box in a cartesian diagram.

    Parameters:
        name (str) : The name of the box.
        dom (closed.Ty) : The domain of the box, i.e. its input.
        cod (closed.Ty) : The codomain of the box, i.e. its output.
    """


class Permutation(markov.Permutation, closed.Permutation, Box):
    "A permutation in a cartesian diagram."


class Swap(Permutation, markov.Swap, closed.Swap, Box):
    "A swap in a cartesian diagram."


class Copy(markov.Copy, Box):
    "A copy in a cartesian diagram."

    def dagger(self) -> Merge:
        return Merge(self.dom, len(self.cod))


class Merge(markov.Merge, Box):
    "A merge in a cartesian diagram."

    def dagger(self) -> Copy:
        return Copy(self.cod, len(self.dom))


class Discard(markov.Discard, Copy):
    "A discard in a cartesian diagram."


class Eval(closed.Eval, Box):
    "The evaluation of an exponential type in a cartesian diagram."


class Coeval(closed.Coeval, Box):
    "The coevaluation of an exponential type in a cartesian diagram."


class Curry(closed.Curry, Box):
    "The currying of a cartesian diagram."


class Sum(markov.Sum, closed.Sum, Box):
    """
    A cartesian sum is a markov and closed sum in a cartesian diagram.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(markov.Functor, closed.Functor):
    """
    A cartesian functor is a markov and closed functor between cartesian
    closed categories.

    Parameters:
        ob_map (Mapping[Ty, Ty]) : Map from :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod (Category) :
            The codomain, :code:`Diagram` by default.
    """
    dom = cod = Diagram


class TermBase(markov.TermBase, Box):
    "A term in the internal language of a cartesian category."


class Constant(markov.Constant, TermBase):
    "A function symbol in a cartesian category."


class Variable(markov.Variable, TermBase):
    "A variable in a cartesian category."


class Application(markov.Application, TermBase):
    """
    A constant applied to terms in a cartesian category: by naturality of
    copy, a repeated subterm is evaluated once and its result copied,
    where :class:`markov.Application` evaluates it once per occurrence.

    Example
    -------
    >>> X, Y, Z = Ty('X'), Ty('Y'), Ty('Z')
    >>> x = Variable('x', X)
    >>> g, f = Constant('g', X, Y), Constant('f', Y @ Y, Z)
    >>> assert f(g(x), g(x)).eval() == g >> Copy(Y) >> f

    whereas the markov evaluation samples ``g`` once per occurrence:

    >>> from discopy import markov
    >>> mX, mY, mZ = map(markov.Ty, "XYZ")
    >>> mx = markov.Variable('x', mX)
    >>> mg = markov.Constant('g', mX, mY)
    >>> mf = markov.Constant('f', mY @ mY, mZ)
    >>> assert mf(mg(mx), mg(mx)).eval() == (
    ...     markov.Copy(mX) >> mg @ mg >> mf)
    """
    def eval(self, functor=None, context=None):
        functor = functor or self.functor
        distinct = list(dict.fromkeys(self.args))
        if len(distinct) == len(self.args):
            return super().eval(functor=functor, context=context)
        context = Context(self.freevars) if context is None else context
        copies = functor.cod.id(functor(context.dom))\
            if len(distinct) == 1\
            else functor.cod.copy(functor(context.dom), len(distinct))
        wiring = functor.cod.id(functor(self.ob()))
        for term in distinct:
            wiring = wiring @ term.eval(functor=functor, context=context)
        results = functor.cod.id(functor(self.ob()))
        for term in distinct:
            results = results @ (
                functor.cod.id(functor(term.cod))
                if self.args.count(term) == 1 else functor.cod.copy(
                    functor(term.cod), self.args.count(term)))
        offsets, doms, perm = {}, [], []
        for term in distinct:
            for _ in range(self.args.count(term)):
                offsets[term] = offsets.get(term, len(doms))
                doms += [functor(self.ob(wire)) for wire in term.cod.inside]
        seen = {term: 0 for term in distinct}
        for term in self.args:
            start = offsets[term] + seen[term] * len(term.cod)
            perm += list(range(start, start + len(term.cod)))
            seen[term] += 1
        reorder = functor.cod.permutation(perm, doms)
        return copies >> wiring >> results >> reorder\
            >> functor(self.symbol)


class Context(markov.Context):
    "A context of cartesian variables."
    category = Diagram


type Term = Constant | Variable | Application


CMap = cmap.CMap[Diagram]
Hypergraph = hypergraph.Hypergraph[Diagram]

Diagram.functor_factory = Functor
Diagram.copy_factory, Diagram.merge_factory = Copy, Merge
Diagram.swap_factory = Swap
Diagram.permutation_factory = Permutation
Diagram.discard_factory = Discard
Diagram.curry_factory = Curry
Diagram.eval_factory = Eval
Diagram.coeval_factory = Coeval
Diagram.sum_factory = Sum
TermBase.functor = Functor.id(Diagram)
TermBase.application_factory = Application
Id = Diagram.id


class Equation(markov.Equation):
    """ The :class:`markov.Equation` of cartesian diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)
