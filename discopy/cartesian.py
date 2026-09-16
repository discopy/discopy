# -*- coding: utf-8 -*-

"""
The free cartesian category, i.e. markov diagrams where every morphism is
deterministic.

For now the free diagrams are the same as :mod:`discopy.markov`: the
naturality of copy — ``f >> Diagram.copy(f.cod) == Diagram.copy(f.dom)
>> f @ f`` — is the axiom that distinguishes
:class:`discopy.abc.CartesianCategory`, to be checked with property
testing. The main example is :class:`discopy.python.Function`.

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

from discopy import markov, cmap, hypergraph
from discopy.abc import CartesianCategory
from discopy.cat import factory
from discopy.monoidal import Ty  # noqa: F401


@factory
class Diagram(markov.Diagram, CartesianCategory):
    """
    A cartesian diagram is a markov diagram whose boxes are read as
    deterministic morphisms.

    Parameters:
        inside(Layer) : The layers inside the diagram.
        dom (monoidal.Ty) : The domain of the diagram, i.e. its input.
        cod (monoidal.Ty) : The codomain of the diagram, i.e. its output.
    """


class Box(markov.Box, Diagram):
    """
    A cartesian box is a markov box in a cartesian diagram.

    Parameters:
        name (str) : The name of the box.
        dom (monoidal.Ty) : The domain of the box, i.e. its input.
        cod (monoidal.Ty) : The codomain of the box, i.e. its output.
    """


class Permutation(markov.Permutation, Box):
    "A permutation in a cartesian diagram."


class Swap(Permutation, markov.Swap, Box):
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


class Sum(markov.Sum, Box):
    """
    A cartesian sum is a markov sum in a cartesian diagram.

    Parameters:
        terms (tuple[Diagram, ...]) : The terms of the formal sum.
        dom (Ty) : The domain of the formal sum.
        cod (Ty) : The codomain of the formal sum.
    """


class Functor(markov.Functor):
    """
    A cartesian functor is a markov functor between cartesian categories.

    Parameters:
        ob_map (Mapping[monoidal.Ty, monoidal.Ty]) :
            Map from :class:`monoidal.Ty` to :code:`cod.ob`.
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
    "A constant applied to terms in a cartesian category."


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
Diagram.sum_factory = Sum
TermBase.functor = Functor.id(Diagram)
TermBase.application_factory = Application
Id = Diagram.id


class Equation(markov.Equation):
    """ The :class:`markov.Equation` of cartesian diagrams. """
    up_to = staticmethod(Diagram.to_hypergraph)
