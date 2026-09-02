# -*- coding: utf-8 -*-

"""
Reverse derivatives of neural diagrams, as optics.

A reverse rule for a box ``f : A -> B`` is an :class:`~discopy.optics.Optic`
over neural diagrams: a residual ``M``, a forward leg ``A -> M @ B``
computing ``f`` and storing what the backward leg needs, and a backward leg
``M @ B -> A`` taking the residual and a cotangent on ``B`` to a cotangent
on ``A``.  Composition tensors the residuals and tensor swaps them past the
outputs, as optics do, and :func:`differentiate` is the functorial fold of
the rules over the layers of a diagram, identities and swaps being
structural.  This is the reverse derivative category read as optics, the
semantics of backpropagation of :cite:t:`CruttwellEtAl22`; the reverse
derivative ``A @ B -> A`` of :func:`rdiff` is the ``put`` of its lens,
discarding the primal output before the backward leg.

Only causal monogamous hypergraphs are accepted.  Identity wires and swaps
have structural rules; every other generator needs an explicit rule.  This
keeps residuals in the diagram rather than in an autograd tape or a module
cache.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    ReverseRule

.. admonition:: Functions

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        reverse_rule
        differentiate
        discard
        rdiff

Example
-------

>>> from discopy.neural import Dim, Network
>>> x, y, m = Dim(2), Dim(3), Dim(5)
>>> rule = reverse_rule(Network("f", x, m @ y), Network("f'", m @ y, x), m)
>>> rule.dom, rule.cod, rule.residual
(optics.Ty[neural.core.Dim](positive=Dim(2), negative=Dim(2)), \
optics.Ty[neural.core.Dim](positive=Dim(3), negative=Dim(3)), Dim(5))
>>> zero = lambda typ: Network("Discard", typ, Dim())
>>> derivative = rdiff(Network("f", x, y).to_hypergraph(),
...                    {Network("f", x, y): rule}, discard_factory=zero)
>>> derivative.dom, derivative.cod
(Dim(2, 3), Dim(2))
"""

from __future__ import annotations

from discopy import optics
from discopy.neural.backend import get_backend
from discopy.neural.core import Diagram, Dim, Hypergraph, Network, Swap
from discopy.utils import MappingOrCallable, assert_isinstance

#: A reverse rule is an optic over neural diagrams between pairs ``(A, A)``.
ReverseRule = optics.Optic[Diagram]

#: The pairs of dimensions the rules go between.
Pair = ReverseRule.ob


def pair(dim: Dim) -> Pair:
    """ The pair ``(dim, dim)`` of a dimension and its cotangent. """
    return Pair(dim, dim)


def reverse_rule(forward: Diagram, backward: Diagram,
                 residual: Dim = Dim()) -> ReverseRule:
    """
    The reverse rule with a forward leg ``A -> residual @ B`` and a
    backward leg ``residual @ B -> A``.

    Parameters:
        forward : The forward leg.
        backward : The backward leg.
        residual : What the forward leg stores for the backward one.
    """
    dom, cod = forward.dom, forward.cod[len(residual):]
    return ReverseRule(pair(dom), pair(cod), forward, backward, residual)


def _generator_rule(box, rules) -> ReverseRule:
    """ Look up and type-check the reverse rule for one generator. """
    if isinstance(box, Swap):
        return ReverseRule.swap(pair(box.dom[:1]), pair(box.dom[1:]))
    try:
        rule = rules[box]
    except KeyError as exception:
        raise ValueError(
            f"Missing reverse rule for generator {box!r}.") from exception
    assert_isinstance(rule, ReverseRule)
    if (rule.dom, rule.cod) != (pair(box.dom), pair(box.cod)):
        raise ValueError(
            f"Expected a rule from {pair(box.dom)} to {pair(box.cod)}, "
            f"got {rule.dom} to {rule.cod}.")
    return rule


def differentiate(graph: Hypergraph, rules) -> ReverseRule:
    """
    The reverse rule of a causal monogamous neural hypergraph: the rules of
    its generators folded over its layers, identities and swaps being the
    structural optics.

    Parameters:
        graph : The hypergraph to differentiate.
        rules : A mapping or callable from generators to reverse rules.
    """
    assert_isinstance(graph, Hypergraph)
    if not graph.is_monogamous:
        raise ValueError("Reverse differentiation requires monogamy.")
    if not graph.is_causal:
        raise ValueError("Reverse differentiation requires causality.")
    rules = MappingOrCallable(rules)
    result = ReverseRule.id(pair(graph.dom))
    for layer in graph.to_diagram().to_staircases().inside:
        left, box, right = layer.boxes_and_types
        result >>= ReverseRule.id(pair(left))\
            @ _generator_rule(box, rules)\
            @ ReverseRule.id(pair(right))
    return result


def discard(typ: Dim) -> Network:
    """
    The all-port-zero discard network, its module supplied by the current
    backend.

    Parameters:
        typ : The dimension to discard.
    """
    assert_isinstance(typ, Dim)
    return Network("Discard", typ, Dim(), module=get_backend().zeros_module())


def rdiff(graph: Hypergraph, rules, discard_factory=discard) -> Hypergraph:
    """
    The reverse derivative ``A @ B -> A`` of ``graph : A -> B``: the forward
    leg beside the cotangent, the primal output discarded, then the
    backward leg -- the ``put`` of the rule's lens.

    Parameters:
        graph : The hypergraph to differentiate.
        rules : A mapping or callable from generators to reverse rules.
        discard_factory : A function from a dimension ``B`` to a diagram
                          ``B -> Dim()``, e.g. a backend-specific discard.
    """
    rule = differentiate(graph, rules)
    dropped = discard_factory(graph.cod)
    assert_isinstance(dropped, Diagram)
    if dropped.dom != graph.cod or dropped.cod != Dim():
        raise ValueError(
            "The discard factory must return a diagram B -> Dim().")
    diagram = rule.forward @ graph.cod\
        >> rule.residual @ dropped @ graph.cod >> rule.backward
    return diagram.to_hypergraph()
