# -*- coding: utf-8 -*-

"""
Reverse derivatives of neural diagrams, as optics.

A reverse rule for a box ``f : A -> B`` is an :class:`~discopy.optics.Optic`
over neural diagrams: a residual ``M``, a forward leg ``A -> B @ M``
computing ``f`` and storing what the backward leg needs, and a backward leg
``M @ B -> A`` taking the residual and a cotangent on ``B`` to a cotangent
on ``A``.  Composition and tensor route the residuals as optics do, and
:func:`differentiate` is the functorial fold of the rules over the layers
of a diagram, identities and swaps being structural.  This is the reverse
derivative category read as optics, the semantics of backpropagation of
:cite:t:`CruttwellEtAl22`; the reverse derivative ``A @ B -> A`` of
:func:`rdiff` is the ``put`` of its lens, discarding the primal output
before the backward leg.

Only monogamous acyclic hypergraphs are accepted, i.e. feed-forward
networks.  Identity wires and permutations have structural rules; every
other generator needs an explicit rule, the dagger of a box included.
This keeps residuals in the diagram rather than in an autograd tape or a
module cache.

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
        generator_rule
        differentiate
        discard
        rdiff

Example
-------

>>> from discopy.neural import Dim, Network
>>> x, y, m = Dim(2), Dim(3), Dim(5)
>>> rule = reverse_rule(Network("f", x, y @ m), Network("f'", m @ y, x), m)
>>> rule.dom, rule.cod, rule.residual
(optics.Ty[neural.core.Dim](positive=Dim(2), negative=Dim(2)), \
optics.Ty[neural.core.Dim](positive=Dim(3), negative=Dim(3)), Dim(5))
>>> zero = lambda typ: Network("Discard", typ, Dim())
>>> derivative = rdiff(Network("f", x, y).to_hypergraph(),
...                    {Network("f", x, y): rule}, discard_factory=zero)
>>> assert derivative == rule.to_lens(zero).put
>>> derivative.dom, derivative.cod
(Dim(2, 3), Dim(2))

The derivative is a neural diagram, which runs as a map:

>>> derivative.to_map().boxes  # doctest: +NORMALIZE_WHITESPACE
(neural.core.Network('f', Dim(2), Dim(3, 5)),
 neural.core.Network('Discard', Dim(3), Dim(0)),
 neural.core.Network("f'", Dim(5, 3), Dim(2)))
"""

from __future__ import annotations

from discopy import optics
from discopy.neural.backend import get_backend
from discopy.neural.core import (
    Diagram, Dim, Hypergraph, Network, Permutation)
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
    The reverse rule with a forward leg ``A -> B @ residual`` and a
    backward leg ``residual @ B -> A``.

    Parameters:
        forward : The forward leg.
        backward : The backward leg.
        residual : What the forward leg stores for the backward one.
    """
    dom = forward.dom
    cod = forward.cod[:len(forward.cod) - len(residual)]
    return ReverseRule(pair(dom), pair(cod), forward, backward, residual)


def generator_rule(box: Network, rules) -> ReverseRule:
    """
    The reverse rule of one generator: the structural optic of a
    permutation, otherwise the rule looked up in ``rules`` and checked to
    go between the pairs of the domain and codomain of the box.

    Parameters:
        box : The generator.
        rules : A mapping or callable from generators to reverse rules.
    """
    if isinstance(box, Permutation):
        return ReverseRule.permutation(
            list(box.perm), [pair(atom) for atom in box.dom])
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
    The reverse rule of a monogamous acyclic neural hypergraph, i.e. a
    feed-forward network: the rules of its generators folded over its
    layers in topological order, identities and swaps being the structural
    optics.

    Parameters:
        graph : The hypergraph to differentiate.
        rules : A mapping or callable from generators to reverse rules.
    """
    assert_isinstance(graph, Hypergraph)
    if not graph.is_monogamous:
        raise ValueError("Reverse differentiation requires monogamy.")
    if not graph.is_acyclic:
        raise ValueError("Reverse differentiation requires an acyclic graph.")
    rules = MappingOrCallable(rules)
    result = ReverseRule.id(pair(graph.dom))
    layers = graph.topological_order().to_diagram().to_staircases().inside
    for layer in layers:
        left, box, right = layer.boxes_and_types
        result >>= ReverseRule.id(pair(left))\
            @ generator_rule(box, rules)\
            @ ReverseRule.id(pair(right))
    return result


def discard(typ: Dim) -> Diagram:
    """
    The all-port-zero discard network, its module supplied by the current
    backend, or the identity on the unit, as the discard of a Markov
    category is.

    Parameters:
        typ : The dimension to discard.
    """
    assert_isinstance(typ, Dim)
    if not typ:
        return Diagram.id(typ)
    return Network("Discard", typ, Dim(), module=get_backend().zeros_module())


def rdiff(graph: Hypergraph, rules, discard_factory=discard) -> Diagram:
    """
    The reverse derivative ``A @ B -> A`` of ``graph : A -> B``, the ``put``
    of the lens of its reverse rule: the forward leg beside the cotangent,
    the primal output discarded, then the backward leg. It is a neural
    diagram, which :meth:`Diagram.to_map <discopy.neural.core.Diagram.to_map>`
    runs with ``causal=True``, every box firing once in topological order.

    Parameters:
        graph : The hypergraph to differentiate.
        rules : A mapping or callable from generators to reverse rules.
        discard_factory : A function from a dimension ``B`` to a diagram
                          ``B -> Dim()``, the zero network of the current
                          backend by default.
    """
    return differentiate(graph, rules).to_lens(discard_factory).put
