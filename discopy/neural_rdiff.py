# -*- coding: utf-8 -*-

"""
Reverse derivatives of causal neural hypergraphs.

A reverse rule for an arrow ``f : A -> B`` factors its evaluation through a
residual ``M``::

    forward : A -> B @ M
    reverse : M @ B -> A

Composition stores the residuals in forward order. Thus, if ``f`` has
residual ``M`` and ``g`` has residual ``N``, then ``f >> g`` has residual
``M @ N``. Tensor products use the same convention. The resulting reverse
derivative has type ``A @ B -> A`` and discards the primal output before
applying ``reverse``.

Only causal monogamous hypergraphs are accepted. Identity wires and swaps
have structural rules; every other generator needs an explicit rule. This
keeps residuals in the diagram rather than in an autograd tape or a module
cache.
"""

from __future__ import annotations

from discopy import neural
from discopy.utils import MappingOrCallable, assert_isinstance


class ReverseRule:
    """
    A forward diagram paired with its residual-consuming reverse diagram.

    Parameters:
        forward : A diagram of type ``A -> B @ M``.
        reverse : A diagram of type ``M @ B -> A``.
        cod : The type ``B``. It is inferred when omitted; passing it removes
              ambiguity when ``B @ M == M @ B``.
    """

    def __init__(
            self, forward: neural.Diagram, reverse: neural.Diagram,
            cod: neural.Dim = None):
        assert_isinstance(forward, neural.Diagram)
        assert_isinstance(reverse, neural.Diagram)
        if forward.dom != reverse.cod:
            raise ValueError(
                "The forward domain must equal the reverse codomain.")
        candidates = tuple(
            forward.cod[:i] for i in range(len(forward.cod) + 1)
            if reverse.dom
            == forward.cod[i:] @ forward.cod[:i])
        if cod is None:
            if not candidates:
                raise ValueError(
                    "The forward output and reverse input do not determine "
                    "a residual.")
            cod = candidates[-1]
        assert_isinstance(cod, neural.Dim)
        if forward.cod[:len(cod)] != cod:
            raise ValueError("The forward codomain must begin with cod.")
        residual = forward.cod[len(cod):]
        if reverse.dom != residual @ cod:
            raise ValueError(
                "The reverse domain must be residual @ cod.")
        self.forward, self.reverse = forward, reverse
        self.dom, self.cod, self.residual = forward.dom, cod, residual

    def __repr__(self):
        return (
            f"ReverseRule({self.forward!r}, {self.reverse!r}, "
            f"cod={self.cod!r})")

    def __eq__(self, other):
        return isinstance(other, ReverseRule) and (
            self.forward, self.reverse, self.cod) == (
                other.forward, other.reverse, other.cod)

    def bind(self, dom: neural.Dim, cod: neural.Dim) -> ReverseRule:
        """ Validate this rule against a specified arrow type. """
        if self.forward.dom != dom or self.reverse.cod != dom:
            raise ValueError(
                f"Expected a rule with domain {dom}, got "
                f"{self.forward.dom}.")
        return type(self)(self.forward, self.reverse, cod=cod)

    @classmethod
    def id(cls, typ: neural.Dim) -> ReverseRule:
        """ The structural reverse rule for an identity. """
        identity = neural.Diagram.id(typ)
        return cls(identity, identity, cod=typ)

    @classmethod
    def swap(cls, box: neural.Swap) -> ReverseRule:
        """ The structural reverse rule for a swap. """
        assert_isinstance(box, neural.Swap)
        return cls(box, box[::-1], cod=box.cod)

    def then(self, other: ReverseRule) -> ReverseRule:
        """ Compose reverse rules while retaining both residuals. """
        assert_isinstance(other, ReverseRule)
        if self.cod != other.dom:
            raise ValueError(
                f"Cannot compose rules of type {self.dom} -> {self.cod} "
                f"and {other.dom} -> {other.cod}.")
        forward = self.forward >> other.forward @ self.residual\
            >> other.cod @ neural.Diagram.swap(
                other.residual, self.residual)
        reverse = self.residual @ other.reverse >> self.reverse
        return type(self)(forward, reverse, cod=other.cod)

    def tensor(self, other: ReverseRule) -> ReverseRule:
        """ Tensor reverse rules while retaining both residuals. """
        assert_isinstance(other, ReverseRule)
        forward = self.forward @ other.forward\
            >> self.cod @ neural.Diagram.swap(
                self.residual, other.cod) @ other.residual
        reverse = self.residual @ neural.Diagram.swap(
            other.residual, self.cod) @ other.cod\
            >> self.reverse @ other.reverse
        return type(self)(forward, reverse, cod=self.cod @ other.cod)

    __rshift__ = then
    __matmul__ = tensor


def _generator_rule(box, rules) -> ReverseRule:
    """ Look up and type-check the reverse rule for one generator. """
    if isinstance(box, neural.Swap):
        return ReverseRule.swap(box)
    try:
        rule = rules[box]
    except (KeyError, TypeError) as exception:
        raise ValueError(
            f"Missing reverse rule for generator {box!r}.") from exception
    assert_isinstance(rule, ReverseRule)
    return rule.bind(box.dom, box.cod)


def differentiate(graph: neural.Hypergraph, rules) -> ReverseRule:
    """
    Build the reverse rule of a causal monogamous neural hypergraph.

    Parameters:
        graph : The hypergraph to differentiate.
        rules : A mapping or callable from generators to reverse rules.
    """
    assert_isinstance(graph, neural.Hypergraph)
    if not graph.is_monogamous:
        raise ValueError("Reverse differentiation requires monogamy.")
    if not graph.is_causal:
        raise ValueError("Reverse differentiation requires causality.")
    rules = MappingOrCallable(rules)
    result = ReverseRule.id(graph.dom)
    for left, box, right in graph.to_diagram().to_staircases():
        layer = ReverseRule.id(left)\
            @ _generator_rule(box, rules)\
            @ ReverseRule.id(right)
        result >>= layer
    return result


def discard(typ: neural.Dim) -> neural.Network:
    """
    Make an all-port-zero discard network.

    Its module is supplied by the current neural execution backend.
    """
    assert_isinstance(typ, neural.Dim)
    return neural.Network(
        "Discard", typ, neural.Dim(),
        module=neural.get_backend().zeros_module())


def rdiff(graph: neural.Hypergraph, rules, discard_factory=discard):
    """
    Return the reverse derivative ``A @ B -> A`` of ``graph : A -> B``.

    ``discard_factory`` must return a diagram ``B -> Dim()``. Supplying it is
    useful for a backend-specific discard implementation.
    """
    rule = differentiate(graph, rules)
    dropped = discard_factory(graph.cod)
    assert_isinstance(dropped, neural.Diagram)
    if dropped.dom != graph.cod or dropped.cod != neural.Dim():
        raise ValueError(
            "The discard factory must return a diagram B -> Dim().")
    diagram = rule.forward @ graph.cod >> dropped @ rule.reverse
    return diagram.to_hypergraph()
