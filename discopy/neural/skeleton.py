# -*- coding: utf-8 -*-

"""
Abstract wirings: who talks to whom, with no widths, no modules and no
torch.

A :class:`Skeleton` is a closed :class:`~discopy.cmap.CMap` whose boxes
carry no data and whose atomic types name the *role* a port plays rather
than the width it will carry, together with the :class:`Signature` of each
box name.  It is pure syntax: it can be built and checked -- degrees,
involution, loop positions -- on a machine with no torch at all.  What
fills the nodes is decided later by an
:class:`~discopy.neural.functor.Interpretation`.

The source category is a parameter, and its ``require_planar``,
``require_acyclic``, ``require_oriented`` and ``require_connected`` flags
do the guarding: a wiring that is illegal in the source category is
rejected when the map is built, not silently accepted.  Only three
categories admit these shapes -- :mod:`~discopy.symmetric` allows the
crossings but no loops, :mod:`~discopy.compact` and
:mod:`~discopy.frobenius` allow both -- and the target category is the
same compact closed :mod:`~discopy.neural` either way, because swaps,
cups, caps and traces are all wiring there.

Two shapes cover the constraint-satisfaction family, each parameterized by
its combinatorics alone:

* :func:`from_incidence` -- the bipartite incidence graph of a family of
  nodes and the relations they belong to, one hyperedge box per relation.
* :func:`from_relation` -- the graph of a binary relation between nodes,
  one wire per related pair and no hyperedge boxes.

and :func:`from_diagram` reads a skeleton off any diagram, so a source
category's own combinators can draw the wiring instead.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Skeleton
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from discopy import frobenius
from discopy.neural.signature import Orbit, Signature

#: The name the node boxes of a skeleton get by default.
NODE = "cell"

#: The name the hyperedge boxes of a skeleton get by default.
RELATION = "unit"


@dataclass(frozen=True)
class Skeleton:
    """
    A closed abstract map together with the signature of each box name.

    Parameters:
        cmap : The closed map, whose boxes carry roles rather than widths.
        signatures : The signature of each box name.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural.signature import Sym
    >>> message, state = Ty("message"), Ty("state")
    >>> node = Signature((Orbit(message, 1), Orbit(state, traced=True)))
    >>> unit = Signature((Orbit(message, 2, Sym.PERM), ))
    >>> line = from_incidence(((0, ), (0, )), node, unit)
    >>> len(line.cmap.boxes), line.cmap.n_ports
    (3, 8)
    >>> line.signatures[NODE].loops()
    ((1, 2),)
    >>> line.wires()
    (((0, 2), (0, 1)), ((0, 0), (2, 0)), ((1, 2), (1, 1)), ((1, 0), (2, 1)))
    """

    cmap: object
    signatures: Mapping[str, Signature] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.cmap.dom) or len(self.cmap.cod):
            raise ValueError("a skeleton is closed; trace its boundary first")
        for box in self.cmap.boxes:
            signature = self.signatures[box.name]
            if tuple(box.dom) + tuple(box.cod) != signature.roles:
                raise ValueError(
                    f"{box.name} does not have the type of its signature")

    @property
    def boxes(self) -> tuple:
        """ The abstract boxes of the skeleton. """
        return self.cmap.boxes

    @property
    def category(self):
        """ The source category the skeleton is drawn in. """
        return type(self.cmap)

    def signature(self, index: int) -> Signature:
        """
        The signature of the box at an index.

        Parameters:
            index : The index of the box.
        """
        return self.signatures[self.cmap.boxes[index].name]

    def indices(self, name: str) -> tuple[int, ...]:
        """
        The indices of the boxes of a given name, in map order.

        Parameters:
            name : The name of the boxes.
        """
        return tuple(index for index, box in enumerate(self.cmap.boxes)
                     if box.name == name)

    def ports(self, index: int) -> tuple[int, ...]:
        """
        The global port indices of a box in logical order -- domain ports
        then codomain ports -- undoing the clockwise order which stores
        the codomain reversed.  Same convention as
        :meth:`discopy.neural.CMap.box_ports`, for any source category.

        Parameters:
            index : The index of the box.
        """
        box = self.cmap.boxes[index]
        start = len(self.cmap.dom) + sum(
            len(other.dom) + len(other.cod)
            for other in self.cmap.boxes[:index])
        arity = len(box.dom)
        found = tuple(range(start, start + arity + len(box.cod)))
        return found[:arity] + tuple(reversed(found[arity:]))

    def wires(self) -> tuple:
        """
        The wires as pairs of ``(box index, port position)`` pairs, i.e.
        the inverse of :meth:`~discopy.cmap.CMap.from_wiring`.

        Example
        -------
        >>> from discopy.frobenius import Ty
        >>> node = Signature((Orbit(Ty("peer"), 1), ))
        >>> from_relation(((1, ), (0, )), node).wires()
        (((0, 0), (1, 0)),)
        """
        logical = {
            port: (index, position)
            for index in range(len(self.cmap.boxes))
            for position, port in enumerate(self.ports(index))}
        return tuple((logical[i], logical[j])
                     for i, j in enumerate(self.cmap.edges) if i < j)

    def __matmul__(self, other: Skeleton) -> Skeleton:
        """
        The disjoint union of two skeletons, i.e. the monoidal product of
        their maps -- the structure a batch of independent problems has.

        Parameters:
            other : The skeleton to put beside this one.

        Example
        -------
        >>> from discopy.frobenius import Ty
        >>> node = Signature((Orbit(Ty("peer"), 1), ))
        >>> pair = from_relation(((1, ), (0, )), node)
        >>> two = pair @ pair
        >>> len(two.cmap.boxes), two.cmap.n_ports
        (4, 4)
        """
        shared = set(self.signatures) & set(other.signatures)
        for name in shared:
            if self.signatures[name] != other.signatures[name]:
                raise ValueError(f"{name} has two different signatures")
        return Skeleton(self.cmap @ other.cmap,
                        {**self.signatures, **other.signatures})


def _wire_loops(wires: list, index: int, signature: Signature) -> None:
    """
    Close the traced ports of a box onto themselves.

    A loop is a trace, and the trace of a compact map is wiring: the same
    map comes out of tracing the boundary of an open one.

    >>> from discopy.frobenius import Box, CMap, Ty
    >>> x = Ty("x")
    >>> g = Box("g", x, x)
    >>> CMap.from_box(g).trace() == CMap.from_wiring(
    ...     (g, ), [((0, 0), (0, 1))])
    True
    """
    wires += [((index, source), (index, target))
              for source, target in signature.loops()]


def from_incidence(incidence: tuple, node: Signature, relation: Signature,
                   node_name: str = NODE, relation_name: str = RELATION,
                   category=frobenius) -> Skeleton:
    """
    The bipartite incidence graph of a family of nodes and the relations
    they belong to: one node box per node with one incidence port per
    relation it belongs to plus its traced loops, one relation box per
    relation with one port per member, and a wire from each node to each
    of its relations.

    Every node must belong to the same number of relations and every
    relation must have the same number of members, so that one shared
    module fills every node site and one fills every relation site.

    Parameters:
        incidence : Per node, the indices of the relations it belongs to;
                    relations are numbered from ``0``.
        node : The signature of a node box, whose first orbit is the
               incidence orbit.
        relation : The signature of a relation box, whose first orbit is
                   the membership orbit.
        node_name : The name every node box carries.
        relation_name : The name every relation box carries.
        category : The source category the wiring is drawn in.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural.signature import Sym
    >>> message, clue = Ty("message"), Ty("clue")
    >>> node = Signature((Orbit(message, 2, Sym.PERM),
    ...                   Orbit(clue, traced=True)))
    >>> unit = Signature((Orbit(message, 3, Sym.PERM), ))
    >>> square = from_incidence(
    ...     ((0, 1), (0, 1), (0, 1)), node, unit)
    >>> len(square.cmap.boxes), square.cmap.n_ports // 2
    (5, 9)
    """
    n_nodes = len(incidence)
    degree = node.orbits[0].arity
    if any(len(relations) != degree for relations in incidence):
        raise ValueError("nodes belong to different numbers of relations")
    n_relations = 1 + max(max(relations) for relations in incidence)
    size = [0] * n_relations
    for relations in incidence:
        for index in relations:
            size[index] += 1
    if len(set(size)) != 1 or size[0] != relation.orbits[0].arity:
        raise ValueError("relations have different numbers of members")

    free = [0] * n_relations
    wires: list = []
    for index, relations in enumerate(incidence):
        for position, other in enumerate(relations):
            wires.append(
                ((index, position), (n_nodes + other, free[other])))
            free[other] += 1
        _wire_loops(wires, index, node)
    for other in range(n_relations):
        _wire_loops(wires, n_nodes + other, relation)

    boxes = (node.box(node_name, category), ) * n_nodes \
        + (relation.box(relation_name, category), ) * n_relations
    return Skeleton(category.CMap.from_wiring(boxes, wires),
                    {node_name: node, relation_name: relation})


def from_relation(relation: tuple, node: Signature, node_name: str = NODE,
                  category=frobenius) -> Skeleton:
    """
    The graph of a binary relation between nodes: one node box per node
    with one port per related node plus its traced loops, and a wire
    between each related pair.  No hyperedge boxes.

    The relation must be symmetric and every node must be related to the
    same number of others, so that one shared module fills every site.

    Parameters:
        relation : Per node, the indices of the nodes it is related to.
        node : The signature of a node box, whose first orbit is the
               relation orbit.
        node_name : The name every node box carries.
        category : The source category the wiring is drawn in.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural.signature import Sym
    >>> node = Signature((Orbit(Ty("peer"), 2, Sym.PERM),
    ...                   Orbit(Ty("state"), traced=True)))
    >>> triangle = from_relation(((1, 2), (0, 2), (0, 1)), node)
    >>> len(triangle.cmap.boxes), triangle.cmap.n_ports // 2
    (3, 6)
    """
    n_nodes = len(relation)
    arity = node.orbits[0].arity
    if any(len(others) != arity for others in relation):
        raise ValueError("nodes are related to different numbers of nodes")

    wires: list = []
    for index, others in enumerate(relation):
        for other in others:
            if index not in relation[other]:
                raise ValueError("the relation is not symmetric")
            if index < other:
                wires.append(((index, relation[index].index(other)),
                              (other, relation[other].index(index))))
        _wire_loops(wires, index, node)

    boxes = (node.box(node_name, category), ) * n_nodes
    return Skeleton(category.CMap.from_wiring(boxes, wires),
                    {node_name: node})


def from_diagram(diagram, signatures: Mapping[str, Signature] = None,
                 category=None) -> Skeleton:
    """
    The skeleton of a diagram: its combinatorial map, with the signature
    of each box read off its type unless one is declared.

    This is the general entry point.  Structure available at the map's
    categorical level becomes wiring and disappears -- a swap, a cup, a
    cap, a trace -- while a box whose legs carry a symmetry survives as a
    box, which is exactly the box a signature has to speak about.

    Parameters:
        diagram : The closed diagram to read.
        signatures : The signatures to declare, by box name; a box with no
                     declared signature gets the least symmetric one, one
                     orbit per port.
        category : The category whose map factory to use, the diagram's by
                   default.

    Example
    -------
    The symmetry is wiring, so it leaves no box behind -- only a different
    involution:

    >>> from discopy.frobenius import Box, Diagram, Ty
    >>> x = Ty("x")
    >>> f, g = Box("f", Ty(), x @ x), Box("g", x @ x, Ty())
    >>> [box.name for box in from_diagram(f >> g).boxes]
    ['f', 'g']
    >>> list(from_diagram(f >> g).cmap.edges)
    [3, 2, 1, 0]
    >>> list(from_diagram(f >> Diagram.swap(x, x) >> g).cmap.edges)
    [2, 3, 0, 1]

    What survives is a box, and a box is what a signature speaks about:
    declaring its legs one :attr:`Sym.PERM` orbit is what asks its module
    to commute, and what
    :func:`~discopy.neural.signature.check_equivariant` then measures.

    >>> from discopy.neural.signature import Sym
    >>> legs = Signature((Orbit(x, 2, Sym.PERM), ))
    >>> declared = from_diagram(f >> g, {"f": legs, "g": legs})
    >>> declared.signature(0).generators()[0].inside
    [1, 0]
    """
    factory = (category or type(diagram)).map_factory
    cmap = factory.from_diagram(diagram)
    declared = dict(signatures or {})
    for box in cmap.boxes:
        if not hasattr(box, "name"):
            raise ValueError(
                f"{box} is not a box: a skeleton needs one signature per "
                "box name, so the diagram must be built from named boxes")
        declared.setdefault(box.name, Signature(tuple(
            Orbit(role) for role in tuple(box.dom) + tuple(box.cod))))
    return Skeleton(cmap, declared)
