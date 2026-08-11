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

from dataclasses import dataclass, field, replace
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
        for index in range(len(self.cmap.boxes)):
            self.signature(index)

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
        The signature of the box at an index: the one declared for its
        name, with the first orbit resized to the degree the box actually
        has.

        One name, one module, several degrees is how a heterogeneous
        family of sites shares its weights -- the runtime already supports
        it, since :meth:`discopy.neural.CMap._fused_routing` groups boxes
        by module *and* port widths and a :class:`~discopy.neural.cells.
        Cell` reads its arity off the width it is handed.  Only the first
        orbit may vary; a box whose type is neither the declared signature
        nor a resizing of its first orbit is rejected.

        Parameters:
            index : The index of the box.

        Example
        -------
        >>> from discopy.frobenius import Ty
        >>> from discopy.neural.signature import Sym
        >>> node = Signature((Orbit(Ty("peer"), 2, Sym.PERM), ))
        >>> path = from_relation(((1, ), (0, 2), (1, )), node)
        >>> [path.signature(i).orbits[0].arity for i in range(3)]
        [1, 2, 1]
        """
        box = self.cmap.boxes[index]
        declared = self.signatures[box.name]
        roles = tuple(box.dom) + tuple(box.cod)
        if roles == declared.roles:
            return declared
        first, rest = declared.orbits[0], declared.orbits[1:]
        fixed = sum(orbit.n_ports for orbit in rest)
        arity, remainder = divmod(
            len(roles) - fixed, len(first.role) * first.copies)
        if not remainder and arity >= 0:
            try:
                resized = Signature((replace(first, arity=arity), ) + rest)
            except ValueError:
                resized = None
            if resized is not None and resized.roles == roles:
                return resized
        raise ValueError(
            f"{box.name} does not have the type of its signature")

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

        Two declarations of one name are compatible when they agree up to
        the arity of their first orbit -- the orbit :meth:`signature`
        resizes per box anyway -- so a batch can mix instances whose
        canonical degrees differ, e.g. two grid sizes of one task.

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
            one, two = self.signatures[name], other.signatures[name]
            if one != two and not _same_up_to_degree(one, two):
                raise ValueError(f"{name} has two different signatures")
        return Skeleton(self.cmap @ other.cmap,
                        {**self.signatures, **other.signatures})


def _same_up_to_degree(one: Signature, two: Signature) -> bool:
    """ Whether two signatures differ only in their first orbit's arity. """
    if len(one.orbits) != len(two.orbits) \
            or one.orbits[1:] != two.orbits[1:]:
        return False
    try:
        return replace(one.orbits[0], arity=two.orbits[0].arity) \
            == two.orbits[0]
    except ValueError:
        return False


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
                   node_name: str = NODE, relation_name=RELATION,
                   category=frobenius) -> Skeleton:
    """
    The bipartite incidence graph of a family of nodes and the relations
    they belong to: one node box per node with one incidence port per
    relation it belongs to plus its traced loops, one relation box per
    relation with one port per member, and a wire from each node to each
    of its relations.

    Neither the degrees nor the sizes need to be uniform: a node of degree
    ``d`` gets the node signature with its first orbit resized to ``d``,
    and likewise a relation of ``m`` members -- one shared module still
    fills every site of a name, reading its arity off the width it is
    handed, at the cost of one batched call per distinct degree.  A
    module pooled with ``"mean"`` keeps its input scale independent of the
    degree; declare that on any generator whose degree varies.

    Parameters:
        incidence : Per node, the indices of the relations it belongs to;
                    relations are numbered from ``0``.
        node : The signature of a node box, whose first orbit is the
               incidence orbit; its declared arity is a default, resized
               per node.
        relation : The signature of a relation box, whose first orbit is
                   the membership orbit, resized per relation; or a
                   mapping from relation name to signature when the
                   relations do not all share one.
        node_name : The name every node box carries.
        relation_name : The name every relation box carries, or one name
                        per relation -- boxes of one name share one
                        module, so a relation playing a different part,
                        e.g. a graph-level readout wired to every node,
                        is a relation with a name of its own.
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

    A graph-level readout is one more relation every node belongs to,
    under its own name -- a generator, not an engine feature:

    >>> shape = from_incidence(
    ...     ((0, 1), (0, 1), (1, )), node, unit,
    ...     relation_name=("unit", "readout"))
    >>> [box.name for box in shape.boxes]
    ['cell', 'cell', 'cell', 'unit', 'readout']
    >>> [len(shape.signature(i).cod) for i in range(5)]
    [4, 4, 3, 2, 3]
    """
    n_nodes = len(incidence)
    n_relations = 1 + max(max(relations) for relations in incidence)
    size = [0] * n_relations
    for relations in incidence:
        for index in relations:
            size[index] += 1
    names = (relation_name, ) * n_relations \
        if isinstance(relation_name, str) else tuple(relation_name)
    if len(names) != n_relations:
        raise ValueError(f"{len(names)} names for {n_relations} relations")
    relations_of = dict.fromkeys(names, relation) \
        if isinstance(relation, Signature) else dict(relation)

    role = node.orbits[0].role
    nodes = [node.resize(role, len(relations)) for relations in incidence]
    units = [relations_of[names[index]].resize(
        relations_of[names[index]].orbits[0].role, size[index])
        for index in range(n_relations)]

    free = [0] * n_relations
    wires: list = []
    for index, relations in enumerate(incidence):
        for position, other in enumerate(relations):
            wires.append(
                ((index, position), (n_nodes + other, free[other])))
            free[other] += 1
        _wire_loops(wires, index, nodes[index])
    for other in range(n_relations):
        _wire_loops(wires, n_nodes + other, units[other])

    boxes = tuple(sig.box(node_name, category) for sig in nodes) + tuple(
        sig.box(names[index], category)
        for index, sig in enumerate(units))
    return Skeleton(category.CMap.from_wiring(boxes, wires),
                    {node_name: node, **relations_of})


def from_relation(relation: tuple, node: Signature, node_name: str = NODE,
                  category=frobenius) -> Skeleton:
    """
    The graph of a binary relation between nodes: one node box per node
    with one port per related node plus its traced loops, and a wire
    between each related pair.  No hyperedge boxes.

    The relation must be symmetric; the degrees need not be uniform, a
    node related to ``d`` others gets the node signature with its first
    orbit resized to ``d``, and the one shared module still fills every
    site (see :func:`from_incidence`).

    Parameters:
        relation : Per node, the indices of the nodes it is related to.
        node : The signature of a node box, whose first orbit is the
               relation orbit; its declared arity is a default, resized
               per node.
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
    >>> path = from_relation(((1, ), (0, 2), (1, )), node)
    >>> [path.signature(i).orbits[0].arity for i in range(3)]
    [1, 2, 1]
    """
    role = node.orbits[0].role
    nodes = [node.resize(role, len(others)) for others in relation]

    wires: list = []
    for index, others in enumerate(relation):
        for other in others:
            if index not in relation[other]:
                raise ValueError("the relation is not symmetric")
            if index < other:
                wires.append(((index, relation[index].index(other)),
                              (other, relation[other].index(index))))
        _wire_loops(wires, index, nodes[index])

    boxes = tuple(sig.box(node_name, category) for sig in nodes)
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
