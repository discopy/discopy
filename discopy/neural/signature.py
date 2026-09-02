# -*- coding: utf-8 -*-

"""
What a generator promises: its ports, grouped into orbits, and the symmetry
each orbit carries.

A functor into :mod:`discopy.neural` preserves swaps, cups, caps and traces
strictly and for free, because they are wiring: a permutation of a flat
tensor.  What it cannot preserve for free is a box whose legs carry a
symmetry -- a spider, a braid, a constraint unit over nine members.  Those
stay boxes, and their equations hold **iff the torch module satisfies
them**.  A :class:`Signature` is where that promise is written down: it says
how many ports a box has, which of them are one orbit under a group, and
which are traced; :func:`~discopy.neural.laws.check_equivariant` then
measures whether the module keeps it.

A signature is not part of the user-facing workflow -- a
:class:`~discopy.neural.MapNN` reads a diagram, not a signature.  It is the
single source of truth for the *port layout of one generator*, and it is
used in three places that would otherwise have to agree by hand:

* :meth:`Signature.cod` builds the type of an abstract box, so that
  :func:`from_incidence` and :func:`from_relation` can draw a wiring out of
  a family's combinatorics alone;
* :meth:`Signature.slices` gives the flat offsets a
  :mod:`~discopy.neural.cells` module reads and writes;
* :meth:`Signature.generators` gives the group
  :func:`~discopy.neural.laws.check_equivariant` runs a module against.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Sym
    Orbit
    Signature

Example
-------

>>> from discopy.frobenius import Ty
>>> message, state, given = Ty("message"), Ty("state"), Ty("given")
>>> cell = Signature((
...     Orbit(message, 3, Sym.PERM), Orbit(state, traced=True),
...     Orbit(given, traced=True)))
>>> print(cell.cod)
message @ message @ message @ state @ state @ given @ given
>>> cell.positions(state)
(3, 4)
>>> cell.loops()
((3, 4), (5, 6))
>>> places = cell.slices({message: 24, state: 96, given: 24})
>>> {str(role): (block.start, block.stop) for role, block in places.items()}
{'message': (0, 72), 'state': (72, 168), 'given': (264, 288)}
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Mapping

from discopy import frobenius
from discopy.neural.core import from_wiring
from discopy.python.finset import Permutation

#: The name the node boxes of a generated wiring get by default.
NODE = "cell"

#: The name the hyperedge boxes of a generated wiring get by default.
RELATION = "unit"


class Sym(StrEnum):
    """
    The symmetry an orbit of ports carries, i.e. the group the module at
    that site must be equivariant under.

    * :attr:`NONE` : the ports are distinguishable, no equation.
    * :attr:`PERM` : the symmetric group, e.g. the members of a constraint
      unit or the legs of a spider.
    * :attr:`CYCLIC` : the cyclic group, e.g. the legs of a planar node.
    """

    NONE = "none"
    PERM = "perm"
    CYCLIC = "cyclic"


@dataclass(frozen=True)
class Orbit:
    """
    A family of ports playing the same role.

    An orbit has ``arity`` legs, each carrying the (possibly composite)
    type ``role``; a traced orbit has each leg twice, the outgoing copy
    followed by the incoming one, which :meth:`Signature.loops` wires
    together.

    A composite ``role`` is one leg carrying several roles at once, which
    is how a recurrent cell with two states -- the ``h`` and ``c`` of an
    ``LSTMCell`` -- keeps them as two named roles on one loop rather than
    as two halves of one wide port.

    Parameters:
        role : The type of one leg, a product of atomic roles.
        arity : The number of legs.
        sym : The group the legs are an orbit under.
        traced : Whether each leg is a self-wired pair of ports.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> hidden, memory = Ty("hidden"), Ty("memory")
    >>> print(Orbit(hidden @ memory, traced=True).cod)
    hidden @ memory @ hidden @ memory
    """

    role: frobenius.Ty
    arity: int = 1
    sym: Sym = Sym.NONE
    traced: bool = False

    def __post_init__(self):
        if self.arity < 0:
            raise ValueError(self.arity)
        if len(self.role) > 1 and self.arity != 1:
            raise ValueError(
                "a leg carrying several roles cannot also be repeated")

    @property
    def copies(self) -> int:
        """ How many times each leg appears: two when traced, one else. """
        return 2 if self.traced else 1

    @property
    def cod(self) -> frobenius.Ty:
        """ The ports of the orbit, as a type. """
        return self.role ** (self.arity * self.copies)

    @property
    def n_ports(self) -> int:
        """ The number of ports of the orbit. """
        return len(self.role) * self.arity * self.copies


@dataclass(frozen=True)
class Signature:
    """
    The ports of one generator, as a tuple of orbits, in logical port order.

    Parameters:
        orbits : The orbits, in the order their ports appear.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> peer, hidden, memory = Ty("peer"), Ty("hidden"), Ty("memory")
    >>> clique = Signature((
    ...     Orbit(peer, 4, Sym.PERM), Orbit(hidden @ memory, traced=True),
    ...     Orbit(Ty("given"), traced=True)))
    >>> clique.positions(hidden), clique.positions(memory)
    ((4, 6), (5, 7))
    >>> clique.loops()
    ((4, 6), (5, 7), (8, 9))
    """

    orbits: tuple[Orbit, ...]

    @property
    def cod(self) -> frobenius.Ty:
        """ The codomain of the abstract box: every port, in order. """
        result = frobenius.Ty()
        for orbit in self.orbits:
            result = result @ orbit.cod
        return result

    @property
    def roles(self) -> tuple[frobenius.Ty, ...]:
        """ The atomic role of each port, in logical port order. """
        return tuple(self.cod)

    def box(self, name: str, category=frobenius):
        """
        The abstract box of this signature: no domain, one port per role.

        The source category is a parameter, so the same signature builds a
        symmetric, compact or frobenius box; its ``require_planar``,
        ``require_acyclic``, ``require_oriented`` and ``require_connected``
        flags then do the guarding when the box is wired into a map.

        Parameters:
            name : The name of the box, which is also the key its module is
                   looked up under.
            category : The module the box and its types come from.

        Example
        -------
        >>> from discopy import compact, frobenius, symmetric
        >>> unit = Signature((Orbit(frobenius.Ty("message"), 3, Sym.PERM), ))
        >>> print(unit.box("unit").cod)
        message @ message @ message
        >>> isinstance(unit.box("unit", category=symmetric), symmetric.Box)
        True
        """
        typ = category.Ty(*[atom.inside[0].name for atom in self.cod])
        return category.Box(name, category.Ty(), typ)

    def positions(self, role: frobenius.Ty) -> tuple[int, ...]:
        """
        Where an atomic role sits in the logical port order.

        For a traced orbit the outgoing copies come first, so the ``i``-th
        and the ``arity + i``-th entries are the two ends of one loop.

        Parameters:
            role : The atomic role to locate.
        """
        return tuple(i for i, other in enumerate(self.roles)
                     if other == role)

    def loops(self) -> tuple[tuple[int, int], ...]:
        """
        The traced pairs of ports, as positions in the logical port order.

        Two readings of one pair, worth keeping apart.  *Structurally* a
        self-wired pair is the categorical trace of the compact target --
        wiring, which a functor preserves strictly.  *Dynamically* it is a
        persistent state channel: what a box writes on one end it reads
        back on the other one round later.  That is delayed feedback under
        finite iteration, not a fixed point; see
        :class:`~discopy.neural.map.Interaction`.
        """
        result, cursor = [], 0
        for orbit in self.orbits:
            span = len(orbit.role) * orbit.arity
            if orbit.traced:
                result += [(cursor + i, cursor + span + i)
                           for i in range(span)]
            cursor += span * orbit.copies
        return tuple(result)

    def slices(self, widths: Mapping) -> dict:
        """
        Where each atomic role sits in the flat message vector of a box,
        given the width of every role.

        This is the one place a port offset is computed.  A module reads and
        writes through these slices, so its cursor arithmetic and the type
        of its abstract box can no longer disagree.

        Parameters:
            widths : The width carried by each atomic role; roles of width
                     zero are erased, exactly as ``Dim(0)`` erases a port.
        """
        result, cursor = {}, 0
        for orbit in self.orbits:
            leg = sum(widths[atom] for atom in orbit.role)
            block = orbit.arity * leg
            inner = cursor
            for atom in orbit.role:
                width = widths[atom]
                if width:
                    result[atom] = slice(
                        inner, inner + (block if len(orbit.role) == 1
                                        else width))
                inner += width
            cursor += orbit.copies * block
        return result

    def width(self, widths: Mapping) -> int:
        """ The total flat width of a box under the given role widths. """
        return sum(
            orbit.copies * orbit.arity
            * sum(widths[atom] for atom in orbit.role)
            for orbit in self.orbits)

    def resize(self, role: frobenius.Ty, arity: int) -> Signature:
        """
        The same signature with the arity of one orbit changed, which is
        how one shared module serves sites of different degree.

        Parameters:
            role : The role of the orbit to resize.
            arity : Its new arity.
        """
        return Signature(tuple(
            replace(orbit, arity=arity) if role in tuple(orbit.role)
            else orbit for orbit in self.orbits))

    def generators(self) -> list[Permutation]:
        """
        The generators of the symmetry group of the signature, as
        permutations of its ports.

        A permutation acts on the *legs* of one orbit and on every copy of
        each leg alike, so a traced orbit stays traced.  The identity of
        the group generated is the equation the module at this site must
        satisfy; :func:`~discopy.neural.laws.check_equivariant` measures
        how far it is from it.

        Example
        -------
        >>> from discopy.frobenius import Ty
        >>> unit = Signature((Orbit(Ty("message"), 3, Sym.PERM), ))
        >>> [tuple(permutation.inside) for permutation in unit.generators()]
        [(1, 0, 2), (1, 2, 0)]
        """
        result, cursor = [], 0
        for orbit in self.orbits:
            span, legs = len(orbit.role), orbit.arity
            block = span * legs
            for cycle in leg_generators(orbit.sym, legs):
                mapping = list(range(len(self.roles)))
                for copy in range(orbit.copies):
                    start = cursor + copy * block
                    for leg in range(legs):
                        for atom in range(span):
                            mapping[start + leg * span + atom] = \
                                start + cycle[leg] * span + atom
                result.append(Permutation(mapping))
            cursor += block * orbit.copies
        return result


def leg_generators(sym: Sym, arity: int) -> list[tuple[int, ...]]:
    """
    The generators of a symmetry group, as permutations of legs.

    This is the group itself, before it acts on anything:
    :meth:`Signature.generators` is the same group acting on ports and
    :mod:`discopy.neural.laws` is where it is read as an action
    :math:`\\rho : G \\to \\mathrm{Aut}(X)`.

    Parameters:
        sym : The symmetry the legs carry.
        arity : The number of legs.

    Example
    -------
    >>> leg_generators(Sym.PERM, 3)
    [(1, 0, 2), (1, 2, 0)]
    >>> leg_generators(Sym.CYCLIC, 3)
    [(1, 2, 0)]
    >>> leg_generators(Sym.NONE, 3), leg_generators(Sym.PERM, 1)
    ([], [])
    """
    if arity < 2 or sym == Sym.NONE:
        return []
    rotation = tuple(range(1, arity)) + (0, )
    if sym == Sym.CYCLIC:
        return [rotation]
    swap = (1, 0) + tuple(range(2, arity))
    return [swap, rotation]


# --- drawing a wiring out of a family's combinatorics ----------------------

def _wire_loops(wires: list, index: int, signature: Signature) -> None:
    """
    Close the traced ports of a box onto themselves.

    A loop is a trace, and the trace of a compact map is wiring: the same
    map comes out of tracing the boundary of an open one.

    >>> from discopy.frobenius import Box, CMap, Ty
    >>> g = Box("g", Ty("x"), Ty("x"))
    >>> CMap.from_box(g).trace() == from_wiring(
    ...     CMap, (g, ), [((0, 0), (0, 1))])
    True
    """
    wires += [((index, source), (index, target))
              for source, target in signature.loops()]


def from_incidence(incidence: tuple, node: Signature, relation: Signature,
                   node_name: str = NODE, relation_name=RELATION,
                   category=frobenius):
    """
    The bipartite incidence graph of a family of nodes and the relations
    they belong to, as a closed map in the source category: one node box
    per node with one incidence port per relation it belongs to plus its
    traced loops, one relation box per relation with one port per member,
    and a wire from each node to each of its relations.

    Neither the degrees nor the sizes need to be uniform: a node of degree
    ``d`` gets the node signature with its first orbit resized to ``d``,
    and likewise a relation of ``m`` members -- one shared module still
    fills every site of a name, reading its arity off the width it is
    handed, at the cost of one batched call per distinct degree.  A module
    pooled with ``"mean"`` keeps its input scale independent of the
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
    >>> message, given = Ty("message"), Ty("given")
    >>> node = Signature((Orbit(message, 2, Sym.PERM),
    ...                   Orbit(given, traced=True)))
    >>> unit = Signature((Orbit(message, 3, Sym.PERM), ))
    >>> square = from_incidence(((0, 1), (0, 1), (0, 1)), node, unit)
    >>> len(square.boxes), square.n_ports // 2
    (5, 9)

    A graph-level readout is one more relation every node belongs to,
    under its own name -- a generator, not a solver feature:

    >>> shape = from_incidence(
    ...     ((0, 1), (0, 1), (1, )), node, unit,
    ...     relation_name=("unit", "readout"))
    >>> [box.name for box in shape.boxes]
    ['cell', 'cell', 'cell', 'unit', 'readout']
    """
    n_nodes = len(incidence)
    n_relations = 1 + max(
        (index for relations in incidence for index in relations),
        default=-1)
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
    return from_wiring(category.CMap, boxes, wires)


def from_relation(relation: tuple, node: Signature, node_name: str = NODE,
                  category=frobenius):
    """
    The graph of a binary relation between nodes, as a closed map in the
    source category: one node box per node with one port per related node
    plus its traced loops, and a wire between each related pair.  No
    hyperedge boxes.

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
    >>> node = Signature((Orbit(Ty("peer"), 2, Sym.PERM),
    ...                   Orbit(Ty("state"), traced=True)))
    >>> triangle = from_relation(((1, 2), (0, 2), (0, 1)), node)
    >>> len(triangle.boxes), triangle.n_ports // 2
    (3, 6)
    >>> [len(box.cod) for box in from_relation(
    ...     ((1, ), (0, 2), (1, )), node).boxes]
    [3, 4, 3]
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
    return from_wiring(category.CMap, boxes, wires)
