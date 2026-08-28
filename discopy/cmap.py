
"""
An implementation of open `combinatorial maps
<https://en.wikipedia.org/wiki/Combinatorial_map>`_.
See :cite:`DelpeuchVicary22` for a comprehensive overview of combinatorial
maps in relation to string diagrams.

A combinatorial map is fully described by a pair of permutations :math:`v` and
:math:`e` acting on a set of ports :math:`P` (also called darts or
half-edges) where:

* :math:`v` is an arbitrary permutation whose decomposition induces a node for
  each cycle, giving an orientation on ports;
* :math:`e` is a fixpoint-free involution, hence its cycle decomposition
  only contains transpositions which are to be understood as wires of the map.

A map morphism from :math:`(P, v, e)` to :math:`(P', v', e')` is then defined
as a function :math:`f : P \\rightarrow P'` such that:

* :math:`f` defines a homomorphism of the underlying graph:
  :math:`e; f = f; e'`;
* :math:`f` respects orientation: :math:`v; f = f; v'`.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    PortKind
    Port
    CMap
"""

from __future__ import annotations

import operator
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property, reduce
from inspect import isclass
from io import BytesIO
from itertools import count
from math import inf, lcm
from string import ascii_lowercase
from typing import TYPE_CHECKING, ClassVar, Literal

from discopy import hypergraph, messages
from discopy.abc import (
    CompactCategory,
    NamedGeneric,
    Pregroup,
    RigidCategory,
    SymmetricCategory,
    TracedCategory,
)
from discopy.cat import Ob
from discopy.python.finset import Permutation
from discopy.utils import (
    AxiomError,
    assert_isatomic,
    assert_isinstance,
    classproperty,
    factory_name,
    unbiased,
)

if TYPE_CHECKING:
    from discopy.monoidal import Box, Diagram, Ty


class PortKind(StrEnum):
    """ The four kinds of ports in a :class:`CMap`. """

    INPUT = "input"
    OUTPUT = "output"
    DOM = "dom"
    COD = "cod"

    @property
    def is_negative(self) -> bool:
        """ Whether the port is a box input or map output. """
        return self == "dom" or self == "output"

    @property
    def is_positive(self) -> bool:
        """ Whether the port is a map input or box output. """
        return self == "input" or self == "cod"

    @property
    def is_boundary(self) -> bool:
        """ Whether the port belongs to the map boundary. """
        return self == "input" or self == "output"

    @property
    def is_input(self) -> bool:
        """ Whether the port is drawn on the input side. """
        return self == "input" or self == "dom"

    @property
    def is_output(self) -> bool:
        """ Whether the port is drawn on the output side. """
        return self == "cod" or self == "output"


@dataclass(frozen=True)
class Port:
    """
    A port in a combinatorial map.

    Parameters:
        kind : The kind of boundary or box port.
        i : The position within its boundary or box side.
        obj : The type carried by the port.
        depth : The box index, with inputs at ``-inf`` and outputs at ``+inf``.
        side : The vertical side on which the port is drawn.
    """
    kind: PortKind
    i: int
    obj: Ob
    depth: float
    side: Literal["up", "down"]


class CMap[C0: Pregroup, C1: CMap](
    CompactCategory[C0, C1], NamedGeneric['category']
):
    r"""
    An open combinatorial map, i.e. a diagram represented as a bijection
    between its ports.

    Contrary to the abstract definition, which has unstructured nodes arising
    from the given orientation permutation, we take DisCoPy boxes as nodes and
    derive a canonical clockwise port orientation on boxes: every box of arity
    :math:`m` and coarity :math:`n` maps to a :math:`(m+n)`-cycle in the
    generated permutation, consisting of contiguous port indices.
    Additionally, we allow two kinds of scalars:

    * `scalar loops` arising from composing cups and caps, parametrized by an
      atomic type;
    * `scalar boxes`, i.e. boxes with empty domain and codomain

    As for the open structure, we represent the map boundary by a virtual apex
    node, whose signature is the dagger of the that of the overall map.

    By default, `CMap` defines the free compact category over a set of boxes,
    but we also want to be able to encode weaker structure, disallowing cups
    and caps or even traced structure altogether.
    We therefore further distinguish port sides by assigning a negative
    polarity on domain ports and a positive polarity on codomain ports
    by equipping the map with a polarity assignment
    :math:`m : P \rightarrow \{-1, +1\}`.

    Following :class:`Hypergraph`, the map is parametrised by a category.
    The functor used by :meth:`from_diagram` is read from
    ``category.functor_factory``; :meth:`Diagram.to_map` parameterises
    ``CMap`` with the concrete diagram category automatically.
    A map is always compact, whatever the category that hosts it, so that
    every compact operation is available when manipulating maps. It is
    :meth:`to_diagram` that needs the structure of ``category``:

    * cups and caps, i.e. same-polarity pairings :math:`e; m = m` (see
      :attr:`is_monogamous`), are made explicit by :meth:`make_monogamous`,
      which needs a category with cups and caps;
    * traces, i.e. cycles and loops (see :attr:`is_acyclic`), are made
      explicit by :meth:`make_causal`, which needs a traced category.

    Parameters:
        dom : The domain of the map.
        cod : The codomain of the map.
        boxes : The boxes inside the map.
        edges : A fixpoint-free involution on ports.
        loops : The types of closed wire components with no ports.
        check : Whether to :meth:`validate` the involution and the wire types.

    Example
    -------
    >>> from discopy.compact import Ty, Box, CMap
    >>> from discopy.python.finset import Permutation
    >>> x, y, z = map(Ty, "xyz")
    >>> f, g = map(CMap.from_box, [
    ...     Box("f", x @ y, x @ z),
    ...     Box("g", z @ z, z),
    ... ])
    >>> cm = f @ z >> x @ g
    >>> # apex: 10 : x, 11 : z ⊢ 2 : x, 1 : y, 0 : z
    >>> # f:    3 : x, 4 : y ⊢ 6 : x, 5 : z
    >>> # g:    7 : z, 8 : z ⊢ 9 : z
    >>> cm.edges == Permutation.from_cycles([
    ...     (0, 3), (1, 4), (2, 8), (5, 7), (6, 10), (9, 11)], 12)
    True
    >>> cm.orientation == Permutation.from_cycles([
    ...     (2, 1, 0, 10, 11), (3, 4, 5, 6), (7, 8, 9)], 12)
    True
    >>> cm.draw(
    ...     doctest="docs/_static/cmap/simple-cmap.dot",
    ...     port_indices=True,
    ...     show=False,
    ... )

    .. graphviz:: /_static/cmap/simple-cmap.dot
        :align: center

    Swaps affect the edge permutation but leave the vertex permutation
    fixed:

    >>> f, g = map(CMap.from_box, [
    ...     Box("f", x @ y, z @ x),
    ...     Box("g", z @ z, z),
    ... ])
    >>> cm = (f >> CMap.swap(z, x)) @ z >> x @ g
    >>> cm.draw(
    ...     doctest="docs/_static/cmap/swapped-cmap.dot",
    ...     port_indices=True,
    ...     show=False,
    ... )

    .. graphviz:: /_static/cmap/swapped-cmap.dot
        :align: center
    """

    category: ClassVar[Diagram] = None
    functor = classproperty(lambda cls: cls.category.functor_factory)
    ob = classproperty(lambda cls: cls.category.ob)

    dom: C0
    cod: C0
    loops: tuple[C0, ...]
    edges: Permutation

    def __init__(
            self, dom: C0, cod: C0, boxes: tuple[Box, ...],
            edges: Iterable[int],
            loops: tuple[C0, ...] = (), *, check: bool = True):
        assert_isinstance(dom, self.category.ob)
        assert_isinstance(cod, self.category.ob)
        for box in boxes:
            assert_isinstance(box, self.category)
        for loop in loops:
            assert_isatomic(loop, self.category.ob)
        self.dom, self.cod, self.boxes = dom, cod, tuple(boxes)
        self.loops = tuple(loops)

        self.edges = Permutation(edges, self.n_ports)
        if check:
            self.validate()

    @cached_property
    def ports(self) -> list[Port]:
        """ The ports in canonical orientation order. """
        def port(kind, i, obj, depth):
            if not kind.is_boundary:
                depth += 0.5 if kind.is_input else -0.5
            return Port(
                kind, i=i, obj=obj, depth=depth,
                side="up" if kind.is_input else "down")

        inputs = [port(PortKind.INPUT, i=i, obj=obj, depth=-inf)
                  for i, obj in enumerate(self.dom)]
        box_ports = reduce(operator.iadd, [
            [
              port(kind, i=i, obj=obj, depth=depth)
              for i, obj in indexed_typ
            ]
            for depth, box in enumerate(self.boxes)
            for kind, indexed_typ in [
                (PortKind.DOM, tuple(enumerate(box.dom))),
                (PortKind.COD, tuple(reversed(tuple(enumerate(box.cod)))))
            ]
        ], [])
        outputs = [port(PortKind.OUTPUT, i=i, obj=obj, depth=inf)
                   for i, obj in enumerate(self.cod)]
        return inputs + box_ports + outputs

    @property
    def n_ports(self) -> int:
        """ The number of ports. """
        return len(self.dom) + sum(
            len(box.dom) + len(box.cod) for box in self.boxes) + len(self.cod)

    @cached_property
    def _box_port_indices(self) -> tuple[tuple[int, ...], ...]:
        """ The consecutive port indices belonging to each box. """
        result, start = [], len(self.dom)
        for box in self.boxes:
            stop = start + len(box.dom) + len(box.cod)
            result.append(tuple(range(start, stop)))
            start = stop
        return tuple(result)

    @property
    def faces(self) -> Permutation:
        """ The face permutation, computed as ``edges ; orientation``. """
        return self.edges.then(self.orientation)

    @property
    def n_vertices(self) -> int:
        """ The number of vertices, including the boundary apex if present. """
        return len(self.boxes) + bool(len(self.dom) or len(self.cod))

    @property
    def n_edges(self) -> int:
        """ The number of edges. """
        return self.n_ports // 2 + len(self.loops)

    @property
    def n_faces(self) -> int:
        """ The number of faces, including closed scalar components. """
        portless_boxes = sum(
            not len(box.dom) and not len(box.cod) for box in self.boxes)
        return len(self.faces.cycles()) + portless_boxes\
            + len(self.loops)

    @property
    def euler_characteristic(self) -> int:
        """
        Euler characteristic ``V - E + F`` with boundary at infinity.

        For maps with non-empty domain or codomain, the input and output ports
        are treated as one virtual boundary/apex, ordered clockwise as inputs
        left-to-right followed by outputs right-to-left. Fully closed maps have
        no boundary apex.

        >>> from discopy.symmetric import Ty, Box, Swap
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> f.to_map().euler_characteristic
        2
        >>> (Swap(y, x) >> f).to_map().euler_characteristic
        0
        """
        if len(self.connected_components) != 1:
            raise ValueError(messages.NOT_CONNECTED.format(self))
        if not self.n_ports and not self.boxes and not self.loops:
            return 2
        return self.n_vertices - self.n_edges + self.n_faces

    @property
    def is_scalar(self) -> bool:
        """
        Whether the map is scalar, i.e. a single box with no ports, or a
        single scalar loop.
        """
        if self.n_ports > 0:
            return False
        if not self.boxes and len(self.loops) == 1:
            return True
        return len(self.boxes) == 1 and not self.loops

    @property
    def is_planar(self) -> bool:
        """
        Whether the combinatorial map is planar, i.e. all of its non-scalar
        components have an Euler characteristic of 2.
        """

        components = [
            component for component in self.connected_components
            if not component.is_scalar]
        if not components:
            return True
        return all(
            component.euler_characteristic == 2 for component in components)

    @property
    def orientation(self) -> Permutation:
        """
        The closed orientation permutation.

        The first cycle is the boundary apex, when the boundary is non-empty.
        Each following non-empty cycle is the contiguous port interval of a
        box in canonical order: domain ports, then codomain ports.

        >>> from discopy.compact import Ty, Box, CMap
        >>> from discopy.python.finset import Permutation
        >>> x, y, z = map(Ty, "xyz")
        >>> f, g = Box('f', x @ y, x @ z), Box('g', z @ z, z)
        >>> cm = (f @ z >> x @ g).to_map()
        >>> assert cm.orientation == Permutation.from_cycles([
        ...     (2, 1, 0, 10, 11), # boundary
        ...     (3, 4, 5, 6),      # f
        ...     (7, 8, 9),         # g
        ... ], 12), f"got {cm.orientation.cycles()!r}"
        """
        boundary = (self.boundary_cycle, ) if self.boundary_cycle else ()
        return Permutation.from_cycles(
            boundary + self._box_port_indices, len(self.ports))

    @property
    def boundary_cycle(self) -> tuple[int, ...]:
        """ The clockwise cycle of the virtual boundary apex. """
        inputs = tuple(range(len(self.dom)))
        outputs = tuple(range(self.n_ports - len(self.cod), self.n_ports))
        return tuple(reversed(inputs)) + outputs

    def validate(self):
        """ Validate the edges involution and the types of each wire. """
        ports = self.ports
        if not self.edges.is_fixpoint_free_involution():
            raise ValueError

        for i, j in enumerate(self.edges):
            if i > j:
                continue
            type(self).validate_wire(ports[i], ports[j])

    @classmethod
    def assert_isrigid(cls):
        """ Assert that :attr:`category` has cups and caps. """
        if not issubclass(cls.category, RigidCategory):
            raise AxiomError(messages.NOT_RIGID.format(
                factory_name(cls.category)))

    @classmethod
    def assert_istraced(cls):
        """ Assert that :attr:`category` has traces. """
        if not issubclass(cls.category, TracedCategory):
            raise AxiomError(messages.NOT_TRACED.format(
                factory_name(cls.category)))

    @property
    def connected_components(self) -> list[CMap]:
        """ The connected components, with the boundary component first. """
        if not self.n_ports:
            # Avoid recursively rebuilding the same portless component.
            if len(self.boxes) + len(self.loops) <= 1:
                return [self]
            components = [
                type(self)(self.ob(), self.ob(), (box, ), (), check=False)
                for box in self.boxes]
            components += [
                type(self)(
                    self.ob(), self.ob(), (), (), loops=(loop, ), check=False)
                for loop in self.loops]
            return components

        component_of = self.edges.coequalizer(self.orientation)
        boundary = set(range(len(self.dom))) | set(range(
            self.n_ports - len(self.cod), self.n_ports))
        boundary_component = component_of[next(iter(boundary))]\
            if boundary else None

        ports_by_component: dict[int, list[int]] = {}
        for port, component in component_of.items():
            ports_by_component.setdefault(component, []).append(port)

        boxes_by_component: dict[int, list[tuple[int, Box]]] = {}
        portless_boxes: list[Box] = []
        for box_index, box in enumerate(self.boxes):
            box_ports = self._box_port_indices[box_index]
            if not box_ports:
                portless_boxes.append(box)
                continue
            component = component_of[box_ports[0]]
            boxes_by_component.setdefault(component, []).append((
                box_index, box))

        if len(ports_by_component) == 1 and not portless_boxes\
                and not self.loops:
            return [self]

        def make_component(component: int) -> CMap:
            dom = self.dom if component == boundary_component else self.ob()
            cod = self.cod if component == boundary_component else self.ob()
            boxes = tuple(box for _, box in boxes_by_component.get(
                component, ()))

            kept_ports = []
            if component == boundary_component:
                kept_ports += list(range(len(self.dom)))
            for box_index, _ in boxes_by_component.get(component, ()):
                kept_ports += list(self._box_port_indices[box_index])
            if component == boundary_component:
                kept_ports += list(range(
                    self.n_ports - len(self.cod), self.n_ports))
            mapping = {old: new for new, old in enumerate(kept_ports)}
            edges = Permutation.from_transpositions(
                ((mapping[i], mapping[j])
                 for i, j in enumerate(self.edges)
                 if i < j and i in mapping and j in mapping),
                len(kept_ports))
            return type(self)(dom, cod, boxes, edges, check=False)

        ordered_components = sorted(
            ports_by_component,
            key=lambda component: (
                component != boundary_component,
                min(ports_by_component[component])))
        components = [make_component(component)
                      for component in ordered_components]
        components += [
            type(self)(self.ob(), self.ob(), (box, ), (), check=False)
            for box in portless_boxes]
        components += [
            type(self)(
                self.ob(), self.ob(), (), (), loops=(loop, ), check=False)
            for loop in self.loops]
        return components

    def splice(
            self, edges: Permutation,
            glue: Permutation,
            ports: list[Port]) -> tuple[Permutation, tuple]:
        """
        Compute the edges and scalars created by a gluing operation.
        """
        components = edges.coequalizer(glue)
        removed = {port for port in range(len(glue)) if glue[port] != port}
        removed_by_component: dict[int, list[int]] = {}
        for port in removed:
            removed_by_component.setdefault(components[port], []).append(port)
        kept = [i for i in range(len(edges)) if i not in removed]
        mapping = {old: new for new, old in enumerate(kept)}
        surviving: dict[int, list[int]] = {}
        for port, component in components.items():
            if port not in removed:
                surviving.setdefault(component, []).append(port)

        edge_pairs = [
            tuple(sorted(mapping[port] for port in ports))
            for ports in surviving.values() if len(ports) == 2]
        scalars, scalar_components = [], set()
        for component, removed_ports in removed_by_component.items():
            if component in surviving or component in scalar_components:
                continue
            scalar = ports[removed_ports[0]].obj
            scalar = scalar if isinstance(scalar, self.category.ob)\
                else self.ob(scalar)
            scalars.append(
                scalar.r if getattr(scalar, "z", 0) % 2 else scalar)
            scalar_components.add(component)
        return (
            Permutation.from_transpositions(edge_pairs, len(kept)),
            tuple(scalars)
        )

    @classmethod
    def validate_equal_types(cls, source: Port, target: Port):
        """ Validate a wire between equal types. """
        if not source.obj == target.obj:
            raise AxiomError(messages.NOT_ADJOINT.format(
                source.obj, target.obj))

    @classmethod
    def validate_adjoint_types(cls, source: Port, target: Port):
        """ Validate a wire between adjoint types. """
        adjoint_types = getattr(source.obj, "r", None) == target.obj\
            or source.obj == getattr(target.obj, "r", None)
        if not adjoint_types:
            raise AxiomError(messages.NOT_ADJOINT.format(
                source.obj, target.obj))

    @classmethod
    def validate_wire(cls, source: Port, target: Port):
        """
        Validate type compatibility for a wire between two ports.

        Raises:
            AxiomError : If the types or orientations are incompatible.
        """
        if source.kind.is_positive and target.kind.is_negative:
            cls.validate_equal_types(source, target)
        elif target.kind.is_positive and source.kind.is_negative:
            cls.validate_equal_types(target, source)
        else:
            cls.validate_adjoint_types(source, target)

    @property
    def is_monogamous(self) -> bool:
        """
        Checks monogamy, i.e. every wire connects a positive to a negative
        port, so that the map has no cups or caps. This is the analogue of
        :attr:`Hypergraph.is_monogamous`, in which case the map lives in a
        traced category.

        >>> from discopy.compact import Ty, CMap
        >>> x = Ty("x")
        >>> assert CMap.id(x).is_monogamous
        >>> assert not CMap.cups(x, x.r).is_monogamous
        """
        ports = self.ports
        return all(
            ports[i].kind.is_positive != ports[j].kind.is_positive
            for i, j in enumerate(self.edges) if i < j)

    @property
    def box_edges(self) -> Iterable[tuple[int, int]]:
        """
        The directed wires from the codomain of a box to the domain of
        another, as pairs of source and target port indices.
        """
        ports = self.ports
        for i, j in enumerate(self.edges):
            if i > j:
                continue
            source, target = (i, j) if ports[i].kind.is_positive else (j, i)
            if ports[source].kind == PortKind.COD\
                    and ports[target].kind == PortKind.DOM:
                yield source, target

    @cached_property
    def box_ranks(self) -> tuple[int, ...] | None:
        """
        The rank of each box, i.e. the longest directed path of
        :attr:`box_edges` that reaches it, or ``None`` if there is a cycle.
        """
        ports = self.ports
        dependents = [[] for _ in self.boxes]
        indegree = [0] * len(self.boxes)
        for source, target in self.box_edges:
            i = int(ports[source].depth + 0.5)
            j = int(ports[target].depth - 0.5)
            dependents[i].append(j)
            indegree[j] += 1
        ranks = [0] * len(self.boxes)
        ready = [i for i, degree in enumerate(indegree) if not degree]
        seen = 0
        while ready:
            source = ready.pop()
            seen += 1
            for target in dependents[source]:
                ranks[target] = max(ranks[target], ranks[source] + 1)
                indegree[target] -= 1
                if not indegree[target]:
                    ready.append(target)
        return tuple(ranks) if seen == len(self.boxes) else None

    @property
    def is_acyclic(self) -> bool:
        """
        Whether the directed wiring has no cycles or scalar loops, i.e.
        whether :attr:`box_edges` admits a topological sort.

        >>> from discopy.compact import Ty, Box
        >>> x = Ty("x")
        >>> f = Box("f", x, x).to_map()
        >>> assert f.is_acyclic
        >>> assert not f.trace().is_acyclic
        """
        return not self.loops and self.box_ranks is not None

    @property
    def is_topologically_ordered(self) -> bool:
        """
        Whether every directed wire points forward in the box order, i.e.
        every wire from the codomain of a box to the domain of another goes
        to a box of greater :attr:`Port.depth`.

        >>> from discopy.compact import Ty, Box
        >>> x = Ty("x")
        >>> f, g = Box("f", x, x), Box("g", x, x)
        >>> assert (f.to_map() >> g.to_map()).is_topologically_ordered
        >>> snakes = f.transpose(left=True) >> g.transpose(left=True)
        >>> assert not snakes.to_map().is_topologically_ordered
        """
        ports = self.ports
        return all(
            int(ports[i].depth + 0.5) < int(ports[j].depth - 0.5)
            for i, j in self.box_edges)

    def reorder(self, order: Iterable[int]) -> CMap:
        """ Relabel ports to put boxes in the given order. """
        order = tuple(order)
        boxes = tuple(self.boxes[i] for i in order)
        mapping = list(range(self.n_ports))
        start = len(self.dom)
        for old in order:
            old_ports = self._box_port_indices[old]
            for source, target in zip(
                    old_ports, range(start, start + len(old_ports))):
                mapping[source] = target
            start += len(old_ports)
        edges = self.edges.conjugate(Permutation(mapping))
        return type(self)(
            self.dom, self.cod, boxes, edges,
            loops=self.loops, check=False)

    def topological_order(self) -> CMap:
        """
        Reorder the boxes so that every directed wire points forward.

        This relabels the box order without touching the wiring. It is the
        identity on :attr:`is_causal` maps.

        Raises:
            AxiomError : If the map has a directed cycle, i.e. it is not
                :attr:`is_acyclic`, so that no such order exists.

        >>> from discopy.compact import Ty, Box
        >>> x = Ty("x")
        >>> f, g = Box("f", x, x), Box("g", x, x)
        >>> snakes = (f.transpose(left=True) >> g.transpose(left=True))
        >>> assert not snakes.to_map().is_topologically_ordered
        >>> assert snakes.to_map().topological_order().boxes == (g, f)
        >>> ordered = (f.to_map() >> g.to_map()) @ f.to_map()
        >>> assert ordered.topological_order() is ordered
        >>> f.to_map().trace().topological_order()
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: ... has a directed cycle, ...
        """
        ranks = self.box_ranks
        if ranks is None:
            raise AxiomError(messages.NOT_ACYCLIC.format(self))
        if self.is_topologically_ordered:
            return self
        return self.reorder(tuple(sorted(
            range(len(self.boxes)), key=lambda i: (ranks[i], i))))

    @property
    def is_causal(self) -> bool:
        """
        Checks causality, i.e. the map is :attr:`is_monogamous` with no
        loops and :attr:`is_topologically_ordered`, which implies
        :attr:`is_acyclic`. A causal map lives in a symmetric category,
        i.e. it can be drawn using only swaps.

        >>> from discopy.compact import Ty, Box, CMap
        >>> x = Ty("x")
        >>> f = Box("f", x, x).to_map()
        >>> assert (f >> f).is_causal
        >>> assert not f.trace().is_causal
        >>> assert not CMap.cups(x, x.r).is_causal
        """
        return not self.loops and self.is_monogamous\
            and self.is_topologically_ordered

    def __repr__(self):
        factory = f"cmap.CMap[{factory_name(self.category)}]"
        return factory\
            + f"(dom={self.dom!r}, cod={self.cod!r}, " \
              f"boxes={self.boxes!r}, edges={self.edges!r}, " \
              f"loops={self.loops!r})"

    def __eq__(self, other: object):
        return isinstance(other, CMap) and (
            self.dom, self.cod, self.boxes, self.edges, self.loops
        ) == (
            other.dom, other.cod, other.boxes, other.edges, other.loops)

    def __hash__(self):
        return hash((
            self.dom, self.cod, self.boxes, self.edges, self.loops))

    @classmethod
    def id(cls, dom=None) -> CMap:
        """ The identity map, with each input wired to its output. """
        dom = cls.ob() if dom is None else dom
        n_ports = 2 * len(dom)
        edge = Permutation.from_transpositions(
            ((i, i + len(dom)) for i in range(len(dom))), n_ports)
        return cls(dom, dom, (), edge, check=False)

    @classmethod
    def from_box(cls, box: Box) -> CMap:
        """ Embed a box, wiring its boundary to fresh box ports. """
        left = len(box.dom)
        right = len(box.cod)
        n_ports = 2 * (left + right)
        edge = Permutation.from_transpositions(
            [(i, left + i) for i in range(left)]
            + [(2 * left + right - i - 1, 2 * left + right + i)
               for i in range(right)],
            n_ports)
        return cls(box.dom, box.cod, (box, ), edge, check=False)

    @classmethod
    def from_glued(cls, dom: Ty, cod: Ty,
                   images: Iterable[tuple[CMap, int]]) -> CMap:
        """
        Glue a sequence of maps onto a scan of open wires, in one pass.

        Each wire of the result is a connected component of the wires of the
        ``images``, computed by union-find as they are glued. This is the
        colimit of the diagram of gluings, i.e. the same map as the iterated
        :meth:`then` of the ``images`` whiskered at their offsets, but built
        once rather than rebuilt at every step.

        Parameters:
            dom : The domain of the result.
            cod : The codomain of the result.
            images : Each map to glue, together with the offset at which its
                domain meets the scan.

        >>> from discopy.compact import Ty, Box, CMap
        >>> x = Ty("x")
        >>> f, g = map(CMap.from_box, [Box("f", x, x), Box("g", x, x)])
        >>> CMap.from_glued(x, x, [(f, 0), (g, 0)]) == f >> g
        True
        >>> CMap.from_glued(Ty(), Ty(), [
        ...     (CMap.caps(x.r, x), 0), (CMap.cups(x.r, x), 0)]).loops == (x, )
        True
        """
        wires, ends, objects = [], [], []

        def fresh(obj):
            wires.append(len(wires))
            ends.append([])
            objects.append(obj)
            return len(wires) - 1

        def find(wire):
            while wires[wire] != wire:
                wires[wire] = wires[wires[wire]]
                wire = wires[wire]
            return wire

        def union(source, target):
            source, target = sorted([find(source), find(target)])
            if source != target:
                ends[source] += ends[target]
                wires[target] = source

        scan = []
        for i, obj in enumerate(dom):
            scan.append(fresh(obj))
            ends[scan[i]].append(i)
        boxes, loops, start = (), (), len(dom)
        for image, offset in images:
            arity, coarity = len(image.dom), len(image.cod)
            local, image_ports = {}, image.ports
            for source, target in enumerate(image.edges):
                if source <= target:
                    local[source] = local[target] = fresh(
                        image_ports[source].obj)
            for port in range(arity, image.n_ports - coarity):
                ends[find(local[port])].append(start + port - arity)
            for i in range(arity):
                union(scan[offset + i], local[i])
            scan[offset:offset + arity] = [
                local[image.n_ports - coarity + i] for i in range(coarity)]
            boxes, loops = boxes + image.boxes, loops + image.loops
            start += image.n_ports - arity - coarity
        for i, wire in enumerate(scan):
            ends[find(wire)].append(start + i)

        edges = list(range(start + len(cod)))
        for wire in {find(wire) for wire in range(len(wires))}:
            if not ends[wire]:
                loop = objects[wire]
                loop = loop if isinstance(loop, cls.category.ob)\
                    else cls.ob(loop)
                loops = loops + (
                    loop.r if getattr(loop, "z", 0) % 2 else loop, )
            else:
                source, target = ends[wire]
                edges[source], edges[target] = target, source
        return cls(dom, cod, boxes, edges, loops=loops)

    @classmethod
    def from_wiring(cls, boxes: tuple[Box, ...], wires) -> CMap:
        """
        A closed map given by boxes and wires between pairs
        ``(box_index, port_position)``, where the position counts the
        domain ports of the box followed by its codomain ports.

        Parameters:
            boxes : The boxes of the map.
            wires : Pairs of ``(box_index, port_position)`` pairs.

        Raises:
            ValueError : If a port is left unwired or wired twice.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box, CMap
        >>> x = Ty('x')
        >>> f, g = Box('f', x, x @ x), Box('g', x @ x, x)
        >>> cm = CMap.from_wiring((f, g), [
        ...     ((0, 0), (1, 2)), ((0, 1), (1, 0)), ((0, 2), (1, 1))])
        >>> assert cm.edges.is_fixpoint_free_involution()
        >>> CMap.from_wiring((f, ), [((0, 0), (0, 0))])
        Traceback (most recent call last):
            ...
        ValueError: Port (0, 0) is wired to itself.
        """
        boxes = tuple(boxes)
        starts, n_ports = [], 0
        for box in boxes:
            starts.append(n_ports)
            n_ports += len(box.dom) + len(box.cod)

        def global_index(box_index: int, position: int) -> int:
            box = boxes[box_index]
            arity, coarity = len(box.dom), len(box.cod)
            if not 0 <= position < arity + coarity:
                raise ValueError(
                    f"Box {box_index} has no port {position}.")
            if position < arity:
                return starts[box_index] + position
            return starts[box_index] + arity\
                + (coarity - 1 - (position - arity))

        pairs, seen = [], set()
        for (one, other) in wires:
            i, j = global_index(*one), global_index(*other)
            if i == j:
                raise ValueError(f"Port {one} is wired to itself.")
            for port, position in ((i, one), (j, other)):
                if port in seen:
                    raise ValueError(f"Port {position} is wired twice.")
            seen.update((i, j))
            pairs.append((i, j))
        if len(seen) != n_ports:
            missing = sorted(set(range(n_ports)) - seen)
            raise ValueError(f"Ports {missing} are left unwired.")
        edges = Permutation.from_transpositions(pairs, n_ports)
        return cls(cls.ob(), cls.ob(), boxes, edges)

    @classmethod
    def from_diagram(cls, old: Diagram) -> CMap:
        """
        Turn a :class:`Diagram` into a :class:`CMap`.

        Structure available at the map's categorical level becomes wiring;
        structure from the next level remains represented by boxes.

        The image of each box is computed by the functor into ``cls``, then
        the images are glued in a single pass with :meth:`from_glued`.

        >>> from discopy.braided import Ty, Braid
        >>> from discopy.monoidal import CMap
        >>> x, y = map(Ty, "xy")
        >>> CMap.from_diagram(Braid(x, y)).boxes == (Braid(x, y),)
        True
        >>> from discopy.symmetric import Ty as STy, Swap
        >>> x, y = map(STy, "xy")
        >>> Swap(x, y).to_map().boxes
        ()
        """
        category = type(old).ar
        factory = cls if cls.category is category else cls[category]
        functor = (factory.functor if cls.category is None else cls.functor)(
            ob_map=lambda typ: typ, ar_map=factory.from_box,
            dom=category, cod=factory)
        return factory.from_glued(old.dom, old.cod, [
            (functor(box), offset)
            for layer in old.inside
            for box, offset in layer.boxes_and_offsets])

    @classmethod
    def swap(cls, left: Ty, right: Ty) -> CMap:
        """ The symmetry encoded as boundary wiring. """
        dom, cod = left @ right, right @ left
        left_len, right_len = len(left), len(right)
        output_start = len(dom)
        edge = Permutation.from_transpositions(
            [(i, output_start + right_len + i)
             for i in range(left_len)]
            + [(left_len + i, output_start + i)
               for i in range(right_len)],
            2 * len(dom))
        return cls(dom, cod, (), edge, check=False)

    cup_factory = classmethod(lambda cls, left, right: cls.from_box(
        cls.category.cup_factory(left, right)))
    cap_factory = classmethod(lambda cls, left, right: cls.from_box(
        cls.category.cap_factory(left, right)))

    @classmethod
    def cups(cls, left: Ty, right: Ty) -> CMap:
        """ A cup encoded as boundary wiring between adjoint types. """
        assert_isinstance(left, Pregroup)
        assert_isinstance(right, Pregroup)
        left.assert_isadjoint(right)
        size = len(left)
        edge = Permutation.from_transpositions(
            ((i, size + size - 1 - i) for i in range(size)),
            2 * size)
        return cls(left @ right, cls.ob(), (), edge, check=False)

    @classmethod
    def caps(cls, left: Ty, right: Ty) -> CMap:
        """ A cap encoded as boundary wiring between adjoint types. """
        assert_isinstance(left, Pregroup)
        assert_isinstance(right, Pregroup)
        right.assert_isadjoint(left)
        size = len(left)
        edge = Permutation.from_transpositions(
            ((i, size + size - 1 - i) for i in range(size)),
            2 * size)
        return cls(cls.ob(), left @ right, (), edge, check=False)

    @classmethod
    def copy(cls, typ: Ty, n: int = 2) -> CMap:
        """ Copy is kept as a box: one input cannot wire to many outputs. """
        return cls.from_box(cls.category.copy(typ, n))

    @classmethod
    def merge(cls, typ: Ty, n: int = 2) -> CMap:
        """ Merge is kept as a box: many inputs cannot wire to one output. """
        return cls.from_box(cls.category.merge(typ, n))

    @classmethod
    def discard(cls, typ: Ty) -> CMap:
        """ Discard is kept as a box. """
        return cls.copy(typ, 0)

    @classmethod
    def ev(cls, base: Ty, exponent: Ty, left: bool = True) -> CMap:
        """
        Evaluation is kept as an explicit box by default, or comes from the
        wiring of cups when the host category is rigid.
        """
        if issubclass(cls.category, RigidCategory):
            return super().ev(base, exponent, left)
        return cls.from_box(cls.category.ev(base, exponent, left))

    def curry(self, n: int = 1, left: bool = True) -> CMap:
        """
        Currying is kept as an explicit curry box by default, the more
        rigorous representation, or comes from the wiring of caps when the
        host category is rigid.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.

        >>> from discopy.compact import Ty, Box
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z).to_map()
        >>> assert f.curry(left=False).uncurry(left=False) == f
        >>> f.curry(left=False).draw(
        ...     doctest="docs/_static/cmap/compact-curry.dot", show=False)

        .. graphviz:: /_static/cmap/compact-curry.dot
            :align: center
        """
        if issubclass(self.category, RigidCategory):
            return super().curry(n, left)
        if n < 0 or n > len(self.dom):
            raise ValueError
        if not n:
            return self
        return self.from_box(self.category.curry_factory(
            self.to_diagram(), n, left))

    def base_and_exponent(self, n: int, left: bool) -> tuple[Ty, Ty]:
        """
        The base and exponent that :meth:`uncurry` evaluates, read off the
        codomain as in the host category.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.
        """
        return self.category.base_and_exponent(self, n, left)

    l = property(lambda self: self.transpose(left=True))
    r = property(lambda self: self.transpose(left=False))

    def dagger(self) -> CMap:
        """
        Reverse a combinatorial map: swap the boundary, dagger each box in
        reverse order and conjugate the edges by the port relabeling.

        Boundary ports keep their order while each box block is reversed:
        the clockwise port order of a daggered box is the reversed clockwise
        order of the original.

        >>> from discopy.compact import Ty, Box
        >>> x, y = map(Ty, "xy")
        >>> f, g = Box('f', x, y @ y), Box('g', y @ y, x)
        >>> assert (f >> g).dagger().to_map() == (f >> g).to_map().dagger()
        >>> assert (f >> g).to_map().dagger().dagger() == (f >> g).to_map()
        """
        n, n_dom, n_cod = self.n_ports, len(self.dom), len(self.cod)
        boxes = tuple(box.dagger() for box in reversed(self.boxes))
        dom_mapping = list(range(n - n_dom, n))
        box_mapping = list(reversed(range(n_cod, n - n_dom)))
        cod_mapping = list(range(n_cod))
        mapping = dom_mapping + box_mapping + cod_mapping
        edges = self.edges.conjugate(Permutation(mapping))
        return type(self)(
            self.cod, self.dom, boxes, edges,
            loops=self.loops, check=False)

    @classmethod
    def spiders(
            cls, n_legs_in: int, n_legs_out: int,
            typ: Ty, phases=None) -> CMap:
        """
        Spiders are kept as boxes, including their phase data.

        Example
        -------
        >>> from discopy.tensor import CMap, Dim, Tensor
        >>> assert CMap.spiders(1, 2, Dim(2, 3)).eval().is_close(
        ...     Tensor.spiders(1, 2, Dim(2, 3)))
        """
        return cls.from_box(cls.category.spiders(
            n_legs_in, n_legs_out, typ, phases))

    @unbiased
    def then(self, other: CMap) -> CMap:
        """
        Compose maps by gluing output ports to input ports.

        Closed components created by gluing are retained in :attr:`loops`.

        >>> from discopy.compact import Ty, CMap
        >>> x = Ty("x")
        >>> scalar = CMap.caps(x.r, x) >> CMap.cups(x.r, x)
        >>> scalar.boxes
        ()
        >>> scalar.loops == (x,)
        True
        """
        if not self.cod == other.dom:
            raise AxiomError(messages.TYPE_ERROR.format(other.dom, self.cod))
        dom, cod = self.dom, other.cod
        boxes = self.boxes + other.boxes
        edge = self.edges.tensor(other.edges)
        ports = self.ports + other.ports
        glue = Permutation.id(self.n_ports - len(self.cod)).tensor(
            Permutation.swap(len(self.cod), len(other.dom)),
            Permutation.id(other.n_ports - len(other.dom)))
        edge, new_scalars = self.splice(
            edge, glue, ports)
        loops = self.loops + other.loops + new_scalars
        return type(self)(
            dom, cod, boxes, edge, loops=loops, check=False)

    def trace(self, n: int = 1, left: bool = False) -> CMap:
        """
        Trace boundary wires by splicing the selected inputs and outputs.

        Parameters:
            n : The number of wires to trace.
            left : Whether to trace the leftmost rather than rightmost wires.
        """
        if n < 0:
            raise ValueError
        if not n:
            return self
        if n > min(len(self.dom), len(self.cod)):
            raise ValueError

        if left:
            dom, cod = self.dom[n:], self.cod[n:]
            traced_inputs = range(n)
            traced_outputs = range(
                self.n_ports - len(self.cod),
                self.n_ports - len(self.cod) + n)
        else:
            dom, cod = self.dom[:-n], self.cod[:-n]
            traced_inputs = range(len(dom), len(self.dom))
            traced_outputs = range(self.n_ports - n, self.n_ports)

        glue = Permutation.from_transpositions(
            zip(traced_inputs, traced_outputs), self.n_ports)
        edge, new_scalars = self.splice(
            self.edges, glue, self.ports)
        loops = self.loops + new_scalars
        return type(self)(
            dom, cod, self.boxes, edge, loops=loops, check=False)

    @unbiased
    def tensor(self, other: CMap) -> CMap:
        """ Tensor product given by disjoint union of the two maps. """
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes = self.boxes + other.boxes
        self_dom, self_cod = len(self.dom), len(self.cod)
        other_dom, other_cod = len(other.dom), len(other.cod)
        self_box_ports = self.n_ports - self_dom - self_cod
        other_box_ports = other.n_ports - other_dom - other_cod
        self_map = (
            tuple(range(self_dom))
            + tuple(range(
                self_dom + other_dom,
                self_dom + other_dom + self_box_ports)))
        other_map = (
            tuple(range(self_dom, self_dom + other_dom))
            + tuple(range(
                self_dom + other_dom + self_box_ports,
                self_dom + other_dom + self_box_ports + other_box_ports)))
        cod_start = self_dom + other_dom + self_box_ports + other_box_ports
        n_ports = self.n_ports + other.n_ports
        self_map += tuple(range(cod_start, cod_start + self_cod))
        other_map += tuple(range(cod_start + self_cod, n_ports))

        edge = self.edges.tensor(other.edges).conjugate(
            Permutation(self_map + other_map))
        return type(self)(
            dom, cod, boxes, edge,
            loops=self.loops + other.loops, check=False)

    def interchange(self, i: int, j: int) -> CMap:
        """
        Interchange boxes at indices ``i`` and ``j``.

        The edges permutation is relabeled so that ports follow the canonical
        order induced by the new box order. Unlike
        :meth:`Diagram.interchange`, the boxes need not commute: the result
        is the same map with its boxes out of order, which
        :meth:`topological_order` can put back.

        >>> from discopy.compact import Ty, Box
        >>> x, y = map(Ty, "xy")
        >>> f, g = Box("f", x, x), Box("g", y, y)
        >>> cmap = f.to_map() @ g.to_map()
        >>> cmap.interchange(0, 1).boxes == (g, f)
        True
        >>> assert not (f.to_map() >> f.to_map()).interchange(
        ...     0, 1).is_topologically_ordered
        """
        order = list(range(len(self.boxes)))
        order[i], order[j] = order[j], order[i]
        return self.reorder(order)

    def merge_inputs(self, indices: tuple[int, ...], box: Box) -> CMap:
        """
        Merge input boundary ports through a box ``x -> x ** k``: the merged
        inputs are replaced by a single new input at position
        ``min(indices)`` wired to the domain of ``box``, and the ``i``-th
        codomain port of ``box`` is wired to the old partner of
        ``indices[i]``.

        Parameters:
            indices : The distinct input positions to merge.
            box : The merging box, with one domain port and ``len(indices)``
                  codomain ports.

        Raises:
            ValueError : If the box does not have the required arity, the
                indices are out of range or not distinct, or two merged
                inputs are wired to each other.

        Example
        -------
        >>> from discopy.symmetric import Ty, Box
        >>> x, y = Ty('x'), Ty('y')
        >>> fm = Box('f', x @ x @ x, y).to_map()
        >>> merged = fm.merge_inputs((0, 2), Box('δ', x, x @ x))
        >>> assert merged.dom == x @ x and len(merged.boxes) == 2
        >>> assert merged.edges.is_fixpoint_free_involution()
        """
        assert_isinstance(box, self.category)
        indices = tuple(indices)
        if len(indices) < 2 or len(box.dom) != 1\
                or len(box.cod) != len(indices):
            raise ValueError(
                f"Expected a box with one input and {len(indices)} outputs, "
                f"got {box}.")
        if len(set(indices)) != len(indices) or not all(
                0 <= i < len(self.dom) for i in indices):
            raise ValueError(f"Expected distinct inputs, got {indices}.")
        if any(self.edges[i] in indices for i in indices):
            raise ValueError(
                f"The inputs {indices} are wired to each other.")

        position = min(indices)
        new_dom = self.ob()
        for i, obj in enumerate(self.dom):
            new_dom = new_dom @ (
                box.dom if i == position
                else self.ob() if i in indices else obj)
        boxes = self.boxes + (box, )

        mapping, new_index, new_input = {}, 0, None
        for i in range(len(self.dom)):
            if i == position:
                new_input = new_index
                new_index += 1
            if i in indices:
                continue
            mapping[i] = new_index
            new_index += 1
        for i in range(len(self.dom), self.n_ports - len(self.cod)):
            mapping[i] = new_index
            new_index += 1

        box_dom = new_index
        box_cods = tuple(
            box_dom + 1 + len(box.cod) - i - 1
            for i in range(len(box.cod)))
        new_index += 1 + len(box.cod)
        for i in range(self.n_ports - len(self.cod), self.n_ports):
            mapping[i] = new_index
            new_index += 1
        n_ports = new_index

        edge_pairs = [(new_input, box_dom)]
        for i, j in enumerate(self.edges):
            if i < j and i not in indices and j not in indices:
                edge_pairs.append((mapping[i], mapping[j]))
        for i, index in enumerate(indices):
            edge_pairs.append((box_cods[i], mapping[self.edges[index]]))
        edges = Permutation.from_transpositions(edge_pairs, n_ports)

        return type(self)(
            new_dom, self.cod, boxes, edges, loops=self.loops)

    def plug_input(
            self, input_index: int, box: Box,
            cod: C0, root_index: int = 0) -> CMap:
        """
        Plug an input boundary and the output root into a new box.

        If ``self : A @ x -> y`` and ``box : y -> z @ x``, then
        ``self.plug_input(i, box, z)`` removes the ``i``-th input, wires the
        old output to the domain of ``box``, wires the removed input to the
        non-root output of ``box``, and leaves ``root_index`` as the new root.

        The new box comes last but wires back to the domain of an earlier box,
        so this can introduce a cycle, which needs a traced category.

        Raises:
            ValueError : If the map or box does not have the required arity,
                or either index is out of range.
        """
        assert_isinstance(box, self.category)
        if len(self.cod) != 1 or len(box.dom) != 1 or len(box.cod) != 2:
            raise ValueError
        if root_index not in [0, 1]:
            raise ValueError
        if input_index < 0 or input_index >= len(self.dom):
            raise ValueError

        old_input, old_output = input_index, self.n_ports - 1
        new_dom = self.ob()
        for i, obj in enumerate(self.dom):
            if i != input_index:
                new_dom = new_dom @ obj
        boxes = self.boxes + (box, )

        mapping, new_index = {}, 0
        for i in range(len(self.dom)):
            if i != old_input:
                mapping[i] = new_index
                new_index += 1
        for i in range(len(self.dom), self.n_ports - len(self.cod)):
            mapping[i] = new_index
            new_index += 1

        box_dom = new_index
        box_outputs = tuple(
            new_index + 1 + len(box.cod) - i - 1
            for i in range(len(box.cod)))
        box_root = box_outputs[root_index]
        box_parameter = box_outputs[1 - root_index]
        new_output = new_index + 3

        edge_pairs = []
        for i, j in enumerate(self.edges):
            if i < j and i not in [old_input, old_output]\
                    and j not in [old_input, old_output]:
                edge_pairs.append((mapping[i], mapping[j]))

        input_partner = self.edges[old_input]
        output_partner = self.edges[old_output]
        if input_partner == old_output:
            edge_pairs.append((box_parameter, box_dom))
        else:
            edge_pairs.append((mapping[input_partner], box_parameter))
            edge_pairs.append((mapping[output_partner], box_dom))
        edge_pairs.append((box_root, new_output))
        edge = Permutation.from_transpositions(edge_pairs, new_output + 1)

        return type(self)(
            new_dom, cod, boxes, edge,
            loops=self.loops)

    def explicit_trace(self, left: bool = False) -> CMap:
        """
        The trace of a map with explicit boxes (trace, cup or cap).

        Parameters:
            left : Whether to trace on the left or right.

        Note
        ----
        When ``category.trace_factory`` is a class, e.g. for symmetric
        diagrams, then the result is just one big trace box wrapped up as a
        map. Otherwise it is a class method, e.g. for compact diagrams, in
        which case we use it to introduce cup and cap boxes.
        """
        type(self).assert_istraced()
        factory = self.category.trace_factory
        if isclass(factory):
            return self.from_box(factory(self.to_diagram(), left))
        return factory.__func__(type(self), self, left)

    def make_monogamous(self) -> CMap:
        """
        Introduce cup and cap boxes to make self :attr:`is_monogamous`,
        i.e. so that every wire connects a positive and a negative port.

        The boxes come from ``category.cup_factory`` and
        ``category.cap_factory``, so this needs a rigid category.

        Example
        -------
        >>> from discopy.compact import Ty, Cup, Cap, CMap
        >>> x = Ty("x")
        >>> assert CMap.cups(x, x.r).make_monogamous()\\
        ...     == CMap.from_box(Cup(x, x.r))
        >>> assert CMap.caps(x.r, x).make_monogamous()\\
        ...     == CMap.from_box(Cap(x.r, x))
        """
        type(self).assert_isrigid()
        ports = self.ports
        for i, j in enumerate(self.edges):
            if i > j or ports[i].kind.is_positive\
                    != ports[j].kind.is_positive:
                continue
            source, target = ports[i].obj, ports[j].obj
            if ports[i].kind.is_positive:
                box = self.category.cup_factory(source, target)
                boxes = self.boxes + (box, )
                insert = self.n_ports - len(self.cod)
                box_wires = [(insert, i), (insert + 1, j)]
            else:
                box = self.category.cap_factory(source, target)
                boxes = (box, ) + self.boxes
                insert = len(self.dom)
                box_wires = [(insert + 1, i), (insert, j)]
            shift = lambda p: p if p < insert else p + 2
            edges = Permutation.from_transpositions(
                [(a, shift(b)) for a, b in box_wires]
                + [(shift(a), shift(b)) for a, b in enumerate(self.edges)
                   if a < b and (a, b) != (i, j)],
                self.n_ports + 2)
            return type(self)(
                self.dom, self.cod, boxes, edges,
                loops=self.loops, check=False).make_monogamous()
        return self

    def make_causal(self) -> CMap:
        """
        Make self :attr:`is_causal`, i.e. so that every wire points forward
        and there are no loops. Boxes that are merely out of order are put
        back in order with :meth:`topological_order`; only a wire that closes
        a cycle is cut into a trace, which needs a traced category.

        Example
        -------
        >>> from discopy.traced import Ty, Box, Trace, CMap
        >>> f = Box("f", Ty("x"), Ty("x"))
        >>> assert f.to_map().trace().make_causal()\\
        ...     == CMap.from_box(Trace(f))
        """
        if not self.is_monogamous:
            return self.make_monogamous().make_causal()
        if self.is_acyclic:
            return self.topological_order()
        type(self).assert_istraced()

        ports = self.ports
        cuts = [
            (i, j, ports[i].obj) for i, j in self.box_edges
            if int(ports[i].depth + 0.5) >= int(ports[j].depth - 0.5)]

        n_traces = len(self.loops) + len(cuts)
        shift = lambda p: p if p < len(self.dom) else p + n_traces
        typ = self.ob().tensor(*self.loops, *(obj for _, _, obj in cuts))
        boundary = [
            (len(self.dom) + i, self.n_ports + n_traces + i)
            for i in range(len(self.loops))]
        for i, (source_port, target_port, _) in enumerate(
                cuts, len(self.loops)):
            boundary += [
                (len(self.dom) + i, shift(target_port)),
                (shift(source_port), self.n_ports + n_traces + i)]
        cut_wires = {
            tuple(sorted(wire[:2])) for wire in cuts}
        edges = Permutation.from_transpositions(
            boundary + [
                (shift(a), shift(b)) for a, b in enumerate(self.edges)
                if a < b and (a, b) not in cut_wires],
            self.n_ports + 2 * n_traces)
        result = type(self)(
            self.dom @ typ, self.cod @ typ, self.boxes, edges,
            check=False)
        for _ in range(n_traces):
            result = result.explicit_trace()
        return result

    def to_compact(self) -> CMap:
        """
        Open every curry bubble into its argument followed by the dagger of
        :meth:`ev`, traced over the curried wires: the map is decoded and
        the image of each box is glued back, like :meth:`from_diagram`.

        Example
        -------
        >>> from discopy.closed import Ty, Box, CMap
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> assert f.to_map().curry().to_compact()\\
        ...     == (f.to_map() >> CMap.ev(z, y).dagger()).trace()
        """
        curry_factory = self.category.curry_factory
        if not any(isinstance(box, curry_factory) for box in self.boxes):
            return self
        functor = self.functor(
            ob_map=lambda typ: typ, ar_map=type(self).from_box,
            dom=self.category, cod=type(self))

        def image(box):
            if not isinstance(box, curry_factory):
                return functor(box)
            exponent = box.cod.exponent
            return (type(self).from_diagram(box.arg).to_compact()
                    >> self.ev(box.cod.base, exponent, box.left).dagger()
                    ).trace(len(exponent), left=not box.left)

        diagram = self.to_diagram()
        return type(self).from_glued(diagram.dom, diagram.cod, [
            (image(box), offset)
            for layer in diagram.inside
            for box, offset in layer.boxes_and_offsets])

    def eval(self, *args, **params):
        """
        Evaluate the map directly with the ``eval`` of the host category,
        e.g. contract a tensor map in a single ``einsum`` call, see
        :meth:`discopy.tensor.Diagram.eval`.
        """
        return self.category.eval(self, *args, **params)

    def to_diagram(self) -> Diagram:
        """
        Downgrade to a diagram directly, preserving box orientation.

        This is where a map has to be a morphism of :attr:`category`: its
        cups, caps and traces are made explicit with :meth:`make_monogamous`
        and :meth:`make_causal`, which raise unless the category is rigid
        resp. traced, then its boxes are put in the order in which they are
        applied with :meth:`topological_order`.

        >>> from discopy.monoidal import Ty, Box, CMap
        >>> x = Ty("x")
        >>> f = Box("f", x, x)
        >>> CMap(Ty(), Ty(), (f, ), (1, 0)).to_diagram()
        Traceback (most recent call last):
        ...
        discopy.utils.AxiomError: monoidal.Diagram has no traces for the \
cycles of this map.

        What remains is decoded like in :meth:`Hypergraph.to_diagram`: we scan
        the currently open wires from left to right, for each box we swap its
        domain wires to the front, apply the box, and replace the consumed
        domain labels by the box codomain labels.

        >>> from discopy.compact import Ty, Box, CMap
        >>> x, y = map(Ty, "xy")
        >>> cmap = Box("f", x, y).to_map()
        >>> cmap.to_diagram().to_map() == cmap
        True
        """
        if not self.is_causal:
            return self.make_causal().to_diagram()

        edge_wire = {}
        for i, j in enumerate(self.edges):
            if i <= j:
                edge_wire[i] = edge_wire[j] = len(edge_wire) // 2

        def swap(left, right):
            if not issubclass(self.category, SymmetricCategory):
                raise AxiomError(messages.NOT_SYMMETRIC.format(
                    factory_name(self.category)))
            return self.category.swap(left, right)

        diagram = self.category.id(self.dom)
        scan = [edge_wire[i] for i in range(len(self.dom))]
        for depth, box in enumerate(self.boxes):
            box_ports = self._box_port_indices[depth]
            dom_ports = box_ports[:len(box.dom)]
            cod_ports = tuple(reversed(box_ports[len(box.dom):]))
            dom_wires = [edge_wire[i] for i in dom_ports]
            cod_wires = [edge_wire[i] for i in cod_ports]

            offset = None
            for i, wire_id in enumerate(dom_wires):
                j = scan.index(wire_id)
                if i == 0:
                    offset = j
                elif j != offset + i:
                    if j > offset + i:
                        diagram >>= diagram.cod[:offset + i] @ swap(
                            diagram.cod[offset + i:j], diagram.cod[j]
                        ) @ diagram.cod[j + 1:]
                        scan = (scan[:offset + i] + scan[j:j + 1]) + (
                            scan[offset + i:j] + scan[j + 1:])
                    else:
                        diagram >>= diagram.cod[:j] @ swap(
                            diagram.cod[j], diagram.cod[j + 1:offset + i]
                        ) @ diagram.cod[offset + i:]
                        scan = (scan[:j] + scan[j + 1:offset + i]) + (
                            scan[j:j + 1] + scan[offset + i:])
                        offset -= 1

            offset = len(scan) if offset is None else offset
            scan = scan[:offset] + cod_wires + scan[offset + len(box.dom):]
            diagram >>= diagram.cod[:offset] @ box @ diagram.cod[
                offset + len(box.dom):]

        cod_wires = [
            edge_wire[self.n_ports - len(self.cod) + i]
            for i in range(len(self.cod))]
        for i, wire_id in enumerate(cod_wires):
            j = scan.index(wire_id)
            if i < j:
                diagram >>= diagram.cod[:i] @ swap(
                    diagram.cod[i:j], diagram.cod[j:j + 1]
                ) @ diagram.cod[j + 1:]
                scan = scan[:i] + scan[j:j + 1] + scan[i:j] + scan[j + 1:]
        return diagram

    def to_term(self):
        """
        Extract the linear lambda term encoded by a rooted trivalent map,
        i.e. the inverse of Zeilberger's isomorphism from linear lambda terms
        to rooted trivalent maps, see :meth:`discopy.closed.TermBase.to_map`.

        The single output port is the root and the input ports are the free
        variables of the term. Box labels and the direction of the wires are
        ignored: the term structure is recovered with the following naive
        algorithm. For each root node we remove it from the map; if the
        result is disconnected then it was an application and we recurse with
        the two disconnected subtrees as function and argument; if the result
        is connected then it was an abstraction, we introduce a fresh
        variable as a new node on the input side and recurse with the subtree
        at the output side as body. Which subtree is which is determined by
        the cyclic order of the ports around the removed node, starting from
        the port facing the root.

        Variable names are recovered from the ``varname`` attribute attached
        to the objects carried by the ports, see
        :func:`discopy.biclosed.annotate`; fresh names are generated from a
        global counter in case the attribute is absent. The result is a
        :class:`discopy.closed.Term` typed by unification, with fresh atomic
        types for the type variables left unconstrained.

        Example
        -------
        >>> from discopy.closed import Ty
        >>> term = Ty("a")(lambda v: v)
        >>> assert term.to_map().to_term() == term
        """
        # Imported here to avoid a circular dependency with biclosed.
        from discopy import biclosed, closed

        if len(self.cod) != 1 or self.loops:
            raise ValueError(
                "Expected a rooted map with a single output port and no "
                f"scalars, got {self}.")
        for box in self.boxes:
            if len(box.dom) + len(box.cod) != 3:
                raise ValueError(f"Expected trivalent boxes, got {box}.")

        ports, edges = self.ports, self.edges
        box_ports = self._box_port_indices
        vertex_of = {
            port: vertex for vertex, indices in enumerate(box_ports)
            for port in indices}

        subst, tvars, atoms = {}, count(), {}

        def resolve(typ):
            while isinstance(typ, int) and typ in subst:
                typ = subst[typ]
            return typ

        def unify(left, right):
            # No occurs check: linear terms are always simply typeable.
            left, right = resolve(left), resolve(right)
            if isinstance(left, int) or isinstance(right, int):
                if not isinstance(left, int):
                    left, right = right, left
                if left != right:
                    subst[left] = right
                return
            unify(left[0], right[0])
            unify(left[1], right[1])

        leaf, variables, visited = {}, {}, set()

        def new_variable(port):
            obj = ports[port].obj
            names = {
                getattr(x, "varname", None)
                for x in getattr(obj, "inside", (obj, ))}
            name = names.pop() if len(names) == 1 else None
            name = biclosed.fresh_name() if name is None else name
            variable = (name, next(tvars))
            leaf[port] = variable
            return variable

        for port, _ in enumerate(self.dom):
            new_variable(port)

        def connected(source, target, live):
            seen, todo = {source}, [source]
            while todo:
                vertex = todo.pop()
                if vertex == target:
                    return True
                for port in box_ports[vertex]:
                    other = vertex_of.get(edges[port])
                    if other in live and other not in seen:
                        seen.add(other)
                        todo.append(other)
            return False

        def extract(entry, live):
            if entry in leaf:
                return ('var', leaf[entry]), leaf[entry][1]
            vertex = vertex_of[entry]
            visited.add(vertex)
            live = live - {vertex}
            cycle = box_ports[vertex]
            index = cycle.index(entry)
            first, second = cycle[(index + 1) % 3], cycle[(index + 2) % 3]
            if edges[first] == second:  # The identity abstraction.
                variable = new_variable(second)
                tree = ('abs', variable, ('var', variable))
                return tree, (variable[1], variable[1])
            far_first, far_second = edges[first], edges[second]
            first_vertex = vertex_of.get(far_first)
            second_vertex = vertex_of.get(far_second)
            if first_vertex in live and second_vertex in live\
                    and connected(first_vertex, second_vertex, live):
                variable = new_variable(second)
                body, body_type = extract(far_first, live)
                return ('abs', variable, body), (variable[1], body_type)
            func, func_type = extract(far_first, live)
            args, args_type = extract(far_second, live)
            result_type = next(tvars)
            unify(func_type, (args_type, result_type))
            return ('app', func, args), result_type

        root = edges[self.n_ports - 1]
        tree, _ = extract(root, set(range(len(self.boxes))))
        if len(visited) != len(self.boxes):
            raise ValueError(f"Expected a connected rooted map, got {self}.")

        def to_ty(typ):
            typ = resolve(typ)
            if isinstance(typ, tuple):
                return to_ty(typ[0]) >> to_ty(typ[1])
            if typ not in atoms:
                index = len(atoms)
                atoms[typ] = closed.Ty(ascii_lowercase[index % 26] + (
                    "" if index < 26 else str(index // 26)))
            return atoms[typ]

        def build(node):
            if node[0] == 'var':
                name, tvar = node[1]
                if node[1] not in variables:
                    variables[node[1]] = closed.Variable(name, to_ty(tvar))
                return variables[node[1]]
            if node[0] == 'app':
                return build(node[1])(build(node[2]))
            variable, body = build(('var', node[1])), build(node[2])
            return closed.Abstraction(variable, body)

        return build(tree)

    def to_hypergraph(self):
        """
        Forget orientation and return the underlying bijective hypergraph
        given by the edge permutation. See documentation of
        :func:``Hypergraph.from_map`` for an example.
        """
        return hypergraph.Hypergraph[self.category].from_map(self)

    def to_dot(
            self, engine="dot", seed=None, graph_attr=None,
            port_indices=False) -> str:
        """
        Encode the combinatorial map as Graphviz DOT.

        The drawing has HTML-table nodes for the boundary interfaces and for
        each box, with one table port for each object in the signature, and
        one direct edge per 2-cycle of ``edges``.

        Parameters:
            engine : The Graphviz layout engine.
            seed : An optional Graphviz layout seed.
            graph_attr : Additional graph attributes.
            port_indices : Whether to display port indices.

        >>> from discopy.compact import Ty, CMap
        >>> CMap.id(Ty("x")).to_dot().startswith("graph cmap")
        True
        """
        attrs = {
            "layout": engine,
            "rankdir": "TB",
            "overlap": "false",
            "splines": "true",
            "outputorder": "edgesfirst",
            "bgcolor": "white",
            "fontname": "DejaVu Sans",
            "margin": "0.04",
        } | (graph_attr or {})
        if seed is not None:
            attrs["start"] = str(seed)

        class Html:
            def __init__(self, value):
                self.value = value

        def escape(value):
            return str(value).replace("\\", "\\\\").replace('"', r'\"')

        def escape_html(value):
            return str(value).replace("&", "&amp;").replace(
                "<", "&lt;").replace(">", "&gt;").replace(
                    '"', "&quot;")

        def attr_string(attributes):
            return ", ".join(
                f'{key}=<{value.value}>' if isinstance(value, Html)
                else f'{key}="{escape(value)}"'
                for key, value in sorted(attributes.items()))

        def boundary_label(port_index):
            return f"{port_index}" if port_indices else ""

        def boundary_cell(port_index, port):
            tooltip = escape_html(f"{port.kind} {port.i}: {port.obj}")
            return (
                f'<TD PORT="p{port_index}" TOOLTIP="{tooltip}" '
                f'BORDER="0" CELLPADDING="4">'
                f'{escape_html(boundary_label(port_index))}</TD>')

        def boundary_table(port_indices):
            return (
                '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR>'
                + "".join(
                    boundary_cell(port_index, self.ports[port_index])
                    for port_index in port_indices)
                + "</TR></TABLE>")

        def port_cell(port_index, port, colspan, width):
            tooltip = escape_html(
                f"{port.kind} {port.i}: {port.obj} ({port.side})")
            text = escape_html(port_index) if port_indices else ""
            cellpadding = 2 if port_indices else 0
            height = 18 if port_indices else 0
            fixedsize = ' FIXEDSIZE="TRUE"' if port_indices else ""
            return (
                f'<TD PORT="p{port_index}" TOOLTIP="{tooltip}" '
                f'BORDER="0" CELLPADDING="{cellpadding}" '
                f'COLSPAN="{colspan}" WIDTH="{width}" '
                f'HEIGHT="{height}"{fixedsize}>{text}</TD>')

        def port_row(port_indices, grid, box_width):
            colspan = grid // len(port_indices)
            width = round(box_width / len(port_indices))
            return "<TR>" + "".join(
                port_cell(
                    port_index, self.ports[port_index], colspan, width)
                for port_index in port_indices) + "</TR>"

        def box_table(vertex, box):
            box_ports = self._box_port_indices[vertex]
            dom_ports = box_ports[:len(box.dom)]
            cod_ports = tuple(reversed(box_ports[len(box.dom):]))
            dom_arity, cod_arity = len(dom_ports), len(cod_ports)
            grid = lcm(dom_arity or 1, cod_arity or 1)
            box_width = 18 * max(dom_arity, cod_arity, 1)
            rows = []
            if dom_ports:
                rows.append(port_row(dom_ports, grid, box_width))
            box_label = getattr(box, "drawing_name", box.name)
            rows.append(
                f'<TR><TD BORDER="1" CELLPADDING="6" '
                f'COLSPAN="{grid}" WIDTH="{box_width}">'
                f'{escape_html(box_label)}</TD></TR>')
            if cod_ports:
                rows.append(port_row(cod_ports, grid, box_width))
            return (
                '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
                + "".join(rows) + "</TABLE>")

        node_attrs = {
            "color": "black", "fontname": "DejaVu Sans", "fontsize": "12",
            "margin": "0", "shape": "plain"}
        edge_attrs = {
            "color": "black", "fontname": "DejaVu Sans", "fontsize": "9",
            "headclip": "true", "penwidth": "1.4", "tailclip": "true"}
        lines = [
            "graph cmap {",
            f"  graph [{attr_string(attrs)}];",
            f"  node [{attr_string(node_attrs)}];",
            f"  edge [{attr_string(edge_attrs)}];",
        ]

        port_nodes = {}
        for vertex in range(len(self.boxes)):
            box = self.boxes[vertex]
            attributes = {"label": Html(box_table(vertex, box))}
            lines.append(
                f"  v{vertex} [{attr_string(attributes)}];")
            for port_index in self._box_port_indices[vertex]:
                compass = "n" if self.ports[
                    port_index].kind == "dom" else "s"
                port_nodes[port_index] = (
                    f"v{vertex}:p{port_index}:{compass}")
        input_ports = [
            i for i, port in enumerate(self.ports)
            if port.kind == PortKind.INPUT]
        output_ports = [
            i for i, port in enumerate(self.ports)
            if port.kind == PortKind.OUTPUT]
        for name, ports, compass in [
                (PortKind.INPUT, input_ports, "s"),
                (PortKind.OUTPUT, output_ports, "n")]:
            if not ports:
                continue
            attributes = {"label": Html(boundary_table(ports))}
            lines.append(f"  {name} [{attr_string(attributes)}];")
            for port_index in ports:
                port_nodes[port_index] = f"{name}:p{port_index}:{compass}"

        for rank, name, ports in [
                ("min", "input", input_ports),
                ("max", "output", output_ports)]:
            if ports:
                lines.append(f"  {{ rank={rank}; {name}; }}")

        for i, loop in enumerate(self.loops):
            attributes = {
                "label": "",
                "width": "0.08",
                "height": "0.08",
                "shape": "point",
                "tooltip": f"loop {i}: {loop}"}
            lines.append(f"  loop{i} [{attr_string(attributes)}];")
            attributes = {"len": "0.85", "label": loop}
            lines.append(
                f"  loop{i} -- loop{i} "
                f"[{attr_string(attributes)}];")

        def node_name(port_index):
            return port_nodes[port_index]

        def port_label(port_index):
            return self.ports[port_index].obj

        def edge_labels(left, right):
            left_label, right_label = port_label(left), port_label(right)
            if left_label == right_label:
                return {"label": left_label}
            return {"taillabel": left_label, "headlabel": right_label}

        for i, j in enumerate(self.edges):
            if i < j:
                attributes = {
                    "len": "0.85", "labeldistance": "1.6"} | edge_labels(i, j)
                lines.append(
                    f'  {node_name(i)} -- {node_name(j)} '
                    f'[{attr_string(attributes)}];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def draw(
            self, path=None, doctest=None, engine="dot", format=None,
            seed=None, show=None, graph_attr=None, port_indices=False,
            block=True, tol=20):
        """
        Draw as a combinatorial map using Graphviz.

        This is intended for map-like pictures rather than the usual DisCoPy
        box-and-wire drawing.

        If ``path`` ends in ``.dot`` or ``.gv``, write DOT source. Otherwise,
        render with Graphviz. When ``show`` is true, display the rendered graph
        in a matplotlib window.

        Parameters:
            path : The output path, or ``None`` to display the map.
            engine : The Graphviz layout engine.
            format : The rendered format, inferred from ``path`` by default.
            seed : An optional Graphviz layout seed.
            show : Whether to display the rendered image.
            graph_attr : Additional Graphviz graph attributes.
            port_indices : Whether to display port indices.
            block : Whether displaying blocks execution.

        Scalar loops are drawn as dots with a loop, but the combinatorial map
        structure does not let us retain inclusion of such loops:

        >>> from discopy.compact import Ty, CMap
        >>> x, y, z = map(Ty, "xyz")
        >>> (CMap.caps((x @ y).r, x @ y) >> CMap.cups((x @ y).l, x @ y)).draw(
        ...     doctest="docs/_static/cmap/scalar-loop.dot", show=False)

        .. graphviz:: /_static/cmap/scalar-loop.dot
            :align: center
        """
        dot = self.to_dot(
            engine=engine, seed=seed, graph_attr=graph_attr,
            port_indices=port_indices)

        from discopy.drawing import backend
        path, compare = backend.doctest_or_path(path, doctest)
        show = show if show is not None else path is None
        if path is not None:
            path_str = str(path)
            suffix = path_str.rsplit(".", 1)[-1].lower()\
                if "." in path_str else ""
            if suffix in ["dot", "gv"]:
                def save(actual_path):
                    with open(
                            actual_path, "w", encoding="utf-8",
                            newline="\n") as stream:
                        stream.write(dot)
                if compare:
                    backend.save_and_compare(path, save, tol=tol)
                else:
                    save(path)
                return

        executable = shutil.which(engine) or shutil.which("dot")
        if executable is None:
            raise RuntimeError(
                f"Graphviz executable {engine!r} was not found.")

        if path is not None:
            output_format = format or suffix or "svg"

            def save(actual_path):
                subprocess.run(
                    [executable, f"-T{output_format}", "-o", actual_path],
                    input=dot.encode(), check=True)
            if compare:
                backend.save_and_compare(path, save, tol=tol)
            else:
                save(path)
        if not show:
            return

        png = subprocess.run(
            [executable, "-Tpng"], input=dot.encode(),
            capture_output=True, check=True).stdout
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
        image = mpimg.imread(BytesIO(png), format="png")
        height, width = image.shape[:2]
        figsize = (max(width / 100, 1), max(height / 100, 1))
        figure, axis = plt.subplots(figsize=figsize, facecolor="white")
        axis.imshow(image)
        axis.axis("off")
        figure.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.show(block=block)
        return
