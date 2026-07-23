# -*- coding: utf-8 -*-

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
from enum import StrEnum

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from itertools import count
from math import lcm
from string import ascii_lowercase
import shutil
import subprocess
from typing import Any, TYPE_CHECKING, ClassVar, Literal

from discopy import messages, hypergraph
from discopy.cat import Ob
from discopy.abc import CompactCategory, NamedGeneric, Pregroup
from discopy.python.finset import Permutation
from discopy.utils import (
    AxiomError,
    assert_isinstance,
    classproperty,
    factory_name,
    unbiased,
)

if TYPE_CHECKING:
    from discopy.monoidal import Ob, Ty, Diagram, Box


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
    side: Literal["up"] | Literal["down"]

    @property
    def direction(self) -> Literal["up"] | Literal["down"]:
        """ The adjoint-aware direction of the wire at the port. """
        is_adjoint = bool(getattr(self.obj, "z", 0) % 2)
        if self.kind.is_input:
            return "down" if is_adjoint else "up"
        return "up" if is_adjoint else "down"


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

    Four knobs are available to restrict the structure:

    * ``require_planar``: the port orientation give us a way to easily compute
      whether the map is planar by computing its component-wise Euler
      characteristic, i.e. disallow swaps;
    * ``require_causal``: checks that the edges are in causal order, i.e.
      they link positive ports to negative ports with higher rank, i.e. no
      traced wires;
    * ``require_oriented``: checks that we connect positive to negative wires,
      and disallow same-polarity pairings, i.e. we can enforce
      :math:`e; m = -m` to disallow cups and caps;
    * ``require_connected``: ensures the map forms a single connected component

    Note that ``require_causal`` implies ``require_oriented`` since cups and
    caps give rise to traced structure. We can therefore represent the
    categorical structures we can guarantee by the following diagram:

    .. tikz::
        :align: center

        \begin{tikzpicture}[
          x={(2.6cm,0cm)}, y={(0cm,1.6cm)},
          every node/.style={font=\scriptsize, align=center},
          label/.style={font=\tiny, text=gray!35!black},
          edge/.style={draw, -latex}
        ]
          \node[label] at (0,1.8) {causal};
          \node[label] at (1,1.8) {non-causal};
          \node[label] at (2,1.8) {non-oriented};
          \node[label, anchor=east] at (-0.65,0) {monoidal};
          \node[label, anchor=east] at (-0.65,1) {symmetric};

          \node (S)  at (0,0) {spacial};
          \node (Y)  at (0,1) {symmetric};
          \node (T)  at (1,1) {traced};
          \node (P)  at (2,0) {pivotal};
          \node (C)  at (2,1) {compact};

          \draw[edge] (S) -- (Y);
          \draw[edge] (Y) -- (T);
          \draw[edge] (T) -- (C);
          \draw[edge] (S) -- (P);
          \draw[edge] (P) -- (C);
        \end{tikzpicture}

    Note that combinatorial maps as such cannot faithfully represent free
    monoidal diagrams as there is no way to account for the nesting of
    connected components, hence the bottom-left corner rather corresponds
    to the free `spacial` category, i.e. diagrams where entire isolated
    components can go through wires at once without intermediate overlapping
    (see :cite:`Selinger10`).

    Parameters:
        dom : The domain of the map.
        cod : The codomain of the map.
        boxes : The boxes inside the map.
        edges : A fixpoint-free involution on ports.
        offsets : Optional drawing offsets, preserved through conversion.
        loops : The types of closed wire components with no ports.

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
    ...     path="docs/_static/cmap/simple-cmap.svg",
    ...     port_indices=True,
    ...     show=False,
    ... )

    .. image:: /_static/cmap/simple-cmap.svg
        :align: center

    Swaps affect the edge permutation but leave the vertex permutation
    fixed:

    >>> f, g = map(CMap.from_box, [
    ...     Box("f", x @ y, z @ x),
    ...     Box("g", z @ z, z),
    ... ])
    >>> cm = (f >> CMap.swap(z, x)) @ z >> x @ g
    >>> cm.draw(
    ...     path="docs/_static/cmap/swapped-cmap.svg",
    ...     port_indices=True,
    ...     show=False,
    ... )

    .. image:: /_static/cmap/swapped-cmap.svg
        :align: center
    """

    category: ClassVar[Diagram] = None
    require_planar: ClassVar[bool] = False
    require_causal: ClassVar[bool] = False
    require_oriented: ClassVar[bool] = False
    require_connected: ClassVar[bool] = False
    functor = classproperty(lambda cls: cls.category.functor_factory)
    ob = classproperty(lambda cls: cls.category.ob)

    dom: C0
    cod: C0
    offsets: tuple[int, ...]
    loops: tuple[C0, ...]
    edges: Permutation

    def __init__(
            self, dom: C0, cod: C0, boxes: tuple[Box, ...],
            edges: Iterable[int],
            offsets: tuple[int | None, ...] | None = None,
            loops: tuple[C0, ...] = ()):
        assert_isinstance(dom, self.category.ob)
        assert_isinstance(cod, self.category.ob)
        for box in boxes:
            assert_isinstance(box, self.category)
        for loop in loops:
            assert_isinstance(loop, self.category.ob)
        self.dom, self.cod, self.boxes = dom, cod, tuple(boxes)
        self.offsets = offsets or tuple(len(boxes) * [None])
        if len(self.offsets) != len(self.boxes):
            raise ValueError
        self.loops = tuple(loops)

        self.edges = Permutation(edges, len(self.ports))
        self.validate()

    @property
    def ports(self) -> list[Port]:
        """ The ports in canonical orientation order. """
        def port(kind, i, obj, depth):
            if not kind.is_boundary:
                depth += 0.5 if kind.is_input else -0.5
            return Port(
                kind, i=i, obj=obj, depth=depth,
                side="up" if kind.is_input else "down")

        inputs = [port(PortKind.INPUT, i=i, obj=obj, depth=float('-inf'))
                  for i, obj in enumerate(self.dom)]
        box_ports = sum([[
            port(kind, i=i, obj=obj, depth=depth)
            for i, obj in indexed_typ]
            for depth, box in enumerate(self.boxes)
            for kind, indexed_typ in [
                (PortKind.DOM, tuple(enumerate(box.dom))),
                (PortKind.COD, tuple(reversed(tuple(enumerate(box.cod)))))]],
            [])
        outputs = [port(PortKind.OUTPUT, i=i, obj=obj, depth=float('+inf'))
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
        """ Validate the edges involution, wires and required planarity. """
        ports = self.ports
        if not self.edges.is_fixpoint_free_involution():
            raise ValueError

        for i, j in enumerate(self.edges):
            if i > j:
                continue
            type(self).validate_wire(ports[i], ports[j])

        if self.require_causal:
            self.validate_forward_edges(ports)

        if self.require_planar and not self.is_planar:
            raise AxiomError(messages.NOT_PLANAR.format(self))

        if self.require_connected and len(self.connected_components) != 1:
            raise AxiomError(messages.NOT_CONNECTED.format(self))

    @property
    def connected_components(self) -> list[CMap]:
        """ The connected components, with the boundary component first. """
        if not self.n_ports:
            # Avoid recursively rebuilding the same portless component.
            if len(self.boxes) + len(self.loops) <= 1:
                return [self]
            components = [
                type(self)(
                    self.ob(), self.ob(), (box, ), (),
                    offsets=(offset, ))
                for box, offset in zip(self.boxes, self.offsets)]
            components += [
                type(self)(self.ob(), self.ob(), (), (), loops=(loop, ))
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
        offsets_by_component: dict[int, list[int | None]] = {}
        portless_boxes: list[tuple[int, Box, int | None]] = []
        for box_index, (box, offset) in enumerate(zip(
                self.boxes, self.offsets)):
            box_ports = self._box_port_indices[box_index]
            if not box_ports:
                portless_boxes.append((box_index, box, offset))
                continue
            component = component_of[box_ports[0]]
            boxes_by_component.setdefault(component, []).append((
                box_index, box))
            offsets_by_component.setdefault(component, []).append(offset)

        if len(ports_by_component) == 1 and not portless_boxes\
                and not self.loops:
            return [self]

        def make_component(component: int) -> CMap:
            dom = self.dom if component == boundary_component else self.ob()
            cod = self.cod if component == boundary_component else self.ob()
            boxes = tuple(box for _, box in boxes_by_component.get(
                component, ()))
            offsets = tuple(offsets_by_component.get(component, ()))

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
            return type(self)(dom, cod, boxes, edges, offsets=offsets)

        ordered_components = sorted(
            ports_by_component,
            key=lambda component: (
                component != boundary_component,
                min(ports_by_component[component])))
        components = [make_component(component)
                      for component in ordered_components]
        components += [
            type(self)(
                self.ob(), self.ob(), (box, ), (), offsets=(offset, ))
            for _, box, offset in portless_boxes]
        components += [
            type(self)(self.ob(), self.ob(), (), (), loops=(loop, ))
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
        elif cls.require_oriented:
            raise AxiomError
        else:
            cls.validate_adjoint_types(source, target)

    def validate_forward_edges(self, ports: list[Port]):
        """ Validate that box-to-box causal wires are acyclic. """
        graph = {i: set() for i in range(len(self.boxes))}

        def has_path(source: int, target: int) -> bool:
            unseen, seen = [source], set()
            while unseen:
                node = unseen.pop()
                if node == target:
                    return True
                if node in seen:
                    continue
                seen.add(node)
                unseen.extend(graph[node])
            return False

        for i, j in enumerate(self.edges):
            if i > j:
                continue
            left, right = ports[i], ports[j]
            if left.kind.is_positive and right.kind.is_negative:
                source, target = left, right
            elif right.kind.is_positive and left.kind.is_negative:
                source, target = right, left
            else:
                continue
            if source.kind != PortKind.COD or target.kind != PortKind.DOM:
                continue
            source_depth = int(source.depth + 0.5)
            target_depth = int(target.depth - 0.5)
            if source_depth == target_depth:
                continue
            if has_path(target_depth, source_depth):
                raise AxiomError(messages.NOT_TRACEABLE.format(
                    source, target))
            graph[source_depth].add(target_depth)

    def __repr__(self):
        def port_repr(index, port):
            port_depth = getattr(port, "depth", None)
            depth = "" if port_depth is None else f"@{port_depth}"
            return (
                f"{port.kind}{depth}[{port.i}]:{port.obj}:"
                f"{port.side}/{port.direction}"
                f"->{self.edges[index]}")

        ports = tuple(
            port_repr(index, port)
            for index, port in enumerate(self.ports))
        return factory_name(type(self))\
            + f"(dom={self.dom!r}, cod={self.cod!r}, " \
              f"boxes={self.boxes!r}, edges={self.edges!r}, " \
              f"ports={ports!r}, scalars={self.loops!r})"

    def __eq__(self, other: Any):
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
        return cls(dom, dom, (), edge)

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
        return cls(box.dom, box.cod, (box, ), edge)

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
        factory = cls if cls.category is not None else cls[category]
        return factory.functor(
            ob_map=lambda typ: typ, ar_map=factory.from_box,
            dom=category, cod=factory)(old)

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
        return cls(dom, cod, (), edge)

    @classmethod
    def cups(cls, left: Ty, right: Ty) -> CMap:
        """ A cup encoded as boundary wiring between adjoint types. """
        adjoint = left.r if hasattr(left, "r") else left[::-1]
        if adjoint != right:
            raise AxiomError
        size = len(left)
        edge = Permutation.from_transpositions(
            ((i, size + size - 1 - i) for i in range(size)),
            2 * size)
        return cls(left @ right, cls.ob(), (), edge)

    @classmethod
    def caps(cls, left: Ty, right: Ty) -> CMap:
        """ A cap encoded as boundary wiring between adjoint types. """
        adjoint = left.r if hasattr(left, "r") else left[::-1]
        if adjoint != right:
            raise AxiomError
        size = len(left)
        edge = Permutation.from_transpositions(
            ((i, size + size - 1 - i) for i in range(size)),
            2 * size)
        return cls(cls.ob(), left @ right, (), edge)

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
        """ Evaluation kept as a box. """
        return cls.from_box(cls.category.ev(base, exponent, left))

    def curry(self, n: int = 1, left: bool = False) -> CMap:
        """
        Curry a combinatorial map using compact wiring.

        Note:
            This will use the free compact structure obtained from the map
            representation by introducing adjoint ports, even if the host
            category already has compact structure.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.

        >>> from discopy.compact import Ty, Box
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z).to_map()
        >>> assert f.curry().uncurry() == f
        >>> f.curry().draw(
        ...     path="docs/_static/cmap/compact-curry.svg", show=False)

        .. image:: /_static/cmap/compact-curry.svg
            :align: center
        """
        if n < 0 or n > len(self.dom):
            raise ValueError
        if not n:
            return self
        if left:
            base, exponent = self.dom[:-n], self.dom[-n:]
            return base @ self.caps(
                exponent, exponent.l) >> self @ exponent.l
        base, exponent = self.dom[n:], self.dom[:n]
        return self.caps(exponent.r, exponent) @ base >> exponent.r @ self

    def uncurry(self, n: int = 1, left: bool = False) -> CMap:
        """
        Uncurry a combinatorial map.

        Parameters:
            n : The number of objects to uncurry.
            left : Whether to uncurry on the left or right.

        This is inverse to :meth:`curry` when applied on the same side.
        """
        if n < 0 or n > len(self.cod):
            raise ValueError
        if not n:
            return self
        if left:
            base, exponent_l = self.cod[:-n], self.cod[-n:]
            exponent = exponent_l.r
            return self @ exponent >> base @ self.cups(
                exponent.l, exponent)
        exponent_r, base = self.cod[:n], self.cod[n:]
        exponent = exponent_r.l
        return exponent @ self >> self.cups(exponent, exponent.r) @ base

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
        offsets = tuple(reversed(self.offsets))
        sizes = [len(box.dom) + len(box.cod) for box in self.boxes]
        starts = [n_cod + sum(sizes[i + 1:]) for i in range(len(sizes))]
        dom_mapping = list(range(n - n_dom, n))
        box_mapping = sum([
            list(reversed(range(start, start + size)))
            for start, size in zip(starts, sizes)], [])
        cod_mapping = list(range(n_cod))
        mapping = dom_mapping + box_mapping + cod_mapping
        edges = self.edges.conjugate(Permutation(mapping))
        return type(self)(
            self.cod, self.dom, boxes, edges, offsets=offsets,
            loops=self.loops)

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

        Closed components created by gluing are retained in :attr:`scalars`.

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
        offsets = self.offsets + other.offsets

        edge = self.edges.tensor(other.edges)
        ports = self.ports + other.ports
        glue = Permutation.id(self.n_ports - len(self.cod)).tensor(
            Permutation.swap(len(self.cod), len(other.dom)),
            Permutation.id(other.n_ports - len(other.dom)))
        edge, new_scalars = self.splice(
            edge, glue, ports)
        loops = self.loops + other.loops + new_scalars
        return type(self)(
            dom, cod, boxes, edge, offsets=offsets,
            loops=loops)

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
            dom, cod, self.boxes, edge, offsets=self.offsets,
            loops=loops)

    @unbiased
    def tensor(self, other: CMap) -> CMap:
        """ Tensor product given by disjoint union of the two maps. """
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes = self.boxes + other.boxes
        offsets = self.offsets + other.offsets

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

        edge = self.edges.embed(self_map, n_ports).then(
            other.edges.embed(other_map, n_ports))
        return type(self)(
            dom, cod, boxes, edge, offsets=offsets,
            loops=self.loops + other.loops)

    def interchange(self, i: int, j: int) -> CMap:
        """
        Interchange boxes at indices ``i`` and ``j``.

        The edges permutation is relabeled so that ports follow the canonical
        order induced by the new box order.

        >>> from discopy.compact import Ty, Box
        >>> x, y = map(Ty, "xy")
        >>> f, g = Box("f", x, x), Box("g", y, y)
        >>> cmap = f.to_map() @ g.to_map()
        >>> cmap.interchange(0, 1).boxes == (g, f)
        True
        """
        boxes, offsets = list(self.boxes), list(self.offsets)
        boxes[i], boxes[j] = boxes[j], boxes[i]
        offsets[i], offsets[j] = offsets[j], offsets[i]
        boxes, offsets = tuple(boxes), tuple(offsets)

        old_ports = self._box_port_indices
        start = len(self.dom)
        new_ports = {}
        for box_index, box in enumerate(boxes):
            stop = start + len(box.dom @ box.cod)
            old_index = j if box_index == i else i if box_index == j\
                else box_index
            new_ports[old_index] = tuple(range(start, stop))
            start = stop

        mapping = list(range(self.n_ports))
        for old_index, ports in enumerate(old_ports):
            for old, new in zip(ports, new_ports[old_index]):
                mapping[old] = new

        edge = self.edges.conjugate(Permutation(mapping))
        return type(self)(
            self.dom, self.cod, boxes, edge, offsets=offsets,
            loops=self.loops)

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
        offsets = self.offsets + (None, )

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
            new_dom, self.cod, boxes, edges, offsets=offsets,
            loops=self.loops)

    def plug_input(
            self, input_index: int, box: Box,
            cod: C0, root_index: int = 0) -> CMap:
        """
        Plug an input boundary and the output root into a new box.

        If ``self : A @ x -> y`` and ``box : y -> z @ x``, then
        ``self.plug_input(i, box, z)`` removes the ``i``-th input, wires the
        old output to the domain of ``box``, wires the removed input to the
        non-root output of ``box``, and leaves ``root_index`` as the new root.

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
        offsets = self.offsets + (None, )

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
            new_dom, cod, boxes, edge, offsets=offsets,
            loops=self.loops)

    def to_diagram(self) -> Diagram:
        """
        Downgrade to a diagram directly, preserving box orientation.

        The construction scans the currently open wire labels from left to
        right. For each box, it swaps boundary wires until the box domain wires
        are adjacent at the requested offset, applies the box, and replaces
        consumed domain labels by the box codomain labels.

        >>> from discopy.compact import Ty, Box
        >>> x, y = map(Ty, "xy")
        >>> cmap = Box("f", x, y).to_map()
        >>> cmap.to_diagram().to_map() == cmap
        True
        """
        edge_wire = {}
        for i, j in enumerate(self.edges):
            if i <= j:
                edge_wire[i] = edge_wire[j] = len(edge_wire) // 2

        diagram = self.category.id(self.dom)
        scan = [edge_wire[i] for i in range(len(self.dom))]
        for depth, (box, offset) in enumerate(zip(self.boxes, self.offsets)):
            box_ports = self._box_port_indices[depth]
            dom_ports = box_ports[:len(box.dom)]
            cod_ports = tuple(reversed(box_ports[len(box.dom):]))
            dom_wires = [edge_wire[i] for i in dom_ports]
            cod_wires = [edge_wire[i] for i in cod_ports]

            for i, wire_id in enumerate(dom_wires):
                j = scan.index(wire_id)
                if i == 0 and offset is None:
                    offset = 0
                if j > offset + i:
                    diagram >>= diagram.cod[:offset + i] @ diagram.swap(
                        diagram.cod[offset + i:j], diagram.cod[j]
                    ) @ diagram.cod[j + 1:]
                    scan = (scan[:offset + i] + scan[j:j + 1]) + (
                        scan[offset + i:j] + scan[j + 1:])
                elif j < offset + i:
                    diagram >>= diagram.cod[:j] @ diagram.swap(
                        diagram.cod[j], diagram.cod[j + 1:offset + i]
                    ) @ diagram.cod[offset + i:]
                    scan = (scan[:j] + scan[j + 1:offset + i]) + (
                        scan[j:j + 1] + scan[offset + i:])
                    offset -= 1

            offset = 0 if offset is None else offset
            scan = scan[:offset] + cod_wires + scan[offset + len(box.dom):]
            diagram >>= diagram.cod[:offset] @ box @ diagram.cod[
                offset + len(box.dom):]

        cod_wires = [
            edge_wire[self.n_ports - len(self.cod) + i]
            for i in range(len(self.cod))]
        for i, wire_id in enumerate(cod_wires):
            j = scan.index(wire_id)
            if i < j:
                diagram >>= diagram.cod[:i] @ diagram.swap(
                    diagram.cod[i:j], diagram.cod[j:j + 1]
                ) @ diagram.cod[j + 1:]
                scan = scan[:i] + scan[j:j + 1] + scan[i:j] + scan[j + 1:]
        return diagram

    def to_hypergraph(self):
        """
        Forget orientation and return the underlying bijective hypergraph
        given by the edge permutation. See documentation of
        :func:``Hypergraph.from_map`` for an example.
        """
        return hypergraph.Hypergraph[self.category].from_map(self)

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
                for key, value in attributes.items())

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
                f"{port.kind} {port.i}: {port.obj} "
                f"({port.side}, {port.direction})")
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

        lines = [
            "graph cmap {",
            f"  graph [{attr_string(attrs)}];",
            '  node [shape=plain, color=black, fontname="Helvetica", '
            'fontsize="12", margin="0"];',
            '  edge [color=black, penwidth="1.4", fontsize="9", '
            'headclip="true", tailclip="true"];',
        ]

        port_nodes = {}
        for vertex in range(len(self.boxes)):
            box = self.boxes[vertex]
            attributes = dict(label=Html(box_table(vertex, box)))
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
            attributes = dict(label=Html(boundary_table(ports)))
            lines.append(f"  {name} [{attr_string(attributes)}];")
            for port_index in ports:
                port_nodes[port_index] = f"{name}:p{port_index}:{compass}"

        for rank, name, ports in [
                ("min", "input", input_ports),
                ("max", "output", output_ports)]:
            if ports:
                lines.append(f"  {{ rank={rank}; {name}; }}")

        for i, loop in enumerate(self.loops):
            attributes = dict(
                label="",
                width="0.08",
                height="0.08",
                shape="point",
                tooltip=f"loop {i}: {loop}")
            lines.append(f"  loop{i} [{attr_string(attributes)}];")
            attributes = dict(len="0.85", label=loop)
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
                return dict(label=left_label)
            return dict(taillabel=left_label, headlabel=right_label)

        for i, j in enumerate(self.edges):
            if i < j:
                attributes = dict(
                    len="0.85", labeldistance="1.6") | edge_labels(i, j)
                lines.append(
                    f'  {node_name(i)} -- {node_name(j)} '
                    f'[{attr_string(attributes)}];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def draw(
            self, path=None, engine="dot", format=None, seed=None,
            show=None, graph_attr=None, port_indices=False, block=True):
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
        ...     path="docs/_static/cmap/scalar-loop.svg", show=False)

        .. image:: /_static/cmap/scalar-loop.svg
            :align: center
        """
        dot = self.to_dot(
            engine=engine, seed=seed, graph_attr=graph_attr,
            port_indices=port_indices)

        show = show if show is not None else path is None
        if path is not None:
            suffix = "" if path is None else (
                path.rsplit(".", 1)[-1].lower() if "." in path else "")
            if suffix in ["dot", "gv"]:
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write(dot)
                return None

        executable = shutil.which(engine) or shutil.which("dot")
        if executable is None:
            raise RuntimeError(
                f"Graphviz executable {engine!r} was not found.")

        if path is not None:
            output_format = format or suffix or "svg"
            subprocess.run(
                [executable, f"-T{output_format}", "-o", path],
                input=dot.encode(), check=True)
        if not show:
            return None

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
        return None
