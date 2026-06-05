# -*- coding: utf-8 -*-

"""
Combinatorial maps with interfaces.

The ports of a map are ordered as in :mod:`discopy.hypergraph`: inputs,
then the domain and codomain ports of each box, then outputs. A map is given by
two permutations on these ports:

* ``edge`` is a fixpoint-free involution pairing left and right ports;
* ``node`` fixes interfaces and orients the ports of each box.
"""

from __future__ import annotations
from discopy.cat import Category, Functor

from collections.abc import Iterable
from math import cos, pi, sin
from typing import Any, TYPE_CHECKING, ClassVar

from discopy import messages, hypergraph
from discopy.drawing import Node
from discopy.utils import (
    AxiomError,
    Composable,
    NamedGeneric,
    Whiskerable,
    assert_isinstance,
    factory_name,
    unbiased,
)

if TYPE_CHECKING:
    from discopy.cat import Ty, Box


Port = Node
""" A port in a combinatorial map. """

Permutation = tuple[int, ...]
""" A permutation, represented by its action on ``range(len(perm))``. """

LEFT_PORTS = {"input", "cod"}
RIGHT_PORTS = {"dom", "output"}
BOUNDARY_PORTS = {"input", "output"}


def _assert_permutation(perm: Permutation, size: int):
    if len(perm) != size:
        raise ValueError
    if sorted(perm) != list(range(size)):
        raise ValueError


def cycles(perm: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """
    Return the cycles of a permutation.

    Examples
    --------
    >>> cycles((1, 0, 3, 2))
    ((0, 1), (2, 3))
    """
    perm = tuple(perm)
    _assert_permutation(perm, len(perm))
    result, seen = [], set()
    for i in range(len(perm)):
        if i in seen:
            continue
        cycle, j = [], i
        while j not in seen:
            seen.add(j)
            cycle.append(j)
            j = perm[j]
        result.append(tuple(cycle))
    return tuple(result)


def permutation_from_cycles(
        permutation_cycles: Iterable[Iterable[int]], size: int) -> Permutation:
    """
    Build a permutation from cycles.

    Examples
    --------
    >>> permutation_from_cycles([(0, 1), (2, 3)], 4)
    (1, 0, 3, 2)
    """
    result = list(range(size))
    seen = set()
    for cycle in map(tuple, permutation_cycles):
        if len(set(cycle)) != len(cycle):
            raise ValueError
        for i in cycle:
            if i < 0 or i >= size or i in seen:
                raise ValueError
            seen.add(i)
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            result[source] = target
    return tuple(result)


def is_fixpoint_free_involution(perm: Iterable[int]) -> bool:
    """ Whether a permutation is a product of disjoint 2-cycles. """
    perm = tuple(perm)
    try:
        _assert_permutation(perm, len(perm))
    except ValueError:
        return False
    return all(perm[i] != i and perm[perm[i]] == i for i in range(len(perm)))


def port_side(port: Port) -> str:
    """
    Return ``"left"`` or ``"right"`` for a port.

    Examples
    --------
    >>> port_side(Node("input", i=0, obj=None))
    'left'
    >>> port_side(Node("output", i=0, obj=None))
    'right'
    """
    if port.kind in LEFT_PORTS:
        return "left"
    if port.kind in RIGHT_PORTS:
        return "right"
    raise ValueError


def _same_type(left, right) -> bool:
    left_r, right_r = getattr(left, "r", left), getattr(right, "r", right)
    return right in [left, left_r] or left in [right, right_r]


class CombinatorialMap(
        Composable, Whiskerable, NamedGeneric["category", "functor"]):
    """
    A bijective oriented hypergraph with interfaces.

    Parameters:
        dom : The domain of the map.
        cod : The codomain of the map.
        boxes : The boxes inside the map.
        edge : A fixpoint-free involution on ports.
        node : A permutation fixing interfaces and cycling each box.
        offsets : Optional drawing offsets, preserved through conversion.
    """
    category = None
    functo = None

    def __init__(
            self, dom: Ty, cod: Ty, boxes: tuple[Box, ...],
            edge: Iterable[int], node: Iterable[int] | None = None,
            offsets: tuple[int | None, ...] | None = None):
        assert_isinstance(dom, self.category.ob)
        assert_isinstance(cod, self.category.ob)
        for box in boxes:
            assert_isinstance(box, self.category.ar)
        self.dom, self.cod, self.boxes = dom, cod, tuple(boxes)
        self.offsets = offsets or tuple(len(boxes) * [None])
        if len(self.offsets) != len(self.boxes):
            raise ValueError

        self.edge = tuple(edge)
        self.node = self.canonical_node() if node is None else tuple(node)
        self._validate()

    @property
    def ports(self) -> list[Port]:
        """ The ports in the map, in fixed Hypergraph order. """
        inputs = [Node("input", i=i, obj=obj)
                  for i, obj in enumerate(self.dom)]
        box_ports = sum([[
            Node(kind, depth=depth, i=i, obj=obj)
            for i, obj in enumerate(typ)]
            for depth, box in enumerate(self.boxes)
            for kind, typ in [("dom", box.dom), ("cod", box.cod)]], [])
        outputs = [Node("output", i=i, obj=obj)
                   for i, obj in enumerate(self.cod)]
        return inputs + box_ports + outputs

    @property
    def n_ports(self) -> int:
        """ The number of ports. """
        return len(self.ports)

    @property
    def box_port_indices(self) -> tuple[tuple[int, ...], ...]:
        """ Port indices for each box. """
        result, start = [], len(self.dom)
        for box in self.boxes:
            stop = start + len(box.dom @ box.cod)
            result.append(tuple(range(start, stop)))
            start = stop
        return tuple(result)

    @property
    def node_cycles(self) -> tuple[tuple[int, ...], ...]:
        """ The node cycles, with empty cycles for zero-arity boxes. """
        result = []
        for box_ports in self.box_port_indices:
            if not box_ports:
                result.append(())
                continue
            cycle, seen, i = [], set(), box_ports[0]
            while i not in seen:
                seen.add(i)
                cycle.append(i)
                i = self.node[i]
            result.append(tuple(cycle))
        return tuple(result)

    @property
    def face_permutation(self) -> Permutation:
        """ The face permutation ``node o edge``. """
        return tuple(self.node[self.edge[i]] for i in range(self.n_ports))

    @property
    def face_cycles(self) -> tuple[tuple[int, ...], ...]:
        """ The cycles of the face permutation. """
        return cycles(self.face_permutation)

    faces = face_cycles

    @property
    def euler_characteristic(self) -> int:
        """ Euler characteristic ``V - E + F``. """
        return len(self.boxes) - self.n_ports // 2 + len(self.face_cycles)

    def canonical_node(self) -> Permutation:
        """ The canonical box orientation in fixed local port order. """
        return permutation_from_cycles(self.box_port_indices, len(self.ports))

    def _validate(self):
        ports = self.ports
        _assert_permutation(self.edge, len(ports))
        _assert_permutation(self.node, len(ports))
        if not is_fixpoint_free_involution(self.edge):
            raise ValueError

        for i, j in enumerate(self.edge):
            if port_side(ports[i]) == port_side(ports[j]):
                raise AxiomError
            if not _same_type(ports[i].obj, ports[j].obj):
                raise AxiomError(messages.TYPE_ERROR.format(
                    ports[i].obj, ports[j].obj))

        for i, port in enumerate(ports):
            if port.kind in BOUNDARY_PORTS and self.node[i] != i:
                raise ValueError

        for box_ports in self.box_port_indices:
            if not box_ports:
                continue
            if {self.node[i] for i in box_ports} != set(box_ports):
                raise ValueError
            if len(cycles(tuple(box_ports.index(self.node[i])
                                for i in box_ports))) != 1:
                raise ValueError

    def __repr__(self):
        return factory_name(type(self))\
            + f"(dom={repr(self.dom)}, cod={repr(self.cod)}, " \
              f"boxes={repr(self.boxes)}, edge={repr(self.edge)}, " \
              f"node={repr(self.node)})"

    def __eq__(self, other: Any):
        return isinstance(other, CombinatorialMap) and (
            self.dom, self.cod, self.boxes, self.edge, self.node
        ) == (other.dom, other.cod, other.boxes, other.edge, other.node)

    def __hash__(self):
        return hash((self.dom, self.cod, self.boxes, self.edge, self.node))

    @classmethod
    def id(cls, dom=None) -> CombinatorialMap:
        dom = cls.category.ob() if dom is None else dom
        n_ports = 2 * len(dom)
        edge = list(range(n_ports))
        for i in range(len(dom)):
            edge[i] = i + len(dom)
            edge[i + len(dom)] = i
        return cls(dom, dom, (), edge)

    @classmethod
    def from_box(cls, box: Box) -> CombinatorialMap:
        left = len(box.dom)
        right = len(box.cod)
        n_ports = 2 * (left + right)
        edge = list(range(n_ports))
        for i in range(left):
            edge[i] = left + i
            edge[left + i] = i
        for i in range(right):
            source = left + left + i
            target = left + left + right + i
            edge[source] = target
            edge[target] = source
        return cls(box.dom, box.cod, (box, ), edge)

    @classmethod
    def from_hypergraph(cls, old: hypergraph.Hypergraph) -> CombinatorialMap:
        """ Build a combinatorial map from a bijective hypergraph. """
        if not old.is_bijective:
            raise ValueError
        factory = cls if cls.category is not None else cls[
            type(old).category, type(old).functor]
        return factory(
            old.dom, old.cod, old.boxes, old.bijection,
            offsets=old.offsets)

    @unbiased
    def then(self, other: CombinatorialMap) -> CombinatorialMap:
        """ Sequential composition, gluing output ports to input ports. """
        if not self.cod == other.dom:
            raise AxiomError
        dom, cod = self.dom, other.cod
        boxes = self.boxes + other.boxes
        offsets = self.offsets + other.offsets

        self_outputs = range(self.n_ports - len(self.cod), self.n_ports)
        other_inputs = range(len(other.dom))
        remove_self = set(self_outputs)
        remove_other = set(other_inputs)

        self_map, other_map, new_index = {}, {}, 0
        for i in range(self.n_ports):
            if i not in remove_self:
                self_map[i] = new_index
                new_index += 1
        for i in range(other.n_ports):
            if i not in remove_other:
                other_map[i] = new_index
                new_index += 1

        edge = list(range(new_index))

        def add_pair(left, right):
            edge[left], edge[right] = right, left

        for i in range(self.n_ports):
            j = self.edge[i]
            if i < j and i not in remove_self and j not in remove_self:
                add_pair(self_map[i], self_map[j])
        for i in range(other.n_ports):
            j = other.edge[i]
            if i < j and i not in remove_other and j not in remove_other:
                add_pair(other_map[i], other_map[j])

        for left, right in zip(self_outputs, other_inputs):
            left_partner = self.edge[left]
            right_partner = other.edge[right]
            add_pair(self_map[left_partner], other_map[right_partner])

        node = list(range(new_index))
        for i, j in enumerate(self.node):
            if i in self_map and j in self_map:
                node[self_map[i]] = self_map[j]
        for i, j in enumerate(other.node):
            if i in other_map and j in other_map:
                node[other_map[i]] = other_map[j]

        return type(self)(dom, cod, boxes, edge, node, offsets)

    @unbiased
    def tensor(self, other: CombinatorialMap) -> CombinatorialMap:
        """ Tensor product, given by disjoint union of permutations. """
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        boxes = self.boxes + other.boxes
        offsets = self.offsets + other.offsets

        self_dom, self_cod = len(self.dom), len(self.cod)
        other_dom, other_cod = len(other.dom), len(other.cod)
        self_box_ports = self.n_ports - self_dom - self_cod
        other_box_ports = other.n_ports - other_dom - other_cod
        self_map, other_map = {}, {}

        for i in range(self_dom):
            self_map[i] = i
        for i in range(other_dom):
            other_map[i] = self_dom + i
        for i in range(self_box_ports):
            self_map[self_dom + i] = self_dom + other_dom + i
        for i in range(other_box_ports):
            other_map[other_dom + i] = (
                self_dom + other_dom + self_box_ports + i)
        cod_start = self_dom + other_dom + self_box_ports + other_box_ports
        for i in range(self_cod):
            self_map[self.n_ports - self_cod + i] = cod_start + i
        for i in range(other_cod):
            other_map[other.n_ports - other_cod + i] = (
                cod_start + self_cod + i)

        n_ports = self.n_ports + other.n_ports
        edge, node = list(range(n_ports)), list(range(n_ports))
        for mapping, old_edge, old_node in [
                (self_map, self.edge, self.node),
                (other_map, other.edge, other.node)]:
            for i, j in enumerate(old_edge):
                edge[mapping[i]] = mapping[j]
            for i, j in enumerate(old_node):
                node[mapping[i]] = mapping[j]
        return type(self)(dom, cod, boxes, edge, node, offsets)

    def to_hypergraph(self) -> hypergraph.Hypergraph:
        """ Forget orientation and return the underlying bijective hypergraph. """
        spider_types, flat_wires = [], [None] * self.n_ports
        for i in range(self.n_ports):
            j = self.edge[i]
            if i > j:
                continue
            spider = len(spider_types)
            spider_types.append(self.ports[i].obj)
            flat_wires[i] = flat_wires[j] = spider
        wires = hypergraph.Hypergraph.rebracket(
            None, flat_wires, dom=self.dom, boxes=self.boxes)
        factory = hypergraph.Hypergraph[self.category, self.functor]
        return factory(
            self.dom, self.cod, self.boxes, wires,
            tuple(spider_types), self.offsets)

    def draw_map(
            self, seed=None, path=None, ax=None, show=True,
            node_size=.09, boundary_size=.05, figsize=None):
        """
        Draw as a combinatorial map, with vertices as black dots.

        This is intended for map-like pictures, closer to the rooted trivalent
        maps in Zeilberger's linear-lambda-terms paper than to the usual
        DisCoPy box-and-wire drawing.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.patches import FancyArrowPatch

        graph, port_vertices = nx.MultiGraph(), {}
        for vertex in range(len(self.boxes)):
            graph.add_node(("box", vertex))
        for port_index, port in enumerate(self.ports):
            if port.kind in BOUNDARY_PORTS:
                node = (port.kind, port.i)
            else:
                node = ("box", port.depth)
            port_vertices[port_index] = node
            graph.add_node(node)
        for i, j in enumerate(self.edge):
            if i < j:
                graph.add_edge(port_vertices[i], port_vertices[j])

        if len(graph) == 0:
            pos = {}
        elif len(graph) == 1:
            node, = graph.nodes
            pos = {node: (0, 0)}
        else:
            pos = nx.spring_layout(graph, seed=seed)
        for i in range(len(self.dom)):
            pos[("input", i)] = (i - (len(self.dom) - 1) / 2, 1)
        for i in range(len(self.cod)):
            pos[("output", i)] = (i - (len(self.cod) - 1) / 2, -1)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (4, 4))
        else:
            fig = ax.figure
        ax.set_aspect("equal")
        ax.axis("off")

        port_pos = {}
        for vertex, cycle in enumerate(self.node_cycles):
            x, y = pos[("box", vertex)]
            for k, port_index in enumerate(cycle):
                angle = 2 * pi * k / len(cycle) + pi / 2 if cycle else 0
                port_pos[port_index] = (
                    x + node_size * cos(angle),
                    y + node_size * sin(angle))
        for port_index, port in enumerate(self.ports):
            if port.kind in BOUNDARY_PORTS:
                port_pos[port_index] = pos[port_vertices[port_index]]

        def draw_edge(source_pos, target_pos, rad=0):
            arrow = FancyArrowPatch(
                source_pos, target_pos, arrowstyle="-",
                connectionstyle=f"arc3,rad={rad}",
                color="black", linewidth=1.4, mutation_scale=1)
            ax.add_patch(arrow)

        pair_counts = {}
        for i, j in enumerate(self.edge):
            if i >= j:
                continue
            source, target = port_vertices[i], port_vertices[j]
            key = tuple(sorted((source, target)))
            count = pair_counts.get(key, 0)
            pair_counts[key] = count + 1
            rad = .35 if source == target else .15 * ((count + 1) // 2)
            if count % 2:
                rad *= -1
            draw_edge(port_pos[i], port_pos[j], rad)

        for vertex in range(len(self.boxes)):
            x, y = pos[("box", vertex)]
            circle = plt.Circle((x, y), node_size, color="black", zorder=3)
            ax.add_patch(circle)
            cycle = self.node_cycles[vertex]
            for port_index in cycle:
                port_x, port_y = port_pos[port_index]
                ax.plot(
                    [x, port_x], [y, port_y],
                    color="black", linewidth=1.4, zorder=4)

        for node in graph.nodes:
            if node[0] == "box":
                continue
            x, y = pos[node]
            ax.add_patch(plt.Circle(
                (x, y), boundary_size, color="black", zorder=3))

        xs = [x for x, _ in pos.values()] or [0]
        ys = [y for _, y in pos.values()] or [0]
        pad = .3
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        if path is not None:
            fig.savefig(path, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

    draw_as_map = draw_map
