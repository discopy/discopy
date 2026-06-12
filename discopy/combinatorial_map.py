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

from collections.abc import Iterable
import shutil
import subprocess
from typing import Any, TYPE_CHECKING

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

LEFT_PORTS = {"input", "cod"}
RIGHT_PORTS = {"dom", "output"}
BOUNDARY_PORTS = {"input", "output"}


class Permutation(tuple):
    """
    A permutation, represented by its action on ``range(len(self))``.

    Examples
    --------
    >>> Permutation((1, 0, 3, 2)).cycles()
    ((0, 1), (2, 3))
    >>> Permutation.from_cycles([(0, 1), (2, 3)], 4)
    (1, 0, 3, 2)
    >>> Permutation((1, 0)).is_fixpoint_free_involution()
    True
    """
    def __new__(cls, inside: Iterable[int] = (), size: int | None = None):
        inside = tuple(inside)
        if size is None:
            size = len(inside)
        if len(inside) != size:
            raise ValueError
        if sorted(inside) != list(range(size)):
            raise ValueError
        return tuple.__new__(cls, inside)

    @classmethod
    def identity(cls, size: int) -> Permutation:
        """ The identity permutation on ``range(size)``. """
        return cls(range(size), size)

    @classmethod
    def from_cycles(
            cls, permutation_cycles: Iterable[Iterable[int]],
            size: int) -> Permutation:
        """ Build a permutation from cycles. """
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
        return cls(result, size)

    @classmethod
    def from_transpositions(
            cls, transpositions: Iterable[tuple[int, int]],
            size: int) -> Permutation:
        """ Build a permutation from disjoint 2-cycles. """
        result = list(range(size))
        seen = set()
        for left, right in transpositions:
            if left == right:
                raise ValueError
            if left < 0 or right < 0 or left >= size or right >= size:
                raise ValueError
            if left in seen or right in seen:
                raise ValueError
            seen.update([left, right])
            result[left], result[right] = right, left
        return cls(result, size)

    @classmethod
    def from_relabels(
            cls, relabelings: Iterable[tuple[Iterable[int], dict[int, int]]],
            size: int) -> Permutation:
        """ Relabel and merge permutations with disjoint target indices. """
        result = list(range(size))
        for old, mapping in relabelings:
            old = cls(old)
            for i, j in enumerate(old):
                if i in mapping and j in mapping:
                    result[mapping[i]] = mapping[j]
        return cls(result, size)

    def cycles(self) -> tuple[tuple[int, ...], ...]:
        """ Return the cycles of the permutation. """
        result, seen = [], set()
        for i in range(len(self)):
            if i in seen:
                continue
            result.append(self.cycle(i, seen))
        return tuple(result)

    def cycle(
            self, start: int,
            seen: set[int] | None = None) -> tuple[int, ...]:
        """ Return the cycle reached from ``start``. """
        if start < 0 or start >= len(self):
            raise ValueError
        cycle, local_seen, i = [], set() if seen is None else seen, start
        while i not in local_seen:
            local_seen.add(i)
            cycle.append(i)
            i = self[i]
        return tuple(cycle)

    def compose(self, other: Iterable[int]) -> Permutation:
        """ Return ``self o other``, i.e. ``result[i] == self[other[i]]``. """
        other = type(self)(other, len(self))
        return type(self)((self[other[i]] for i in range(len(self))), len(self))

    def inverse(self) -> Permutation:
        """ Return the inverse permutation. """
        result = list(range(len(self)))
        for source, target in enumerate(self):
            result[target] = source
        return type(self)(result, len(self))

    def tensor(self, other: Iterable[int]) -> Permutation:
        """ Return the disjoint union of two permutations. """
        other = type(self)(other)
        shift = len(self)
        return type(self)(
            tuple(self) + tuple(shift + i for i in other),
            len(self) + len(other))

    def relabel(self, mapping: dict[int, int], size: int) -> Permutation:
        """ Relabel preserved indices, fixing everything else. """
        return type(self).from_relabels([(self, mapping)], size)

    def is_fixpoint_free_involution(self) -> bool:
        """ Whether this is a product of disjoint 2-cycles. """
        return all(self[i] != i and self[self[i]] == i
                   for i in range(len(self)))


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
    functor = None

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

        self.edge = Permutation(edge, len(self.ports))
        self.node = self.canonical_node() if node is None\
            else Permutation(node, len(self.ports))
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
            result.append(self.node.cycle(box_ports[0]))
        return tuple(result)

    @property
    def face_permutation(self) -> Permutation:
        """ The face permutation ``node o edge``. """
        return self.node.compose(self.edge)

    @property
    def face_cycles(self) -> tuple[tuple[int, ...], ...]:
        """ The cycles of the face permutation. """
        return self.face_permutation.cycles()

    faces = face_cycles

    @property
    def euler_characteristic(self) -> int:
        """ Euler characteristic ``V - E + F``. """
        return len(self.boxes) - self.n_ports // 2 + len(self.face_cycles)

    def canonical_node(self) -> Permutation:
        """ The canonical box orientation in fixed local port order. """
        return Permutation.from_cycles(self.box_port_indices, len(self.ports))

    def _validate(self):
        ports = self.ports
        if not self.edge.is_fixpoint_free_involution():
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
            if len(Permutation(tuple(box_ports.index(self.node[i])
                                     for i in box_ports)).cycles()) != 1:
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
        edge = Permutation.from_transpositions(
            ((i, i + len(dom)) for i in range(len(dom))), n_ports)
        return cls(dom, dom, (), edge)

    @classmethod
    def from_box(cls, box: Box) -> CombinatorialMap:
        left = len(box.dom)
        right = len(box.cod)
        n_ports = 2 * (left + right)
        edge = Permutation.from_transpositions(
            [(i, left + i) for i in range(left)]
            + [(left + left + i, left + left + right + i)
               for i in range(right)],
            n_ports)
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

        edge_pairs = []
        for i in range(self.n_ports):
            j = self.edge[i]
            if i < j and i not in remove_self and j not in remove_self:
                edge_pairs.append((self_map[i], self_map[j]))
        for i in range(other.n_ports):
            j = other.edge[i]
            if i < j and i not in remove_other and j not in remove_other:
                edge_pairs.append((other_map[i], other_map[j]))

        for left, right in zip(self_outputs, other_inputs):
            left_partner = self.edge[left]
            right_partner = other.edge[right]
            edge_pairs.append((self_map[left_partner], other_map[right_partner]))

        edge = Permutation.from_transpositions(edge_pairs, new_index)
        node = Permutation.from_relabels(
            [(self.node, self_map), (other.node, other_map)], new_index)

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
        edge = Permutation.from_relabels(
            [(self.edge, self_map), (other.edge, other_map)], n_ports)
        node = Permutation.from_relabels(
            [(self.node, self_map), (other.node, other_map)], n_ports)
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

    def to_dot(self, engine="neato", seed=None, graph_attr=None) -> str:
        """ Encode the combinatorial map as Graphviz DOT. """
        attrs = {
            "layout": engine,
            "overlap": "false",
            "splines": "true",
            "outputorder": "edgesfirst",
            "bgcolor": "transparent",
            "margin": "0.04",
        } | (graph_attr or {})
        if seed is not None:
            attrs["start"] = str(seed)

        def attr_string(attributes):
            return ", ".join(
                f'{key}="{value}"' for key, value in attributes.items())

        lines = [
            "graph combinatorial_map {",
            f"  graph [{attr_string(attrs)}];",
            '  node [shape=circle, color=black, fixedsize=true, '
            'fontname="Helvetica", fontsize="12"];',
            '  edge [color=black, penwidth="1.4", fontsize="9"];',
        ]

        port_nodes = {}
        for vertex in range(len(self.boxes)):
            box = self.boxes[vertex]
            label = "@" if (len(box.dom), len(box.cod)) == (2, 1)\
                else "ꟛ" if (len(box.dom), len(box.cod)) == (1, 2)\
                else getattr(box, "name", "")
            lines.append(
                f'  v{vertex} [label="{label}", width="0.22", '
                f'height="0.22"];')
            for port_index in self.node_cycles[vertex]:
                port_nodes[port_index] = f"v{vertex}"
        for port_index, port in enumerate(self.ports):
            if port.kind in BOUNDARY_PORTS:
                lines.append(
                    f'  b{port_index} [label="", style=filled, '
                    f'fillcolor=black, width="0.06", height="0.06", '
                    f'xlabel="{port.kind} {port.i}"];')
                port_nodes[port_index] = f"b{port_index}"

        def node_name(port_index):
            return port_nodes[port_index]

        for i, j in enumerate(self.edge):
            if i < j:
                lines.append(
                    f'  {node_name(i)} -- {node_name(j)} '
                    f'[len="0.85", taillabel="{i}", headlabel="{j}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def draw_map(
            self, path=None, engine="neato", format=None, seed=None,
            show=False, graph_attr=None):
        """
        Draw as a combinatorial map using Graphviz.

        This is intended for map-like pictures, closer to the rooted trivalent
        maps in Zeilberger's linear-lambda-terms paper than to the usual
        DisCoPy box-and-wire drawing.

        If ``path`` is ``None``, return the DOT source. If ``path`` ends in
        ``.dot`` or ``.gv``, write DOT source. Otherwise, render with Graphviz.
        """
        dot = self.to_dot(engine=engine, seed=seed, graph_attr=graph_attr)
        if path is None:
            return dot

        path = str(path)
        suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if suffix in ["dot", "gv"]:
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(dot)
            return path

        executable = shutil.which(engine) or shutil.which("dot")
        if executable is None:
            raise RuntimeError(
                f"Graphviz executable {engine!r} was not found.")

        output_format = format or suffix or "svg"
        subprocess.run(
            [executable, f"-T{output_format}", "-o", path],
            input=dot, text=True, check=True)
        if show:
            try:
                from IPython.display import Image, SVG, display
                display(SVG(filename=path) if output_format == "svg"
                        else Image(filename=path))
            except ImportError:
                pass
        return path

    draw_as_map = draw_map
