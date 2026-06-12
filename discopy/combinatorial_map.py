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
from discopy.abc import MonoidalCategory, NamedGeneric

from collections.abc import Iterable
import shutil
import subprocess
from typing import Any, TYPE_CHECKING

from discopy import messages, hypergraph
from discopy.drawing import Node
from discopy.utils import (
    AxiomError,
    assert_isinstance,
    classproperty,
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


class CombinatorialMap(MonoidalCategory, NamedGeneric['functor']):
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
    functor = None
    category = classproperty(lambda cls: cls.functor.dom)
    ty_factory = classproperty(lambda cls: cls.category.ty_factory)

    def __init__(
            self, dom: Ty, cod: Ty, boxes: tuple[Box, ...],
            edge: Iterable[int], node: Iterable[int] | None = None,
            offsets: tuple[int | None, ...] | None = None):
        assert_isinstance(dom, self.category.ty_factory)
        assert_isinstance(cod, self.category.ty_factory)
        for box in boxes:
            assert_isinstance(box, self.category)
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
        dom = cls.category.ty_factory() if dom is None else dom
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

    def plug_input(
            self, input_index: int, box: Box,
            cod: Ty) -> CombinatorialMap:
        """
        Plug an input boundary and the output root into a new box.

        If ``self : A @ x -> y`` and ``box : y -> z @ x``, then
        ``self.plug_input(i, box, z)`` removes the ``i``-th input, wires the
        old output to the domain of ``box``, wires the removed input to the
        second output of ``box``, and leaves the first output of ``box`` as the
        new root.
        """
        assert_isinstance(box, self.category)
        if len(self.cod) != 1 or len(box.dom) != 1 or len(box.cod) != 2:
            raise ValueError
        if input_index < 0 or input_index >= len(self.dom):
            raise ValueError

        old_input, old_output = input_index, self.n_ports - 1
        new_dom = self.category.ty_factory()
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
        box_root = new_index + 1
        box_parameter = new_index + 2
        new_output = new_index + 3

        edge_pairs = []
        for i, j in enumerate(self.edge):
            if i < j and i not in [old_input, old_output]\
                    and j not in [old_input, old_output]:
                edge_pairs.append((mapping[i], mapping[j]))

        input_partner = self.edge[old_input]
        output_partner = self.edge[old_output]
        if input_partner == old_output:
            edge_pairs.append((box_parameter, box_dom))
        else:
            edge_pairs.append((mapping[input_partner], box_parameter))
            edge_pairs.append((mapping[output_partner], box_dom))
        edge_pairs.append((box_root, new_output))
        edge = Permutation.from_transpositions(edge_pairs, new_output + 1)

        node = Permutation.from_cycles(
            [tuple(mapping[i] for i in cycle) for cycle in self.node_cycles]
            + [(box_dom, box_root, box_parameter)],
            new_output + 1)
        return type(self)(new_dom, cod, boxes, edge, node, offsets)

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
        factory = hypergraph.Hypergraph[self.functor]
        return factory(
            self.dom, self.cod, self.boxes, wires,
            tuple(spider_types), self.offsets)

    def to_term(self, input_names: Iterable[str] | None = None):
        """
        Recover the linear lambda term represented by a rooted trivalent map.

        This implements the inverse transformation from Zeilberger's
        correspondence. The map is expected to have its ordered free-variable
        context on ``dom`` and a singleton root on ``cod``.
        """
        from discopy.closed import (
            Abstraction,
            Application,
            Exp,
            Variable,
            assert_term_map,
        )

        self._assert_rooted_trivalent_map()
        names = tuple(input_names or (f"x{i}" for i in range(len(self.dom))))
        if len(names) != len(self.dom):
            raise ValueError

        def term_type(obj):
            if isinstance(obj, self.category.ty_factory)\
                    and len(obj) == 1\
                    and isinstance(obj.inside[0], Exp):
                return obj.inside[0]
            return obj

        variables = tuple(Variable(obj, name)
                          for obj, name in zip(map(term_type, self.dom), names))
        counter = [len(variables)]

        def fresh(obj):
            variable = Variable(obj, f"x{counter[0]}")
            counter[0] += 1
            return variable

        def same_map_shape(left, right):
            return (left.dom, left.cod, left.edge, left.node) == (
                right.dom, right.cod, right.edge, right.node)

        def go(cmap, context):
            cmap._assert_rooted_trivalent_map()
            if not cmap.boxes:
                if len(cmap.dom) != 1 or len(cmap.cod) != 1:
                    raise ValueError
                if cmap.edge != (1, 0):
                    raise ValueError
                return context[0]

            root_output = cmap.n_ports - 1
            root_port = cmap.edge[root_output]
            root_box = cmap.ports[root_port].depth
            root_cycle = set(cmap.node_cycles[root_box])
            removed = root_cycle | {root_output}
            components = cmap._components(removed)

            ports = cmap.ports
            attachments = []
            for port in cmap.node_cycles[root_box]:
                partner = cmap.edge[port]
                if partner not in removed:
                    attachments.append((port, partner))
            component_of = {
                port: index
                for index, component in enumerate(components)
                for port in component}
            attached_components = {
                component_of[partner] for _, partner in attachments}

            if len(attached_components) == 2:
                subterms = []
                for component_index in attached_components:
                    component = components[component_index]
                    root_partner, = [
                        partner for _, partner in attachments
                        if partner in component]
                    submap = cmap._component_as_map(component, root_partner)
                    sub_context = tuple(
                        context[ports[i].i]
                        for i in sorted(component)
                        if ports[i].kind == "input")
                    subterms.append(go(submap, sub_context))

                for func, arg in [subterms, tuple(reversed(subterms))]:
                    try:
                        term = Application(func, arg)
                    except (TypeError, ValueError):
                        continue
                    if same_map_shape(term.to_map(type(cmap)), cmap):
                        return term
                raise ValueError

            if len(attached_components) > 1:
                raise ValueError

            cod = term_type(cmap.cod[0])
            if len(cmap.cod) != 1 or not isinstance(cod, Exp):
                raise ValueError
            variable = fresh(cod.exponent)

            root_box_ports = cmap.box_port_indices[root_box]
            body_ports = [i for i in root_box_ports
                          if ports[i].kind == "dom"]
            parameter_ports = [
                i for i in root_box_ports
                if ports[i].kind == "cod" and i != root_port]
            if len(body_ports) != 1 or len(parameter_ports) != 1:
                raise ValueError
            body_port, parameter_port = body_ports[0], parameter_ports[0]
            body_partner = cmap.edge[body_port]
            parameter_partner = cmap.edge[parameter_port]

            candidates = []
            if body_partner == parameter_port\
                    and parameter_partner == body_port:
                if len(cmap.dom) != 0:
                    raise ValueError
                candidates.append((type(cmap).id(variable.cod), (variable, )))
            else:
                if body_partner in removed or parameter_partner in removed:
                    raise ValueError
                if component_of[body_partner] != component_of[
                        parameter_partner]:
                    raise ValueError
                component = components[component_of[body_partner]]
                old_context = tuple(
                    context[ports[i].i]
                    for i in sorted(component)
                    if ports[i].kind == "input")
                for input_index in range(len(old_context) + 1):
                    body_context = (
                        old_context[:input_index]
                        + (variable, )
                        + old_context[input_index:])
                    candidates.append((
                        cmap._component_as_map(
                            component, body_partner,
                            extra_input_root=parameter_partner,
                            extra_input_obj=variable.cod,
                            extra_input_index=input_index),
                        body_context))

            for body_map, body_context in candidates:
                try:
                    term = Abstraction(variable, go(body_map, body_context))
                except (TypeError, ValueError):
                    continue
                term_map = term.to_map(type(cmap))
                if same_map_shape(term_map, cmap):
                    assert_term_map(term_map, term, type(cmap))
                    return term
            raise ValueError

        return go(self, variables)

    def _assert_rooted_trivalent_map(self):
        if len(self.cod) != 1:
            raise ValueError
        if self.n_ports == 0 or self.ports[-1].kind != "output":
            raise ValueError
        if self.node[-1] != self.n_ports - 1:
            raise ValueError
        if self.edge[-1] == self.n_ports - 1:
            raise ValueError
        if any(len(cycle) != 3 for cycle in self.node_cycles):
            raise ValueError

    def _components(self, removed: set[int]) -> tuple[set[int], ...]:
        """ Connected components of the port graph after deleting ports. """
        kept = set(range(self.n_ports)) - removed
        adjacency = {port: set() for port in kept}
        for port in tuple(kept):
            other = self.edge[port]
            if other in kept:
                adjacency[port].add(other)
                adjacency[other].add(port)
        for cycle in self.node_cycles:
            for left, right in zip(cycle, cycle[1:] + cycle[:1]):
                if left in kept and right in kept:
                    adjacency[left].add(right)
                    adjacency[right].add(left)

        components, unseen = [], set(kept)
        while unseen:
            start = unseen.pop()
            component, stack = {start}, [start]
            while stack:
                port = stack.pop()
                for other in adjacency[port]:
                    if other in unseen:
                        unseen.remove(other)
                        component.add(other)
                        stack.append(other)
            components.append(component)
        return tuple(components)

    def _component_as_map(
            self, component: set[int], output_root: int,
            extra_input_root: int | None = None,
            extra_input_obj: Any = None,
            extra_input_index: int | None = None) -> CombinatorialMap:
        """ Re-root a connected component as a smaller map. """
        ports = self.ports

        def port_type(obj):
            return obj if isinstance(obj, self.category.ty_factory)\
                else self.category.ty_factory(obj)

        input_ports = [
            port for port in sorted(component)
            if ports[port].kind == "input"]
        if extra_input_root is None:
            extra_input_index = None
        elif extra_input_index is None\
                or extra_input_index < 0\
                or extra_input_index > len(input_ports):
            raise ValueError

        dom = self.category.ty_factory()
        mapping, new_index = {}, 0
        input_iter = iter(input_ports)
        for i in range(len(input_ports) + (extra_input_root is not None)):
            if i == extra_input_index:
                dom = dom @ extra_input_obj
                new_input = new_index
                new_index += 1
            else:
                port = next(input_iter)
                mapping[port] = new_index
                dom = dom @ port_type(ports[port].obj)
                new_index += 1

        included_boxes, offsets = [], []
        for depth, box_ports in enumerate(self.box_port_indices):
            if set(box_ports) <= component:
                included_boxes.append(self.boxes[depth])
                offsets.append(self.offsets[depth])
                for port in box_ports:
                    mapping[port] = new_index
                    new_index += 1

        cod = port_type(ports[output_root].obj)
        output = new_index
        size = output + 1
        edge_pairs = []
        for port in sorted(component):
            other = self.edge[port]
            if port < other and other in component:
                edge_pairs.append((mapping[port], mapping[other]))
        edge_pairs.append((mapping[output_root], output))
        if extra_input_root is not None:
            edge_pairs.append((new_input, mapping[extra_input_root]))

        edge = Permutation.from_transpositions(edge_pairs, size)
        node = Permutation.from_cycles(
            [tuple(mapping[port] for port in self.node_cycles[depth])
             for depth, box_ports in enumerate(self.box_port_indices)
             if set(box_ports) <= component],
            size)
        return type(self)(
            dom, cod, tuple(included_boxes), edge, node, tuple(offsets))

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
