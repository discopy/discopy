# -*- coding: utf-8 -*-

"""
Combinatorial maps with interfaces.

The ports of a map are ordered as in :mod:`discopy.hypergraph`: inputs,
then the domain and codomain ports of each box, then outputs. A map is given by
two permutations on these ports:

* ``edge`` is a fixpoint-free involution pairing left and right ports;
* ``node`` is derived from the canonical counter-clockwise port order of boxes.
"""

from __future__ import annotations
from discopy.abc import MonoidalCategory, NamedGeneric, Monoid

from collections.abc import Iterable
from io import BytesIO
import shutil
import subprocess
from typing import Any, TYPE_CHECKING, ClassVar

from discopy import messages, hypergraph
from discopy.drawing import Node
from discopy.python.finset import Permutation
from discopy.utils import (
    AxiomError,
    assert_isinstance,
    classproperty,
    factory_name,
    unbiased,
)

if TYPE_CHECKING:
    from discopy.monoidal import Ty, Box, Diagram, Functor


Port = Node
""" A port in a combinatorial map. """

NEGATIVE_PORTS = {"input", "cod"}
POSITIVE_PORTS = {"dom", "output"}
BOUNDARY_PORTS = {"input", "output"}
IN_PORTS = {"input", "dom"}
OUT_PORTS = {"cod", "output"}


def port_side(port: Port) -> str:
    """
    Return ``"up"`` or ``"down"`` for a port.

    Examples
    --------
    >>> port_side(Node("input", i=0, obj=None))
    'up'
    >>> port_side(Node("output", i=0, obj=None))
    'down'
    """
    is_adjoint = bool(getattr(port.obj, "z", 0) % 2)
    if port.kind in NEGATIVE_PORTS:
        return "down" if is_adjoint else "up"
    if port.kind in POSITIVE_PORTS:
        return "up" if is_adjoint else "down"
    raise ValueError


def port_direction(port: Port) -> str:
    """ Return ``"in"`` or ``"out"`` for a port. """
    is_adjoint = bool(getattr(port.obj, "z", 0) % 2)
    if port.kind in IN_PORTS:
        return "out" if is_adjoint else "in"
    if port.kind in OUT_PORTS:
        return "in" if is_adjoint else "out"
    raise ValueError


def _same_type(left, right) -> bool:
    left_r, right_r = getattr(left, "r", left), getattr(right, "r", right)
    return right in [left, left_r] or left in [right, right_r]


class CombinatorialMap[C0: Monoid, C1: CombinatorialMap](
    MonoidalCategory[C0, C1], NamedGeneric['functor']
):
    """
    A bijective oriented hypergraph with interfaces.

    Parameters:
        dom : The domain of the map.
        cod : The codomain of the map.
        boxes : The boxes inside the map.
        edge : A fixpoint-free involution on ports.
        offsets : Optional drawing offsets, preserved through conversion.
    """
    functor: ClassVar[Functor]
    category = classproperty(lambda cls: cls.functor.dom)
    ob = classproperty(lambda cls: cls.category.ob)

    def __init__(
            self, dom: C0, cod: C0, boxes: tuple[Box, ...],
            edge: Iterable[int],
            offsets: tuple[int | None, ...] | None = None):
        assert_isinstance(dom, self.category.ob)
        assert_isinstance(cod, self.category.ob)
        for box in boxes:
            assert_isinstance(box, self.category)
        self.dom, self.cod, self.boxes = dom, cod, tuple(boxes)
        self.offsets = offsets or tuple(len(boxes) * [None])
        if len(self.offsets) != len(self.boxes):
            raise ValueError

        self.edge = Permutation(edge, len(self.ports))
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

    @property
    def euler_characteristic(self) -> int:
        """ Euler characteristic ``V - E + F``. """
        return len(self.boxes) - self.n_ports // 2 + len(self.face_cycles)

    @property
    def node(self) -> Permutation:
        """
        The canonical box orientation.

        Box ports are already in their canonical local counter-clockwise
        order, so each box contributes its consecutive port interval.
        """
        cycles = list(self.box_port_indices)
        return Permutation.from_cycles(cycles, len(self.ports))

    def _validate(self):
        ports = self.ports
        if not self.edge.is_fixpoint_free_involution():
            raise ValueError

        for i, j in enumerate(self.edge):
            type(self).validate_wire(ports[i], ports[j])

    @classmethod
    def validate_wire(cls, source: Port, target: Port):
        """ Validate whether two ports can be connected by a wire. """
        if source.kind in NEGATIVE_PORTS and target.kind in NEGATIVE_PORTS\
                or source.kind in POSITIVE_PORTS and target.kind in POSITIVE_PORTS:
            raise AxiomError
        if source.obj != target.obj:
            raise AxiomError(messages.TYPE_ERROR.format(
                source.obj, target.obj))

    def __repr__(self):
        def port_repr(index, port):
            port_depth = getattr(port, "depth", None)
            depth = "" if port_depth is None else f"@{port_depth}"
            return (
                f"{port.kind}{depth}[{port.i}]:{port.obj}:"
                f"{port_side(port)}/{port_direction(port)}"
                f"->{self.edge[index]}")

        ports = tuple(
            port_repr(index, port)
            for index, port in enumerate(self.ports))
        return factory_name(type(self))\
            + f"(dom={repr(self.dom)}, cod={repr(self.cod)}, " \
              f"boxes={repr(self.boxes)}, edge={repr(self.edge)}, " \
              f"ports={repr(ports)})"

    def __eq__(self, other: Any):
        return isinstance(other, CombinatorialMap) and (
            self.dom, self.cod, self.boxes, self.edge
        ) == (other.dom, other.cod, other.boxes, other.edge)

    def __hash__(self):
        return hash((self.dom, self.cod, self.boxes, self.edge))

    @classmethod
    def id(cls, dom=None) -> CombinatorialMap:
        dom = cls.ob() if dom is None else dom
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
    def from_diagram(cls, old: Diagram) -> CombinatorialMap:
        """
        Turn a :class:`Diagram` into a :class:`CombinatorialMap`.

        This follows the same architecture as :meth:`Hypergraph.from_diagram`:
        traverse the diagram with the hierarchy-specific functor and let the
        codomain map decide which categorical structure is represented as
        wiring and which structure is kept as boxes.
        """
        factory = cls if cls.functor is not None else cls[
            type(old), type(old).functor]
        return factory.functor(
            ob=lambda typ: typ, ar=factory.from_box,
            dom=type(old), cod=factory)(old)

    @classmethod
    def from_hypergraph(cls, old: hypergraph.Hypergraph) -> CombinatorialMap:
        """ Build a combinatorial map from a bijective hypergraph. """
        if not old.is_bijective:
            raise ValueError
        factory = cls if cls.functor is not None else cls[
            type(old).category, type(old).functor]
        return factory(
            old.dom, old.cod, old.boxes, old.bijection,
            offsets=old.offsets)

    @classmethod
    def braid(cls, left: Ty, right: Ty) -> CombinatorialMap:
        """ The braid, remembered as a box to preserve over/under data. """
        return cls.from_box(cls.category.braid(left, right))

    @classmethod
    def twist(cls, dom: Ty) -> CombinatorialMap:
        """ The twist, remembered as a box. """
        return cls.from_box(cls.category.twist(dom))

    @classmethod
    def swap(cls, left: Ty, right: Ty) -> CombinatorialMap:
        """ The symmetry, encoded as boundary wiring. """
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
    def cups(cls, left: Ty, right: Ty) -> CombinatorialMap:
        """ Cups, encoded as boundary wiring when types are adjoint. """
        if not getattr(left, "r", left[::-1]) == right:
            raise AxiomError
        size = len(left)
        edge = Permutation.from_transpositions(
            ((i, size + size - 1 - i) for i in range(size)),
            2 * size)
        return cls(left @ right, cls.ob(), (), edge)

    @classmethod
    def caps(cls, left: Ty, right: Ty) -> CombinatorialMap:
        """ Caps, encoded as boundary wiring when types are adjoint. """
        if not getattr(left, "r", left[::-1]) == right:
            raise AxiomError
        size = len(left)
        edge = Permutation.from_transpositions(
            ((i, size + size - 1 - i) for i in range(size)),
            2 * size)
        return cls(cls.ob(), left @ right, (), edge)

    @classmethod
    def copy(cls, typ: Ty, n: int = 2) -> CombinatorialMap:
        """ Copy is kept as a box: one input cannot wire to many outputs. """
        return cls.from_box(cls.category.copy(typ, n))

    @classmethod
    def merge(cls, typ: Ty, n: int = 2) -> CombinatorialMap:
        """ Merge is kept as a box: many inputs cannot wire to one output. """
        return cls.from_box(cls.category.merge(typ, n))

    @classmethod
    def discard(cls, typ: Ty) -> CombinatorialMap:
        """ Discard is kept as a box. """
        return cls.copy(typ, 0)

    @classmethod
    def spiders(
            cls, n_legs_in: int, n_legs_out: int,
            typ: Ty, phases=None) -> CombinatorialMap:
        """ Spiders are kept as boxes, including their phase data. """
        return cls.from_box(cls.category.spiders(
            n_legs_in, n_legs_out, typ, phases))

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

        shift = self.n_ports
        removed = remove_self | {shift + i for i in remove_other}
        kept = [i for i in range(self.n_ports + other.n_ports)
                if i not in removed]
        mapping = {old: new for new, old in enumerate(kept)}
        edge = dict(enumerate(self.edge))
        edge.update({
            shift + i: shift + j for i, j in enumerate(other.edge)})
        glue = dict(zip(self_outputs, (shift + i for i in other_inputs)))
        glue.update({j: i for i, j in glue.items()})

        def follow(port):
            port = edge[port]
            seen = set()
            while port in removed:
                if port in seen:
                    return None
                seen.add(port)
                port = edge[glue[port]]
            return port

        edge_pairs = []
        for i in kept:
            j = follow(i)
            if j is not None and i < j:
                edge_pairs.append((mapping[i], mapping[j]))

        edge = Permutation.from_transpositions(edge_pairs, len(kept))
        return type(self)(dom, cod, boxes, edge, offsets=offsets)

    def trace(self, n: int = 1, left: bool = False) -> CombinatorialMap:
        """ Partial trace, encoded by splicing traced boundary wires. """
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

        trace_pair = dict(zip(traced_inputs, traced_outputs))
        trace_pair.update(dict(zip(traced_outputs, traced_inputs)))
        removed = set(trace_pair)
        kept = [i for i in range(self.n_ports) if i not in removed]
        mapping = {old: new for new, old in enumerate(kept)}

        def follow(port):
            seen = set()
            while port in removed:
                if port in seen:
                    return None
                seen.add(port)
                port = self.edge[trace_pair[port]]
            return port

        edge_pairs = []
        for i in kept:
            j = follow(self.edge[i])
            if j is not None and i < j:
                edge_pairs.append((mapping[i], mapping[j]))

        edge = Permutation.from_transpositions(edge_pairs, len(kept))
        return type(self)(dom, cod, self.boxes, edge, offsets=self.offsets)

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
        edge_pairs = []
        for old_edge, mapping in [(self.edge, self_map),
                                  (other.edge, other_map)]:
            for i, j in enumerate(old_edge):
                if i < j:
                    edge_pairs.append((mapping[i], mapping[j]))
        edge = Permutation.from_transpositions(edge_pairs, n_ports)
        return type(self)(dom, cod, boxes, edge, offsets=offsets)

    def interchange(self, i: int, j: int) -> CombinatorialMap:
        """
        Interchange boxes at indices ``i`` and ``j``.

        The edge permutation is relabeled so that ports follow the canonical
        order induced by the new box order.
        """
        boxes, offsets = list(self.boxes), list(self.offsets)
        boxes[i], boxes[j] = boxes[j], boxes[i]
        offsets[i], offsets[j] = offsets[j], offsets[i]
        boxes, offsets = tuple(boxes), tuple(offsets)

        old_ports = self.box_port_indices
        start = len(self.dom)
        new_ports = {}
        for box_index, box in enumerate(boxes):
            stop = start + len(box.dom @ box.cod)
            old_index = j if box_index == i else i if box_index == j\
                else box_index
            new_ports[old_index] = tuple(range(start, stop))
            start = stop

        mapping = {i: i for i in range(self.n_ports)}
        for old_index, ports in enumerate(old_ports):
            for old, new in zip(ports, new_ports[old_index]):
                mapping[old] = new

        port_permutation = Permutation(
            (mapping[port] for port in range(self.n_ports)), self.n_ports)
        edge = self.edge.conjugate(port_permutation)
        return type(self)(self.dom, self.cod, boxes, edge, offsets=offsets)

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

        return type(self)(new_dom, cod, boxes, edge, offsets=offsets)

    def to_hypergraph(self) -> hypergraph.Hypergraph:
        """
        Forget orientation and return the underlying bijective hypergraph.
        """
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
        factory = getattr(
            self.category, "hypergraph_factory",
            hypergraph.Hypergraph[self.functor])
        return factory(
            self.dom, self.cod, self.boxes, wires,
            tuple(spider_types), self.offsets)

    def to_diagram(self) -> Diagram:
        """
        Downgrade to a diagram directly, preserving node orientation.

        The construction scans the currently open wire labels from left to
        right. For each box, it swaps boundary wires until the box domain wires
        are adjacent at the requested offset, applies the box, and replaces
        consumed domain labels by the box codomain labels.
        """
        edge_wire = {}
        for i, j in enumerate(self.edge):
            if i <= j:
                edge_wire[i] = edge_wire[j] = len(edge_wire) // 2

        diagram = self.category.id(self.dom)
        scan = [edge_wire[i] for i in range(len(self.dom))]

        for depth, (box, offset) in enumerate(zip(self.boxes, self.offsets)):
            box_ports = self.box_port_indices[depth]
            dom_ports = box_ports[:len(box.dom)]
            cod_ports = box_ports[len(box.dom):]
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

    def to_term(
            self, input_names: Iterable[str] | None = None):
        """
        Recover a term by an oriented DFS from the root, building up a term
        in continuation-passing style.
        """
        from discopy.closed import (
            Abstraction,
            Application,
            Coeval,
            Eval,
            Exp,
            Variable,
        )

        self._assert_rooted_trivalent_map()
        names = tuple(input_names or (f"x{i}" for i in range(len(self.dom))))
        if len(names) != len(self.dom):
            raise ValueError

        variables = tuple(
            Variable(obj, name)
            for obj, name in zip(self.dom, names)
        )
        counter = len(variables)

        def fresh(obj):
            nonlocal counter
            variable = Variable(obj, f"x{counter}")
            counter += 1
            return variable

        def dfs(port, bound_ports, continuation):
            port = self.edge[port]
            if port in bound_ports:
                return continuation(bound_ports[port])

            node = self.ports[port]
            if node.kind == "input":
                return continuation(variables[node.i])
            if node.kind in BOUNDARY_PORTS or node.depth is None:
                raise ValueError

            box = self.boxes[node.depth]
            box_ports = self.box_port_indices[node.depth]

            if isinstance(box, Eval):
                if node.kind != "cod" or node.i != 0:
                    raise ValueError
                func_port, arg_port = [
                    i for i in box_ports if self.ports[i].kind == "dom"]
                return dfs(func_port, bound_ports, lambda func:
                    dfs(arg_port, bound_ports, lambda arg:
                        continuation(Application(func, arg))
                    )
                )

            if isinstance(box, Coeval):
                cod = self.ports[port].obj
                if node.kind != "cod" or node.i != 0\
                        or not isinstance(cod, Exp):
                    raise ValueError
                body_port, = [
                    i for i in box_ports if self.ports[i].kind == "dom"]
                parameter_port, = [
                    i for i in box_ports
                    if self.ports[i].kind == "cod" and i != port]
                variable = fresh(cod.exponent)
                return dfs(
                    body_port,
                    bound_ports | {parameter_port: variable},
                    lambda body: continuation(Abstraction(variable, body)))

            raise ValueError

        return dfs(self.n_ports - 1, {}, lambda term: term)

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

    def to_dot(
            self, engine="neato", seed=None, graph_attr=None,
            boundary_labels=True,
            box_labels=None) -> str:
        """
        Encode the combinatorial map as Graphviz DOT.

        The drawing has one node per box, one point per boundary port, and one
        edge per 2-cycle of ``edge``. Port indices are shown as edge endpoint
        labels rather than drawn as separate nodes.
        """
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

        def escape(value):
            return str(value).replace("\\", "\\\\").replace('"', r'\"')

        def attr_string(attributes):
            return ", ".join(
                f'{key}="{escape(value)}"'
                for key, value in attributes.items())

        def box_label(box):
            if box_labels is not None:
                return box_labels(box)
            arity = len(box.dom), len(box.cod)
            return getattr(box, "drawing_name", None)\
                or getattr(box, "name", None)\
                or f"{arity[0]}->{arity[1]}"

        def boundary_label(port):
            if not boundary_labels:
                return ""
            return f"{port.kind} {port.i}"

        lines = [
            "graph combinatorial_map {",
            f"  graph [{attr_string(attrs)}];",
            '  node [shape=circle, color=black, fontname="Helvetica", '
            'fontsize="12"];',
            '  edge [color=black, penwidth="1.4", fontsize="9"];',
        ]

        port_nodes = {}
        for vertex in range(len(self.boxes)):
            box = self.boxes[vertex]
            attributes = dict(
                label=box_label(box), width="0.32", height="0.32")
            lines.append(
                f"  v{vertex} [{attr_string(attributes)}];")
            for port_index in self.node_cycles[vertex]:
                port_nodes[port_index] = f"v{vertex}"
        for port_index, port in enumerate(self.ports):
            if port.kind in BOUNDARY_PORTS:
                attributes = dict(
                    shape="point", label="", width="0.08", height="0.08",
                    xlabel=boundary_label(port))
                lines.append(
                    f"  b{port_index} [{attr_string(attributes)}];")
                port_nodes[port_index] = f"b{port_index}"

        def node_name(port_index):
            return port_nodes[port_index]

        def edge_direction(left, right):
            left_in = port_direction(self.ports[left]) == "in"
            right_in = port_direction(self.ports[right]) == "in"
            if left_in and right_in:
                return "both"
            if left_in:
                return "back"
            if right_in:
                return "forward"
            return "none"

        for i, j in enumerate(self.edge):
            if i < j:
                attributes = dict(
                    len="0.85", taillabel=i, headlabel=j,
                    labeldistance="1.6", dir=edge_direction(i, j))
                lines.append(
                    f'  {node_name(i)} -- {node_name(j)} '
                    f'[{attr_string(attributes)}];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def draw(
            self, path=None, engine="neato", format=None, seed=None,
            show=True, graph_attr=None, boundary_labels=True,
            trivalent_symbols=True, box_labels=None, block=True):
        """
        Draw as a combinatorial map using Graphviz.

        This is intended for map-like pictures rather than the usual DisCoPy
        box-and-wire drawing.

        If ``path`` ends in ``.dot`` or ``.gv``, write DOT source. Otherwise,
        render with Graphviz. When ``show`` is true, display the rendered graph
        in a matplotlib window.
        """
        dot = self.to_dot(
            engine=engine, seed=seed, graph_attr=graph_attr,
            boundary_labels=boundary_labels,
            box_labels=box_labels)

        path = None if path is None else str(path)
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
