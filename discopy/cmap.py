# -*- coding: utf-8 -*-

"""
Combinatorial maps with interfaces.

See `combinatorial map
<https://en.wikipedia.org/wiki/Combinatorial_map>`_ for background.

The ports of a map are ordered as in :mod:`discopy.hypergraph`: inputs,
then the domain and codomain ports of each box, then outputs. A map is given by
two permutations on these ports:

* ``edge`` is a fixpoint-free involution pairing ports into wires;
* ``node`` is derived from the canonical counter-clockwise port order of boxes.

Their composite determines the faces of the map. Closed wire components are
stored separately in ``scalars`` together with their types.

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
from io import BytesIO
from math import lcm
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
    from discopy.monoidal import Ob, Ty, Diagram, Box, Functor


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
        """ Whether the port is drawn on the input side of a node. """
        return self == "input" or self == "dom"

    @property
    def is_output(self) -> bool:
        """ Whether the port is drawn on the output side of a node. """
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
    def direction(self) -> Literal["in"] | Literal["out"]:
        """ Return whether the port is an input or an output. """
        is_adjoint = bool(getattr(self.obj, "z", 0) % 2)
        if self.kind.is_input:
            return "out" if is_adjoint else "in"
        return "in" if is_adjoint else "out"


class CMap[C0: Pregroup, C1: CMap](
    CompactCategory[C0, C1], NamedGeneric['functor']
):
    """
    An oriented bijective hypergraph with interfaces.

    The edge involution gives the wires, while the cyclic order of ports around
    each box gives the orientation.

    Parameters:
        dom : The domain of the map.
        cod : The codomain of the map.
        boxes : The boxes inside the map.
        edge : A fixpoint-free involution on ports.
        offsets : Optional drawing offsets, preserved through conversion.
        scalars : The types of closed wire components with no ports.

    Example
    -------
    >>> from discopy.compact import Ty, Box
    >>> x, y, z = map(Ty, "xyz")
    >>> cmap = Box("f", x @ y, z).to_map()
    >>> [port.kind.value for port in cmap.ports]
    ['input', 'input', 'dom', 'dom', 'cod', 'output']
    >>> tuple(cmap.edge), cmap.node_cycles
    ((2, 3, 0, 1, 5, 4), ((2, 3, 4),))
    """
    functor: ClassVar[Functor]
    require_planar: ClassVar[bool] = True
    category = classproperty(lambda cls: cls.functor.dom)
    ob = classproperty(lambda cls: cls.category.ob)

    def __init__(
            self, dom: C0, cod: C0, boxes: tuple[Box, ...],
            edge: Iterable[int],
            offsets: tuple[int | None, ...] | None = None,
            scalars: tuple[C0, ...] = ()):
        assert_isinstance(dom, self.category.ob)
        assert_isinstance(cod, self.category.ob)
        for box in boxes:
            assert_isinstance(box, self.category)
        for scalar in scalars:
            assert_isinstance(scalar, self.category.ob)
        self.dom, self.cod, self.boxes = dom, cod, tuple(boxes)
        self.offsets = offsets or tuple(len(boxes) * [None])
        if len(self.offsets) != len(self.boxes):
            raise ValueError
        self.scalars = tuple(scalars)

        self.edge = Permutation(edge, len(self.ports))
        self.validate()

    @property
    def ports(self) -> list[Port]:
        """ The ports in fixed hypergraph order. """
        def port(kind, i, obj, depth):
            return Port(
                kind, i=i, obj=obj, depth=depth,
                side="up" if kind.is_input else "down")

        inputs = [port(PortKind.INPUT, i=i, obj=obj, depth=float('-inf'))
                  for i, obj in enumerate(self.dom)]
        box_ports = sum([[
            port(kind, i=i, obj=obj, depth=depth)
            for i, obj in enumerate(typ)]
            for depth, box in enumerate(self.boxes)
            for kind, typ in [
                (PortKind.DOM, box.dom), (PortKind.COD, box.cod)]], [])
        outputs = [port(PortKind.OUTPUT, i=i, obj=obj, depth=float('+inf'))
                   for i, obj in enumerate(self.cod)]
        return inputs + box_ports + outputs

    @property
    def n_ports(self) -> int:
        """ The number of ports. """
        return len(self.ports)

    @property
    def box_port_indices(self) -> tuple[tuple[int, ...], ...]:
        """ The consecutive port indices belonging to each box. """
        result, start = [], len(self.dom)
        for box in self.boxes:
            stop = start + len(box.dom @ box.cod)
            result.append(tuple(range(start, stop)))
            start = stop
        return tuple(result)

    @property
    def node_cycles(self) -> tuple[tuple[int, ...], ...]:
        """ The oriented port cycle of each box, including empty boxes. """
        result = []
        for box_ports in self.box_port_indices:
            if not box_ports:
                result.append(())
                continue
            result.append(self.node.cycle(box_ports[0]))
        return tuple(result)

    @property
    def face_permutation(self) -> Permutation:
        """ The face permutation ``edge ; node``. """
        return self.edge.then(self.node)

    @property
    def face_cycles(self) -> tuple[tuple[int, ...], ...]:
        """ The cycles of the face permutation. """
        return self.face_permutation.cycles()

    @property
    def euler_characteristic(self) -> int:
        """
        Euler characteristic ``V - E + F`` with boundary at infinity.

        For open maps, the input and output ports are treated as points on one
        outer boundary circle, ordered as inputs left-to-right followed by
        outputs right-to-left. Thus a connected planar open map has Euler
        characteristic 2.

        >>> from discopy.symmetric import Ty, Box, Swap
        >>> x, y, z = map(Ty, "xyz")
        >>> f = Box("f", x @ y, z)
        >>> f.to_map().euler_characteristic
        2
        >>> (Swap(y, x) >> f).to_map().euler_characteristic
        0
        """
        if not self.n_ports:
            return len(self.boxes) + len(self.scalars)

        cycles, vertices = [], len(self.boxes)
        for box, box_ports in zip(self.boxes, self.box_port_indices):
            dom_ports = box_ports[:len(box.dom)]
            cod_ports = box_ports[len(box.dom):]
            cycle = tuple(reversed(dom_ports)) + tuple(cod_ports)
            if cycle:
                cycles.append(cycle)

        boundary = tuple(range(len(self.dom))) + tuple(range(
            self.n_ports - 1, self.n_ports - len(self.cod) - 1, -1))
        if boundary:
            cycles.append(boundary)
            vertices += 1

        node = Permutation.from_cycles(cycles, self.n_ports)
        return vertices - self.n_ports // 2\
            + len(self.edge.then(node).cycles()) + len(self.scalars)

    @property
    def node(self) -> Permutation:
        """
        The canonical box orientation.

        Box ports are already in their canonical local counter-clockwise
        order, so each box contributes its consecutive port interval.
        """
        cycles = list(self.box_port_indices)
        return Permutation.from_cycles(cycles, len(self.ports))

    def validate(self):
        """ Validate the edge involution, wires and required planarity. """
        ports = self.ports
        if not self.edge.is_fixpoint_free_involution():
            raise ValueError

        for i, j in enumerate(self.edge):
            if i > j:
                continue
            type(self).validate_wire(ports[i], ports[j])

        if not self.require_planar and self.n_ports\
                and self.euler_characteristic != 2:
            raise AxiomError(messages.NOT_PLANAR.format(self))

    def _new_scalars(self, edge, glue, removed, objects) -> tuple:
        """
        Compute the scalar types created by a gluing operation.
        """
        def scalar_type(obj):
            result = obj if isinstance(obj, self.category.ob)\
                else self.ob(obj)
            return result.r if getattr(result, "z", 0) % 2 else result

        removed, seen, result = set(removed), set(), []
        for start in removed:
            if start in seen:
                continue
            stack, internal = [start], True
            seen.add(start)
            while stack:
                port = stack.pop()
                for neighbor in (edge[port], glue[port]):
                    if neighbor not in removed:
                        internal = False
                    elif neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            if internal:
                result.append(scalar_type(objects[start]))
        return tuple(result)

    @classmethod
    def validate_wire(cls, source: Port, target: Port):
        """
        Validate a wire between two ports.

        Subclasses override this hook to enforce their categorical structure.

        Raises:
            AxiomError : If the types or orientations are incompatible.
        """
        def compatible(left, right):
            return left == right or getattr(left, "r", None) == right\
                or left == getattr(right, "r", None)

        def oriented(left, right):
            return left.kind.is_positive and right.kind.is_negative

        def boundary_adjunction(left, right):
            return left.kind.is_boundary and right.kind.is_boundary\
                and left.direction != right.direction

        def compact_adjunction(left, right):
            return left.kind.is_boundary != right.kind.is_boundary\
                and left.direction == right.direction

        if not compatible(source.obj, target.obj):
            raise AxiomError(messages.NOT_ADJOINT.format(
                source.obj, target.obj))
        if not (
                oriented(source, target)
                or oriented(target, source)
                or boundary_adjunction(source, target)
                or compact_adjunction(source, target)):
            raise AxiomError(
                messages.NOT_TRACEABLE.format(source, target)
            )

    def __repr__(self):
        def port_repr(index, port):
            port_depth = getattr(port, "depth", None)
            depth = "" if port_depth is None else f"@{port_depth}"
            return (
                f"{port.kind}{depth}[{port.i}]:{port.obj}:"
                f"{port.side}/{port.direction}"
                f"->{self.edge[index]}")

        ports = tuple(
            port_repr(index, port)
            for index, port in enumerate(self.ports))
        return factory_name(type(self))\
            + f"(dom={self.dom!r}, cod={self.cod!r}, " \
              f"boxes={self.boxes!r}, edge={self.edge!r}, " \
              f"ports={ports!r}, scalars={self.scalars!r})"

    def __eq__(self, other: Any):
        return isinstance(other, CMap) and (
            self.dom, self.cod, self.boxes, self.edge, self.scalars
        ) == (
            other.dom, other.cod, other.boxes, other.edge, other.scalars)

    def __hash__(self):
        return hash((
            self.dom, self.cod, self.boxes, self.edge, self.scalars))

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
            + [(2 * left + i, 2 * left + right + i)
               for i in range(right)],
            n_ports)
        return cls(box.dom, box.cod, (box, ), edge)

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
        factory = cls if cls.functor is not None else cls[
            type(old), type(old).functor]
        return factory.functor(
            ob=lambda typ: typ, ar=factory.from_box,
            dom=type(old), cod=factory)(old)

    @classmethod
    def from_hypergraph(cls, old: hypergraph.Hypergraph) -> CMap:
        """
        Build a combinatorial map from a bijective hypergraph.

        Raises:
            ValueError : If a spider is not incident to zero or two ports.
        """
        if not old.is_bijective:
            raise ValueError
        factory = cls if cls.functor is not None else cls[
            type(old).category, type(old).functor]
        return factory(
            old.dom, old.cod, old.boxes, old.bijection,
            offsets=old.offsets)

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
        if not getattr(left, "r", left[::-1]) == right:
            raise AxiomError
        size = len(left)
        edge = Permutation.from_transpositions(
            ((i, size + size - 1 - i) for i in range(size)),
            2 * size)
        return cls(left @ right, cls.ob(), (), edge)

    @classmethod
    def caps(cls, left: Ty, right: Ty) -> CMap:
        """ A cap encoded as boundary wiring between adjoint types. """
        if not getattr(left, "r", left[::-1]) == right:
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
            This will use the free closed structure obtained from the map
            representation by introducing adjoint ports, even if the host
            category already has closed structure.

        Parameters:
            n : The number of objects to curry.
            left : Whether to curry on the left or right.

        >>> from discopy.compact import Ty, Box
        >>> X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
        >>> f = Box("f", X @ Y, Z).to_map()
        >>> assert f.curry().uncurry() == f
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

    @classmethod
    def spiders(
            cls, n_legs_in: int, n_legs_out: int,
            typ: Ty, phases=None) -> CMap:
        """ Spiders are kept as boxes, including their phase data. """
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
        >>> scalar.scalars == (x,)
        True
        """
        if not self.cod == other.dom:
            raise AxiomError(messages.TYPE_ERROR.format(other.dom, self.cod))
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
        objects = {i: port.obj for i, port in enumerate(self.ports)}
        objects.update({
            shift + i: port.obj for i, port in enumerate(other.ports)})
        scalars = self.scalars + other.scalars\
            + self._new_scalars(edge, glue, removed, objects)

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
        return type(self)(
            dom, cod, boxes, edge, offsets=offsets,
            scalars=scalars)

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

        trace_pair = dict(zip(traced_inputs, traced_outputs))
        trace_pair.update(dict(zip(traced_outputs, traced_inputs)))
        removed = set(trace_pair)
        kept = [i for i in range(self.n_ports) if i not in removed]
        mapping = {old: new for new, old in enumerate(kept)}
        scalars = self.scalars + self._new_scalars(
            dict(enumerate(self.edge)), trace_pair, removed,
            {i: port.obj for i, port in enumerate(self.ports)})

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
        return type(self)(
            dom, cod, self.boxes, edge, offsets=self.offsets,
            scalars=scalars)

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
        return type(self)(
            dom, cod, boxes, edge, offsets=offsets,
            scalars=self.scalars + other.scalars)

    def interchange(self, i: int, j: int) -> CMap:
        """
        Interchange boxes at indices ``i`` and ``j``.

        The edge permutation is relabeled so that ports follow the canonical
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
        return type(self)(
            self.dom, self.cod, boxes, edge, offsets=offsets,
            scalars=self.scalars)

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
        box_outputs = (new_index + 1, new_index + 2)
        box_root = box_outputs[root_index]
        box_parameter = box_outputs[1 - root_index]
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

        return type(self)(
            new_dom, cod, boxes, edge, offsets=offsets,
            scalars=self.scalars)

    def to_hypergraph(self) -> hypergraph.Hypergraph:
        """
        Forget orientation and return the underlying bijective hypergraph.

        >>> from discopy.compact import Ty, Box, CMap
        >>> x, y = map(Ty, "xy")
        >>> cmap = Box("f", x, y).to_map()
        >>> CMap.from_hypergraph(cmap.to_hypergraph()) == cmap
        True
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

        >>> from discopy.compact import Ty, Box
        >>> x, y = map(Ty, "xy")
        >>> cmap = Box("f", x, y).to_map()
        >>> cmap.to_diagram().to_map() == cmap
        True
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

    def to_dot(
            self, engine="dot", seed=None, graph_attr=None,
            port_indices=False) -> str:
        """
        Encode the combinatorial map as Graphviz DOT.

        The drawing has HTML-table nodes for the boundary interfaces and for
        each box, with one table port for each object in the signature, and
        one direct edge per 2-cycle of ``edge``.

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
            text = escape_html(port.i) if port_indices else ""
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
            box_ports = self.box_port_indices[vertex]
            dom_ports = box_ports[:len(box.dom)]
            cod_ports = box_ports[len(box.dom):]
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
            for port_index in self.box_port_indices[vertex]:
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

        def node_name(port_index):
            return port_nodes[port_index]

        def port_label(port_index):
            return self.ports[port_index].obj

        def edge_labels(left, right):
            left_label, right_label = port_label(left), port_label(right)
            if left_label == right_label:
                return dict(label=left_label)
            return dict(taillabel=left_label, headlabel=right_label)

        for i, j in enumerate(self.edge):
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
            show=None, graph_attr=None, boundary_labels=True,
            box_labels=None, port_indices=False, block=True):
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
            boundary_labels : Accepted for drawing API compatibility.
            box_labels : Accepted for drawing API compatibility.
            port_indices : Whether to display port indices.
            block : Whether displaying blocks execution.

        >>> from discopy.compact import Ty, Box
        >>> X, Y, Z = Ty("X"), Ty("Y"), Ty("Z")
        >>> Box("f", X @ Y, Z).to_map().curry().draw(
        ...     path="docs/_static/cmap/curry.png", show=False)

        .. image:: /_static/cmap/curry.png
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
