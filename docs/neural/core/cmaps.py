# -*- coding: utf-8 -*-

"""
Task-agnostic helpers for combinatorial maps: reading a closed map back
into boxes, wires and roles, interpreting an abstract skeleton into a
:class:`discopy.neural.CMap`, and the routing helpers that make message
passing resumable.

Nothing here knows about any particular task.  A task package (e.g.
``sudoku``) builds its abstract skeletons with :mod:`discopy.frobenius`,
defines the functors giving them widths and modules, and calls
:func:`interpret`; :class:`Router` then reads and writes families of ports
of the interpreted map.  Like :mod:`discopy.neural` itself, this module
does not import ``torch``: only :func:`route` and :class:`Router` need it,
at call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from discopy import frobenius
from discopy.neural import CMap, Functor, Network
from discopy.utils import assert_isinstance

if TYPE_CHECKING:
    import torch


# --- reading a closed map back --------------------------------------------

def logical_ports(cmap, index: int) -> tuple[int, ...]:
    """
    The global port indices of a box in logical order -- domain ports then
    codomain ports -- undoing the clockwise order which stores the codomain
    reversed.  Same convention as :meth:`discopy.neural.CMap.box_ports`,
    for maps of any category.

    Parameters:
        cmap : The map the box lives in.
        index : The index of the box.
    """
    start = len(cmap.dom)
    for box in cmap.boxes[:index]:
        start += len(box.dom) + len(box.cod)
    box = cmap.boxes[index]
    arity = len(box.dom)
    ports = tuple(range(start, start + arity + len(box.cod)))
    return ports[:arity] + tuple(reversed(ports[arity:]))


def wires_of(cmap) -> tuple:
    """
    The wires of a closed map as pairs of ``(box_index, port_position)``
    pairs, i.e. the inverse of :meth:`CMap.from_wiring`: positions count
    the domain ports of a box followed by its codomain ports.

    Parameters:
        cmap : The closed map to read.
    """
    assert not len(cmap.dom) and not len(cmap.cod), "the map is not closed"
    to_logical = {
        port: (index, position)
        for index in range(len(cmap.boxes))
        for position, port in enumerate(logical_ports(cmap, index))}
    return tuple((to_logical[i], to_logical[j])
                 for i, j in enumerate(cmap.edges) if i < j)


def roles_of(cmap, index: int) -> tuple:
    """
    The role of each logical port of a box, i.e. the atomic types of its
    domain followed by its codomain.

    Parameters:
        cmap : The map the box lives in.
        index : The index of the box.
    """
    box = cmap.boxes[index]
    return tuple(box.dom) + tuple(box.cod)


# --- interpreting a skeleton ----------------------------------------------

def interpret(functor: Functor, abstract: frobenius.CMap) -> CMap:
    """
    Apply a :class:`discopy.neural.Functor` to a closed abstract map,
    port by port: each box becomes its image :class:`Network`, each wire
    becomes a wire between the image ports.  This is the boundary between
    syntax and semantics: the skeleton fixes the combinatorics, the functor
    fixes the widths and the modules.

    The functor must send each atomic role to an atomic ``Dim`` -- one
    abstract port becomes one concrete port -- or to ``Dim(0)``, the
    monoidal unit, in which case the port vanishes from the image box and
    the wire is erased with it.  A wire may only be erased whole: a role
    wired to a surviving role cannot be erased.

    Parameters:
        functor : The neural functor giving the widths and the networks.
        abstract : The closed skeleton to interpret.
    """
    boxes = tuple(functor(box) for box in abstract.boxes)
    for box, image in zip(abstract.boxes, boxes):
        assert_isinstance(image, Network)
        assert (image.dom, image.cod) \
            == (functor(box.dom), functor(box.cod)), \
            f"the image of {box} does not have the image type"

    erased, position = {}, {}
    for index in range(len(abstract.boxes)):
        cursor = 0
        for pos, role in enumerate(roles_of(abstract, index)):
            width = functor(role)
            assert len(width) <= 1, f"{role} maps to the non-atomic {width}"
            erased[index, pos] = not len(width)
            position[index, pos] = cursor
            cursor += len(width)

    wires = []
    for (one, other) in wires_of(abstract):
        if erased[one] and erased[other]:
            continue
        assert not (erased[one] or erased[other]), \
            f"the wire {one} -- {other} is only erased at one end"
        wires.append(((one[0], position[one]), (other[0], position[other])))
    return CMap.from_wiring(boxes, wires)


# --- resumable message passing --------------------------------------------

def route(cmap: CMap, outgoing) -> list:
    """
    Turn the per-box *outgoing* messages returned by a closed map into the
    per-port *incoming* messages that start the next round.

    This is one application of the ``edges`` involution: the message a box
    emits on a port arrives, next round, on the port at the other end of the
    wire. Composing ``forward`` with ``route`` therefore resumes message
    passing exactly, which is what a segmented outer loop relies on.

    Parameters:
        cmap : The closed map whose boxes emitted the messages.
        outgoing : One tensor per box, in the logical port order of that box.
    """
    import torch
    widths = cmap.port_widths
    per_port: list = [None] * len(widths)
    for index, emitted in enumerate(outgoing):
        ports = cmap.box_ports(index)
        chunks = torch.split(emitted, [widths[port] for port in ports], -1)
        for port, chunk in zip(ports, chunks):
            per_port[port] = chunk
    return [per_port[cmap.edges[port]] for port in range(len(widths))]


class Router:
    """
    The vectorized :func:`route`: one cached permutation of the flat message
    tensor, so that a macro-step costs one ``gather`` rather than a Python
    loop over the boxes, plus cached indices to read and write a chosen
    family of ports.

    The permutation is built from exactly the same data as :func:`route` --
    ``box_ports``, ``port_widths`` and the ``edges`` involution -- so the fast
    path and the readable one are two spellings of one routing rule.

    Parameters:
        cmap : The closed map to route.
    """
    def __init__(self, cmap: CMap):
        import torch
        widths = cmap.port_widths
        position, cursor = [0] * len(widths), 0
        for index in range(len(cmap.boxes)):
            for port in cmap.box_ports(index):
                position[port] = cursor
                cursor += widths[port]
        assert cursor == sum(widths), "the map is not closed"
        offset, total, source = [], 0, []
        for port, width in enumerate(widths):
            offset.append(total)
            total += width
            partner = cmap.edges[port]
            assert widths[partner] == width, "a wire changes width"
            source.extend(range(position[partner], position[partner] + width))
        self.cmap, self.total = cmap, total
        self.offset, self.widths = offset, widths
        self._source = torch.tensor(source, dtype=torch.long)
        self._cache: dict = {}

    def _index(self, ports: tuple[int, ...], device) -> "torch.Tensor":
        """ The flat indices of a family of equally wide ports, cached. """
        import torch
        key = (ports, device)
        if key not in self._cache:
            width = self.widths[ports[0]]
            assert all(self.widths[port] == width for port in ports), \
                "ports of different widths cannot be read as one block"
            self._cache[key] = torch.tensor(
                [k for port in ports
                 for k in range(self.offset[port], self.offset[port] + width)],
                dtype=torch.long, device=device)
        return self._cache[key]

    def __call__(self, outgoing) -> "torch.Tensor":
        """ The flat incoming messages that start the next round. """
        import torch
        flat = torch.cat(list(outgoing), -1)
        if self._source.device != flat.device:
            self._source = self._source.to(flat.device)
        return flat[:, self._source]

    def read(self, flat, ports: tuple[int, ...]):
        """ The messages at ``ports``, of shape ``(batch, len(ports), w)``. """
        index = self._index(tuple(ports), flat.device)
        return flat.index_select(1, index).reshape(
            len(flat), len(ports), self.widths[ports[0]])

    def write(self, flat, ports: tuple[int, ...], values):
        """ A copy of ``flat`` with ``values`` written at ``ports``. """
        index = self._index(tuple(ports), flat.device)
        return flat.index_copy(
            1, index, values.reshape(len(flat), -1).to(flat.dtype))
