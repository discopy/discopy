# -*- coding: utf-8 -*-

"""
From syntax to semantics: interpreting a skeleton into a runnable map.

A :class:`~discopy.neural.skeleton.Skeleton` fixes the combinatorics; an
:class:`Interpretation` fixes the widths and the modules.  Applying one to
the other is :func:`interpret`, and what comes back is a :class:`Wiring`:
the closed :class:`~discopy.neural.CMap` a solver runs, the
:class:`Router` that reads and writes families of its ports, and the
global port index of every ``(box name, role)`` pair -- so nothing
downstream ever has to compute a port offset.

Because ``Dim(0)`` is the monoidal unit, an interpretation can send a role
to ``Dim(0)`` and thereby *erase* its ports and the wires on them.  That
is how one skeleton serves two models: sending the answer role to
``Dim(0)`` gives a plain factor graph, sending it to ``Dim(y)`` gives the
same factor graph with an answer loop.

Like :mod:`discopy.neural.core`, this module does not import ``torch``;
only :func:`route` and :class:`Router` need it, at call time.

Summary
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Interpretation
    Wiring
    Router
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from discopy.neural.core import CMap, Functor, Network
from discopy.utils import assert_isinstance

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Interpretation:
    """
    What a skeleton means: the width of every role and the module of every
    box name.

    Parameters:
        ob : The :class:`Dim` each atomic role carries, ``Dim(0)`` to
             erase it.
        ar : The torch module filling each box name.  One shared module
             means one shared box, hence one batched call per round for a
             whole family of sites.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Orbit, Signature
    >>> from discopy.neural.skeleton import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> pair = from_relation(((1, ), (0, )), node)
    >>> meaning = Interpretation(
    ...     ob={peer: Dim(3), state: Dim(5)}, ar={"cell": None})
    >>> meaning.widths[state]
    5
    """

    ob: Mapping = field(default_factory=dict)
    ar: Mapping = field(default_factory=dict)

    @property
    def widths(self) -> dict:
        """ The width each role carries, as a plain integer. """
        return {role: sum(dim.inside) for role, dim in self.ob.items()}

    def erases(self, role) -> bool:
        """
        Whether a role is sent to the monoidal unit, so that its ports and
        the wires on them vanish from the image.

        Parameters:
            role : The atomic role.
        """
        return not len(self.ob[role])

    def functor(self, skeleton) -> Functor:
        """
        The neural functor: each role to the ``Dim`` it carries, each
        abstract box to the :class:`Network` of the image type around the
        module of the same name.

        Parameters:
            skeleton : The skeleton whose boxes to interpret.
        """
        source = type(skeleton.cmap).category.ar
        types = Functor(ob_map=self.ob, dom=source)
        networks = {
            box: Network(box.name, types(box.dom), types(box.cod),
                         module=self.ar[box.name])
            for box in dict.fromkeys(skeleton.boxes)}
        return Functor(ob_map=self.ob, ar_map=networks, dom=source)


@dataclass(frozen=True)
class Wiring:
    """
    An interpreted skeleton: what a solver needs to run.

    Parameters:
        cmap : The closed :class:`~discopy.neural.CMap` to run.
        router : The router reading and writing families of its ports.
        ports : The global port indices of each ``(box name, role)``, in
                box order then position order.
        heads : The subset of :attr:`ports` a module *reads*: the outgoing
                copy of a traced role, every leg of an untraced one.
    """

    cmap: CMap
    router: "Router"
    ports: Mapping = field(default_factory=dict)
    heads: Mapping = field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        """ Message passing over the interpreted map. """
        return self.cmap(*args, **kwargs)

    @property
    def n_wires(self) -> int:
        """ The number of wires of the map. """
        return self.cmap.n_ports // 2


def interpret(interpretation: Interpretation, skeleton) -> Wiring:
    """
    Apply an interpretation to a skeleton, port by port: each box becomes
    its image :class:`Network`, each wire a wire between the image ports.

    A role must go to an atomic ``Dim`` -- one abstract port becomes one
    concrete port -- or to ``Dim(0)``, in which case the port vanishes and
    the wire is erased with it.  A wire may only be erased whole: a role
    wired to a surviving role cannot be erased.

    Parameters:
        interpretation : The widths and the modules.
        skeleton : The closed skeleton to interpret.

    Example
    -------
    >>> from discopy.frobenius import Ty
    >>> from discopy.neural import Dim, Orbit, Signature
    >>> from discopy.neural.skeleton import from_relation
    >>> peer, state = Ty("peer"), Ty("state")
    >>> node = Signature((Orbit(peer, 1), Orbit(state, traced=True)))
    >>> pair = from_relation(((1, ), (0, )), node)
    >>> kept = interpret(Interpretation(
    ...     {peer: Dim(3), state: Dim(5)}, {"cell": None}), pair)
    >>> kept.cmap.port_widths  # ports are stored clockwise, codomain last
    (5, 5, 3, 5, 5, 3)
    >>> kept.heads[("cell", state)], kept.ports[("cell", state)]
    ((1, 4), (1, 0, 4, 3))
    >>> erased = interpret(Interpretation(
    ...     {peer: Dim(3), state: Dim(0)}, {"cell": None}), pair)
    >>> erased.cmap.port_widths
    (3, 3)
    """
    functor = interpretation.functor(skeleton)
    boxes = tuple(functor(box) for box in skeleton.boxes)
    for box, image in zip(skeleton.boxes, boxes):
        assert_isinstance(image, Network)
        if (image.dom, image.cod) != (functor(box.dom), functor(box.cod)):
            raise ValueError(f"the image of {box} has the wrong type")

    erased, position = {}, {}
    for index, box in enumerate(skeleton.boxes):
        cursor = 0
        for place, role in enumerate(skeleton.signature(index).roles):
            width = functor(role)
            if len(width) > 1:
                raise ValueError(f"{role} maps to the non-atomic {width}")
            erased[index, place] = not len(width)
            position[index, place] = cursor
            cursor += len(width)

    wires = []
    for one, other in skeleton.wires():
        if erased[one] and erased[other]:
            continue
        if erased[one] or erased[other]:
            raise ValueError(
                f"the wire {one} -- {other} is only erased at one end")
        wires.append(((one[0], position[one]), (other[0], position[other])))
    cmap = CMap.from_wiring(boxes, wires)
    return Wiring(cmap, Router(cmap), *_families(interpretation, skeleton,
                                                 cmap, position))


def _families(interpretation: Interpretation, skeleton, cmap,
              position: Mapping) -> tuple[dict, dict]:
    """ The global port indices of each ``(box name, role)`` pair. """
    ports: dict = {}
    heads: dict = {}
    for index, box in enumerate(skeleton.boxes):
        signature = skeleton.signature(index)
        concrete = cmap.box_ports(index)
        for role in dict.fromkeys(signature.roles):
            if interpretation.erases(role):
                continue
            for key, places in ((ports, signature.positions(role)),
                                (heads, signature.heads(role))):
                key.setdefault((box.name, role), []).extend(
                    concrete[position[index, place]] for place in places)
    return ({key: tuple(value) for key, value in ports.items()},
            {key: tuple(value) for key, value in heads.items()})


# --- resumable message passing --------------------------------------------

def route(cmap: CMap, outgoing) -> list:
    """
    Turn the per-box *outgoing* messages returned by a closed map into the
    per-port *incoming* messages that start the next round.

    This is one application of the ``edges`` involution: the message a box
    emits on a port arrives, next round, on the port at the other end of
    the wire.  Composing ``forward`` with ``route`` therefore resumes
    message passing exactly, which is what a segmented outer loop relies
    on.

    Parameters:
        cmap : The closed map whose boxes emitted the messages.
        outgoing : One tensor per box, in the logical port order of that
                   box.
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
    The vectorized :func:`route`: one cached permutation of the flat
    message tensor, so that a macro-step costs one ``gather`` rather than a
    Python loop over the boxes, plus cached indices to read and write a
    chosen family of ports.

    The permutation is built from exactly the same data as :func:`route` --
    ``box_ports``, ``port_widths`` and the ``edges`` involution -- so the
    fast path and the readable one are two spellings of one routing rule.

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
