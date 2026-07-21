# -*- coding: utf-8 -*-

"""
The semantics of the three sudoku maps, and the routing helper that makes
message passing resumable.

The *syntax* of each model -- which cell talks to which unit or peer -- is an
abstract, torch-free skeleton from :mod:`experiments.skeleton`, whose atomic
types are roles (``message``, ``state``, ``clue``, ...) rather than widths.
This module fills the skeleton in: one :class:`discopy.neural.Functor` per
model sends each role to the :class:`Dim` it carries and each abstract box to
the :class:`Network` computing it, and :func:`interpret` applies the functor
to the skeleton, giving the closed :class:`discopy.neural.CMap` the model
runs.  Models A and C interpret the *same* skeleton -- their wiring is
identical -- with two different functors: model A sends the ``answer`` role
to ``Dim(0)``, the monoidal unit, which erases the answer loop altogether;
model C sends it to ``Dim(y_dim)``.

Each builder returns the interpreted map together with a :class:`Layout`, a
description of the port layout of a cell read off the same functor, so that
the models know where to inject clues and where to read states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from discopy import frobenius
from discopy.neural import CMap, Dim, Functor, Network
from discopy.utils import assert_isinstance

from experiments import skeleton
from experiments.skeleton import (  # noqa: F401 -- re-exported for callers
    N, N_CELLS, peers_of, positional_ids)

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Layout:
    """
    Where a cell keeps what, as positions in its logical port order.

    Parameters:
        n_message : The number of message ports of a cell.
        message_width : The width of one message port.
        state : The pair of state-loop positions.
        clue : The pair of clue-loop positions.
        answer : The pair of answer-loop positions, empty except for model C.
        state_width : The width of a state port.
        clue_width : The width of a clue port.
        answer_width : The width of an answer port.
        n_cells : The number of cell boxes, which come first in the map.
    """
    n_message: int
    message_width: int
    state: tuple[int, int]
    clue: tuple[int, int]
    answer: tuple[int, ...]
    state_width: int
    clue_width: int
    answer_width: int
    n_cells: int = N_CELLS

    @property
    def state_offset(self) -> int:
        """ Where the state starts inside a cell's flat message vector. """
        return self.n_message * self.message_width


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
        for pos, role in enumerate(skeleton.roles_of(abstract, index)):
            width = functor(role)
            assert len(width) <= 1, f"{role} maps to the non-atomic {width}"
            erased[index, pos] = not len(width)
            position[index, pos] = cursor
            cursor += len(width)

    wires = []
    for (one, other) in skeleton.wires_of(abstract):
        if erased[one] and erased[other]:
            continue
        assert not (erased[one] or erased[other]), \
            f"the wire {one} -- {other} is only erased at one end"
        wires.append(((one[0], position[one]), (other[0], position[other])))
    return CMap.from_wiring(boxes, wires)


def layout_of(functor: Functor, abstract: frobenius.CMap) -> Layout:
    """
    The port layout of a cell under a functor: where each surviving role
    sits among the concrete ports of a cell box, and how wide it is.  Read
    off the same functor as :func:`interpret`, so the two cannot drift.

    Parameters:
        functor : The neural functor giving the widths.
        abstract : The skeleton whose cells to lay out.
    """
    roles = skeleton.roles_of(abstract, 0)
    kept = [role for role in roles if len(functor(role))]

    def positions(role):
        return tuple(i for i, other in enumerate(kept) if other == role)

    def width(role):
        return sum(functor(role).inside)

    message = kept[0]
    assert message not in skeleton.LOOP_ROLES, "a cell starts with a loop"
    return Layout(
        n_message=len(positions(message)), message_width=width(message),
        state=positions(skeleton.STATE), clue=positions(skeleton.CLUE),
        answer=positions(skeleton.ANSWER),
        state_width=width(skeleton.STATE), clue_width=width(skeleton.CLUE),
        answer_width=width(skeleton.ANSWER) if positions(skeleton.ANSWER)
        else 0,
        n_cells=sum(box.name == "cell" for box in abstract.boxes))


def _functor(ob: dict, modules: dict, abstract: frobenius.CMap) -> Functor:
    """
    The neural functor with the given object map, sending each abstract box
    of a skeleton to a :class:`Network` of the image type around the module
    of the same name.  One shared module means one shared box.

    Parameters:
        ob : Map from atomic role to the :class:`Dim` it carries.
        modules : Map from box name to the torch module filling it.
        abstract : The skeleton whose boxes to interpret.
    """
    types = Functor(ob=ob, dom=frobenius.Diagram)
    networks = {
        box: Network(box.name, types(box.dom), types(box.cod),
                     module=modules[box.name])
        for box in dict.fromkeys(abstract.boxes)}
    return Functor(ob=ob, ar=networks, dom=frobenius.Diagram)


def factor_functor(cell_module, factor_module, dim: int, state_dim: int,
                   answer_dim: int = 0, n: int = N) -> Functor:
    """
    The functor interpreting :func:`experiments.skeleton.factor_graph` as
    model A when ``answer_dim`` is ``0`` -- the answer loop is erased -- and
    as model C otherwise.

    Parameters:
        cell_module : The shared module of every cell box.
        factor_module : The shared module of every unit box.
        dim : The width of a message and of a clue embedding.
        state_dim : The width of a cell's recurrent state.
        answer_dim : The width of model C's answer loop, ``0`` for model A.
        n : The size of the grid.
    """
    return _functor(
        ob={skeleton.MESSAGE: Dim(dim), skeleton.STATE: Dim(state_dim),
            skeleton.CLUE: Dim(dim), skeleton.ANSWER: Dim(answer_dim)},
        modules={"cell": cell_module, "unit": factor_module},
        abstract=skeleton.factor_graph(n))


def clique_functor(cell_module, state_dim: int, dim: int, n: int = N
                   ) -> Functor:
    """
    The functor interpreting :func:`experiments.skeleton.clique` as model B:
    a peer wire carries a full hidden state of width ``state_dim``, and the
    state loop carries the concatenated hidden and cell state of the
    ``LSTMCell``, hence its width of ``2 * state_dim``.

    Parameters:
        cell_module : The shared module of every cell box.
        state_dim : The width of a hidden state, i.e. of a peer wire.
        dim : The width of a clue embedding.
        n : The size of the grid.
    """
    return _functor(
        ob={skeleton.PEER: Dim(state_dim), skeleton.STATE: Dim(2 * state_dim),
            skeleton.CLUE: Dim(dim)},
        modules={"cell": cell_module},
        abstract=skeleton.clique(n))


def build_factor_graph(cell_module, factor_module, dim: int, state_dim: int,
                       answer_dim: int = 0, n: int = N
                       ) -> tuple[CMap, Layout]:
    """
    The bipartite factor graph of models A and C: the skeleton of
    :func:`experiments.skeleton.factor_graph` interpreted by
    :func:`factor_functor`.

    Parameters:
        cell_module : The shared module of every cell box.
        factor_module : The shared module of every unit box.
        dim : The width of a message and of a clue embedding.
        state_dim : The width of a cell's recurrent state.
        answer_dim : The width of model C's answer loop, ``0`` for model A.
        n : The size of the grid.
    """
    abstract = skeleton.factor_graph(n)
    functor = factor_functor(
        cell_module, factor_module, dim, state_dim, answer_dim, n)
    return interpret(functor, abstract), layout_of(functor, abstract)


def build_clique(cell_module, state_dim: int, dim: int, n: int = N
                 ) -> tuple[CMap, Layout]:
    """
    The pairwise peer clique of model B: the skeleton of
    :func:`experiments.skeleton.clique` interpreted by
    :func:`clique_functor`.

    Parameters:
        cell_module : The shared module of every cell box.
        state_dim : The width of a hidden state, i.e. of a peer wire.
        dim : The width of a clue embedding.
        n : The size of the grid.
    """
    abstract = skeleton.clique(n)
    functor = clique_functor(cell_module, state_dim, dim, n)
    return interpret(functor, abstract), layout_of(functor, abstract)


# --- resumable message passing --------------------------------------------

def route(cmap: CMap, outgoing) -> list:
    """
    Turn the per-box *outgoing* messages returned by a closed map into the
    per-port *incoming* messages that start the next round.

    This is one application of the ``edges`` involution: the message a box
    emits on a port arrives, next round, on the port at the other end of the
    wire. Composing ``forward`` with ``route`` therefore resumes message
    passing exactly, which is what model C's outer loop relies on.

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
