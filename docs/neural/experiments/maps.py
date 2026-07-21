# -*- coding: utf-8 -*-

"""
The wirings of the three sudoku maps, and the routing helper that makes
message passing resumable.

Each builder returns a *closed* :class:`discopy.neural.CMap` together with a
description of the port layout of a cell, so that the models know where to
inject clues and where to read states. Only the wiring and the port types
differ between the three: A and C wire cells to shared factor boxes, B wires
peers directly; A and C carry a message of width ``dim`` on a cell-to-unit
wire, B carries a full hidden state of width ``state_dim`` on a peer wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from discopy.neural import CMap, Dim, Network

if TYPE_CHECKING:
    import torch

N = 9
N_CELLS = 81


@lru_cache(maxsize=None)
def positional_ids(n: int = N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ The row, column and block index of each of the ``n * n`` cells. """
    index = np.arange(n * n)
    root = int(round(n ** 0.5))
    row, col = index // n, index % n
    block = (row // root) * root + (col // root)
    return row, col, block


@lru_cache(maxsize=None)
def peers_of(n: int = N) -> tuple[tuple[int, ...], ...]:
    """ The cells each cell must differ from: its row, column and block. """
    row, col, block = positional_ids(n)
    return tuple(tuple(
        other for other in range(n * n)
        if other != cell and (
            row[other] == row[cell] or col[other] == col[cell]
            or block[other] == block[cell]))
        for cell in range(n * n))


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


def _cell_ports(layout: Layout) -> Dim:
    """ The codomain of a cell box, from its layout. """
    dim = Dim(layout.message_width) ** layout.n_message
    dim = dim @ Dim(layout.state_width) ** 2 @ Dim(layout.clue_width) ** 2
    if layout.answer:
        dim = dim @ Dim(layout.answer_width) ** 2
    return dim


def _self_loops(wires: list, cell: int, layout: Layout) -> None:
    """ Wire a cell's state, clue and answer ports to themselves. """
    for pair in (layout.state, layout.clue, layout.answer):
        if pair:
            wires.append(((cell, pair[0]), (cell, pair[1])))


def build_factor_graph(cell_module, factor_module, dim: int, state_dim: int,
                       answer_dim: int = 0, n: int = N
                       ) -> tuple[CMap, Layout]:
    """
    The bipartite factor graph of models A and C: one shared cell box per
    cell with three message ports, one shared factor box per row, column and
    block, and a wire from each cell to each of its three units.

    Parameters:
        cell_module : The shared module of every cell box.
        factor_module : The shared module of every unit box.
        dim : The width of a message and of a clue embedding.
        state_dim : The width of a cell's recurrent state.
        answer_dim : The width of model C's answer loop, ``0`` for model A.
        n : The size of the grid.
    """
    n_cells = n * n
    row, col, block = positional_ids(n)
    layout = Layout(
        n_message=3, message_width=dim, state=(3, 4), clue=(5, 6),
        answer=(7, 8) if answer_dim else (),
        state_width=state_dim, clue_width=dim, answer_width=answer_dim,
        n_cells=n_cells)
    cell = Network("cell", Dim(0), _cell_ports(layout), module=cell_module)
    factors = tuple(
        Network("unit", Dim(0), Dim(dim) ** n, module=factor_module)
        for _ in range(3 * n))

    units = [(int(row[i]), n + int(col[i]), 2 * n + int(block[i]))
             for i in range(n_cells)]
    free = [0] * (3 * n)
    wires: list = []
    for index in range(n_cells):
        for position, unit in enumerate(units[index]):
            wires.append(((index, position), (n_cells + unit, free[unit])))
            free[unit] += 1
        _self_loops(wires, index, layout)
    assert all(slot == n for slot in free), "a unit box is not full"
    return CMap.from_wiring((cell, ) * n_cells + factors, wires), layout


def build_clique(cell_module, state_dim: int, dim: int, n: int = N
                 ) -> tuple[CMap, Layout]:
    """
    The pairwise peer clique of model B: one shared cell box per cell with
    one message port per peer, and one wire between each pair of peers. No
    factor boxes, and the wires carry a full hidden state rather than a
    message of width ``dim``.

    The state loop carries the concatenated hidden and cell state of the
    ``LSTMCell``, hence its width of ``2 * state_dim``.

    Parameters:
        cell_module : The shared module of every cell box.
        state_dim : The width of a hidden state, i.e. of a peer wire.
        dim : The width of a clue embedding.
        n : The size of the grid.
    """
    n_cells = n * n
    peers = peers_of(n)
    n_peers = len(peers[0])
    layout = Layout(
        n_message=n_peers, message_width=state_dim,
        state=(n_peers, n_peers + 1), clue=(n_peers + 2, n_peers + 3),
        answer=(), state_width=2 * state_dim, clue_width=dim, answer_width=0,
        n_cells=n_cells)
    cell = Network("cell", Dim(0), _cell_ports(layout), module=cell_module)

    wires: list = []
    for index in range(n_cells):
        for other in peers[index]:
            if index < other:
                wires.append(((index, peers[index].index(other)),
                              (other, peers[other].index(index))))
        _self_loops(wires, index, layout)
    return CMap.from_wiring((cell, ) * n_cells, wires), layout


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
