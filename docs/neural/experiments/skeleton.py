# -*- coding: utf-8 -*-

"""
The abstract wirings of the sudoku maps: pure syntax, no torch, no widths.

A skeleton is a closed :class:`discopy.frobenius.CMap` whose boxes are plain
:class:`discopy.frobenius.Box` -- no modules inside -- and whose atomic types
are the *roles* a port can play rather than the width it will carry:

* :data:`MESSAGE` : a cell-to-unit wire of the bipartite factor graph,
* :data:`PEER` : a cell-to-cell wire of the pairwise clique,
* :data:`STATE`, :data:`CLUE`, :data:`ANSWER` : the traced loops, each a
  self-wired pair of ports on a cell carrying its recurrent state, its clue
  and -- for model C -- its current answer between rounds.

Frobenius types are self-dual, which is what lets a wire connect two
codomain ports of the same role; and ``import discopy`` does not import
``torch``, so a skeleton can be built and checked -- degrees, involution,
loop positions -- on a machine with no torch at all.  What fills the nodes
is decided later, by the :class:`discopy.neural.Functor` of each model: see
:func:`experiments.maps.interpret`.  Typing each role as a distinct atomic
object is what gives the functor's object map one knob per width -- and,
since ``Dim(0)`` is the monoidal unit, sending a role to ``Dim(0)`` erases
its ports altogether, which is how one skeleton serves both model A (no
answer loop) and model C (an answer loop of width ``y_dim``).

There are two skeletons for the three models:

* :func:`factor_graph` : the bipartite cell/unit graph shared by models A
  and C -- one cell box per cell with three :data:`MESSAGE` ports, one unit
  box per row, column and block, a wire from each cell to each of its three
  units, and all three loops on every cell;
* :func:`clique` : the peer clique of model B -- one cell box per cell with
  one :data:`PEER` port per peer, a wire between each pair of peers, and
  state and clue loops.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from discopy.frobenius import Box, CMap, Ty

N = 9
N_CELLS = 81

#: The roles a port can play, as self-dual atomic types.
MESSAGE, PEER = Ty("message"), Ty("peer")
STATE, CLUE, ANSWER = Ty("state"), Ty("clue"), Ty("answer")

#: The roles that live on a traced loop, in the order they appear on a cell.
LOOP_ROLES = (STATE, CLUE, ANSWER)


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


def _loops(wires: list, cell: int, cod: Ty) -> None:
    """ Wire the consecutive pair of ports of each loop role to itself. """
    atoms = list(cod)
    for role in LOOP_ROLES:
        positions = [i for i, atom in enumerate(atoms) if atom == role]
        if positions:
            assert len(positions) == 2, f"{role} is not a pair of ports"
            wires.append(((cell, positions[0]), (cell, positions[1])))


@lru_cache(maxsize=None)
def factor_graph(n: int = N) -> CMap:
    """
    The bipartite factor graph shared by models A and C: one cell box per
    cell with three :data:`MESSAGE` ports and the three loops, one unit box
    per row, column and block with one :data:`MESSAGE` port per member, and
    a wire from each cell to each of its three units.

    Parameters:
        n : The size of the grid.
    """
    n_cells = n * n
    row, col, block = positional_ids(n)
    cell = Box(
        "cell", Ty(), MESSAGE ** 3 @ STATE ** 2 @ CLUE ** 2 @ ANSWER ** 2)
    unit = Box("unit", Ty(), MESSAGE ** n)

    units = [(int(row[i]), n + int(col[i]), 2 * n + int(block[i]))
             for i in range(n_cells)]
    free = [0] * (3 * n)
    wires: list = []
    for index in range(n_cells):
        for position, unit_index in enumerate(units[index]):
            wires.append(
                ((index, position), (n_cells + unit_index, free[unit_index])))
            free[unit_index] += 1
        _loops(wires, index, cell.cod)
    assert all(slot == n for slot in free), "a unit box is not full"
    return CMap.from_wiring((cell, ) * n_cells + (unit, ) * (3 * n), wires)


@lru_cache(maxsize=None)
def clique(n: int = N) -> CMap:
    """
    The pairwise peer clique of model B: one cell box per cell with one
    :data:`PEER` port per peer plus state and clue loops, and a wire
    between each pair of peers.  No unit boxes and no answer loop.

    Parameters:
        n : The size of the grid.
    """
    n_cells = n * n
    peers = peers_of(n)
    n_peers = len(peers[0])
    cell = Box("cell", Ty(), PEER ** n_peers @ STATE ** 2 @ CLUE ** 2)

    wires: list = []
    for index in range(n_cells):
        for other in peers[index]:
            if index < other:
                wires.append(((index, peers[index].index(other)),
                              (other, peers[other].index(index))))
        _loops(wires, index, cell.cod)
    return CMap.from_wiring((cell, ) * n_cells, wires)


# --- reading a skeleton back ----------------------------------------------

def logical_ports(cmap: CMap, index: int) -> tuple[int, ...]:
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


def wires_of(cmap: CMap) -> tuple:
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


def roles_of(cmap: CMap, index: int) -> tuple[Ty, ...]:
    """
    The role of each logical port of a box, i.e. the atomic types of its
    domain followed by its codomain.

    Parameters:
        cmap : The map the box lives in.
        index : The index of the box.
    """
    box = cmap.boxes[index]
    return tuple(box.dom) + tuple(box.cod)
