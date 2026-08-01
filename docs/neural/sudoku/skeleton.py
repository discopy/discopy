# -*- coding: utf-8 -*-

"""
The sudoku combinatorics, and the two skeletons they induce.

Everything structural about the game is here and nothing else: which cell
sits in which row, column and block (:func:`positional_ids`), which cells
must differ (:func:`peers_of`), and the two abstract wirings a solver can
interpret -- the bipartite cell/unit factor graph and the pairwise peer
clique -- obtained by handing those combinatorics to the generic builders
of :mod:`discopy.neural.skeleton`.  The roles and the signatures live in
:mod:`sudoku.signature`; only the membership structure is sudoku's.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from discopy.neural import skeleton as skeletons
from sudoku import signature
from sudoku.config import N


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


@lru_cache(maxsize=None)
def memberships(n: int = N) -> tuple[tuple[int, ...], ...]:
    """
    The three units each cell belongs to -- its row, its column and its
    block, numbered in that order.

    Parameters:
        n : The size of the grid.
    """
    row, col, block = positional_ids(n)
    return tuple(
        (int(row[i]), n + int(col[i]), 2 * n + int(block[i]))
        for i in range(n * n))


@lru_cache(maxsize=None)
def factor_graph(n: int = N):
    """
    The bipartite factor graph of the factor-graph and recursion solvers:
    every cell is wired to the three units it belongs to.

    Parameters:
        n : The size of the grid.

    Example
    -------
    >>> grid = factor_graph()
    >>> len(grid.boxes), grid.cmap.n_ports // 2
    (108, 486)
    """
    return skeletons.from_incidence(
        memberships(n), signature.cell(3), signature.unit(n))


@lru_cache(maxsize=None)
def clique(n: int = N):
    """
    The pairwise peer clique of the clique solver: every cell is wired to
    each of its peers.

    Parameters:
        n : The size of the grid.

    Example
    -------
    >>> grid = clique()
    >>> len(grid.boxes), grid.cmap.n_ports // 2
    (81, 1053)
    """
    peers = peers_of(n)
    return skeletons.from_relation(peers, signature.peer_cell(len(peers[0])))
