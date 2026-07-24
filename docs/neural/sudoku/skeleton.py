# -*- coding: utf-8 -*-

"""
The sudoku combinatorics, and the two skeletons they induce.

Everything structural about the game is here and nothing else: which cell
sits in which row, column and block (:func:`positional_ids`), which cells
must differ (:func:`peers_of`), and the two abstract wirings a solver can
interpret -- the bipartite cell/unit factor graph and the pairwise peer
clique -- obtained by handing those combinatorics to the generic builders
of :mod:`core.skeletons`.  The roles (``message``, ``state``, ``clue``,
...) and the shapes are the solver family's; only the membership structure
is sudoku's.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from core import skeletons
from core.skeletons import (  # noqa: F401 -- the family's roles, re-exported
    ANSWER, CLUE, LOOP_ROLES, MESSAGE, PEER, STATE)
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
def factor_graph(n: int = N):
    """
    The bipartite factor graph of the factor-graph and recursion solvers:
    every cell is wired to the three units it belongs to -- its row, its
    column and its block, numbered in that order.

    Parameters:
        n : The size of the grid.
    """
    row, col, block = positional_ids(n)
    return skeletons.factor_graph(tuple(
        (int(row[i]), n + int(col[i]), 2 * n + int(block[i]))
        for i in range(n * n)))


@lru_cache(maxsize=None)
def clique(n: int = N):
    """
    The pairwise peer clique of the clique solver: every cell is wired to
    each of its peers.

    Parameters:
        n : The size of the grid.
    """
    return skeletons.clique(peers_of(n))
