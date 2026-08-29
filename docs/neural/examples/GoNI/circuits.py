# -*- coding: utf-8 -*-

"""
The dataflow circuits of the GoNI study: the algorithm as a diagram.

An algorithm whose data flow does not depend on its data is a circuit: one
generator per elementary step, wired by the dependencies between the steps.
Interpreting the circuit with :mod:`discopy.neural` gives the model of the
Geometry of Neural Interaction: the wiring is the algorithm's own and the
generators are learned, so the same weights run at every input size and
out-of-distribution generalization is a property of the family, not a hope.

:func:`lcs` draws the dynamic-programming grid of the longest common
subsequence of two words, the circuit the benchmark calls ``lcs_length``:

    L[i][j] = L[i-1][j-1] + 1           if a[i] == b[j]
              max(L[i-1][j], L[i][j-1]) otherwise

The grid is not planar: the value of a cell is read by three later cells
and the symbols thread along the rows and down the columns, across the
value wires.  Every crossing is a :class:`discopy.symmetric.Permutation`
layer, which :meth:`~discopy.monoidal.Diagram.to_map` absorbs into the
wiring of the combinatorial map: the map of the grid holds one box per
cell and nothing else.  Symmetry is plumbing, not computation, which is
what lets the same construction scale from the training sizes to the
evaluation ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from discopy.monoidal import Diagram


def lcs(m: int, n: int, cell) -> Diagram:
    """
    The LCS grid on words of length ``m`` and ``n``, built from one shared
    ``cell`` with domain ``(a, b, diag, up, left)`` and codomain
    ``(b, down, diag, a, right)``.

    The domain reads, left to right: one ``(diag, a, left)`` triple per
    row, from row ``m`` down to row ``1``, then one ``(diag, b, up)``
    triple per column, from column ``1`` to ``n``.  The symbol wires carry
    the letters of the two words and every other wire carries the boundary
    value of the grid, i.e. zero.  The codomain starts with the bottom
    frontier of the grid and ends with one dead triple per row, from row
    ``m`` down to row ``1``; the answer ``L[m][n]`` is the wire at
    :func:`answer`.

    The construction is a row-major scan.  Between two rows the frontier
    holds one ``(diag, b, up)`` chunk per column; along a row the carried
    state is a ``(diag, a, left)`` triple.  Each column costs two layers:
    a permutation taking the carried ``left`` wire across the chunk to the
    cell's last input, then the cell itself, whose output order lays the
    next frontier chunk and restores the carried triple with no second
    shuffle.
    """
    graph = type(cell).ar
    a, b, value = (cell.dom[i:i + 1] for i in range(3))
    if cell.dom != a @ b @ value @ value @ value \
            or cell.cod != b @ value @ value @ a @ value:
        raise ValueError(
            f"a cell reads (a, b, diag, up, left) and writes "
            f"(b, down, diag, a, right), got {cell.dom} -> {cell.cod}")
    row, chunk = value @ a @ value, value @ b @ value
    diagram = graph.id(row ** m @ chunk ** n)
    for pending in reversed(range(m)):
        for j in range(n):
            wires, position = diagram.cod, 3 * (pending + j)
            crossing = graph.permutation(
                [2, 1, 3, 0], wires[position + 2:position + 6])
            diagram >>= wires[:position + 2] @ crossing \
                @ wires[position + 6:]
            wires = diagram.cod
            diagram >>= wires[:position + 1] @ cell @ wires[position + 6:]
    return diagram


def answer(n: int) -> int:
    """ The index of ``L[m][n]`` in the codomain of :func:`lcs`. """
    return 3 * n + 2
