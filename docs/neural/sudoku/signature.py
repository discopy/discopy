# -*- coding: utf-8 -*-

"""
The roles a port of a sudoku solver can play, and the signature of each
box.

A role is an atomic type naming *what a wire carries*, not how wide it is;
the width is chosen later, by an interpretation.  Six roles cover the
three models:

* :data:`MESSAGE` -- a cell-to-unit wire of the bipartite factor graph,
* :data:`PEER` -- a cell-to-cell wire of the pairwise clique,
* :data:`STATE` -- a cell's recurrent state, on a traced loop,
* :data:`HIDDEN`, :data:`MEMORY` -- the two states of an ``LSTMCell``, two
  named roles on *one* traced loop rather than two halves of a wide port,
* :data:`CLUE` -- the clue embedding, on a traced loop,
* :data:`ANSWER` -- the current guess of a recursion solver, on a traced
  loop.  Sending it to ``Dim(0)`` erases the loop, which is how one
  skeleton serves the model with an answer and the model without.

The signatures say how many ports of each role a box has and which of them
are one orbit under a group: the members of a constraint unit are a
:attr:`Sym.PERM` orbit, and so are the units a cell belongs to and the
peers of a cell, since a sudoku constraint does not care which of its
members is which.
"""

from __future__ import annotations

from discopy.frobenius import Ty
from discopy.neural import Dim, Orbit, Signature, Sym

#: The roles of the factor-graph models.
MESSAGE = Ty("message")
STATE, CLUE, ANSWER = Ty("state"), Ty("clue"), Ty("answer")

#: The roles of the clique model: a peer wire carries a full hidden state,
#: and the loop carries the two states of an ``LSTMCell``.
PEER, HIDDEN, MEMORY = Ty("peer"), Ty("hidden"), Ty("memory")


def cell(degree: int = 3) -> Signature:
    """
    A cell of the factor graph: one message port per unit it belongs to,
    plus a state, a clue and an answer loop.

    Parameters:
        degree : The number of units a cell belongs to.

    Example
    -------
    >>> print(cell().cod)
    message @ message @ message @ state @ state @ clue @ clue @ answer @ \
answer
    """
    return Signature((
        Orbit(MESSAGE, degree, Sym.PERM), Orbit(STATE, traced=True),
        Orbit(CLUE, traced=True), Orbit(ANSWER, traced=True)))


def unit(size: int = 9) -> Signature:
    """
    A constraint unit: one port per member, permutation-symmetric.

    Parameters:
        size : The number of cells in a unit.
    """
    return Signature((Orbit(MESSAGE, size, Sym.PERM), ))


def peer_cell(peers: int = 20) -> Signature:
    """
    A cell of the clique: one port per peer, plus one traced loop carrying
    both states of an ``LSTMCell`` and one carrying the clue.

    The two states share a loop rather than a wide port, so the cell reads
    them by name; laid out this way they occupy exactly the bytes the one
    wide port did, and the routing permutation is unchanged.

    Parameters:
        peers : The number of peers of a cell.

    Example
    -------
    >>> print(peer_cell(2).cod)
    peer @ peer @ hidden @ memory @ hidden @ memory @ clue @ clue
    >>> peer_cell(2).loops()
    ((2, 4), (3, 5), (6, 7))
    """
    return Signature((
        Orbit(PEER, peers, Sym.PERM), Orbit(HIDDEN @ MEMORY, traced=True),
        Orbit(CLUE, traced=True)))


def factor_widths(widths, answer_dim: int = 0) -> dict:
    """
    The width each role of the factor graph carries.

    Parameters:
        widths : The :class:`core.study.Widths` of the model.
        answer_dim : The width of the answer loop, ``0`` to erase it.
    """
    return {MESSAGE: Dim(widths.dim), STATE: Dim(widths.state_dim),
            CLUE: Dim(widths.dim), ANSWER: Dim(answer_dim)}


def clique_widths(widths) -> dict:
    """
    The width each role of the clique carries: a peer wire carries a full
    hidden state, and so does each of the two states on the loop.

    Parameters:
        widths : The :class:`core.study.Widths` of the model.
    """
    return {PEER: Dim(widths.state_dim), HIDDEN: Dim(widths.state_dim),
            MEMORY: Dim(widths.state_dim), CLUE: Dim(widths.dim)}
