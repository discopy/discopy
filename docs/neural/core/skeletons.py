# -*- coding: utf-8 -*-

"""
The abstract wirings of the solver family: pure syntax, no torch, no
widths, no task.

A skeleton is a closed :class:`discopy.frobenius.CMap` whose boxes are
plain :class:`discopy.frobenius.Box` -- no modules inside -- and whose
atomic types are the *roles* a port can play rather than the width it will
carry:

* :data:`MESSAGE` : a variable-to-unit wire of a bipartite factor graph,
* :data:`PEER` : a variable-to-variable wire of a pairwise clique,
* :data:`STATE`, :data:`CLUE`, :data:`ANSWER` : the traced loops, each a
  self-wired pair of ports on a cell carrying its recurrent state, its
  input and -- for a recursion solver -- its current answer between rounds.

Frobenius types are self-dual, which is what lets a wire connect two
codomain ports of the same role; and ``import discopy`` does not import
``torch``, so a skeleton can be built and checked -- degrees, involution,
loop positions -- on a machine with no torch at all.  What fills the nodes
is decided later, by a :class:`discopy.neural.Functor`: see
:mod:`core.functors`.  Typing each role as a distinct atomic object is
what gives the functor's object map one knob per width -- and, since
``Dim(0)`` is the monoidal unit, sending a role to ``Dim(0)`` erases its
ports altogether, which is how one skeleton can serve solvers with and
without an answer loop.

The two shapes are parameterized by their combinatorics alone: a task
supplies who belongs to which constraint unit (:func:`factor_graph`) or
who is whose peer (:func:`clique`), and gets back the closed map.
"""

from __future__ import annotations

from discopy.frobenius import Box, CMap, Ty

#: The roles a port can play, as self-dual atomic types.
MESSAGE, PEER = Ty("message"), Ty("peer")
STATE, CLUE, ANSWER = Ty("state"), Ty("clue"), Ty("answer")

#: The roles that live on a traced loop, in the order they appear on a cell.
LOOP_ROLES = (STATE, CLUE, ANSWER)


def add_loops(wires: list, cell: int, cod: Ty) -> None:
    """
    Wire the consecutive pair of ports of each loop role of a box to
    itself, appending to ``wires``.

    Parameters:
        wires : The wiring under construction.
        cell : The index of the box.
        cod : The codomain of the box, holding the loop-role pairs.
    """
    atoms = list(cod)
    for role in LOOP_ROLES:
        positions = [i for i, atom in enumerate(atoms) if atom == role]
        if positions:
            assert len(positions) == 2, f"{role} is not a pair of ports"
            wires.append(((cell, positions[0]), (cell, positions[1])))


def factor_graph(memberships: tuple) -> CMap:
    """
    The bipartite factor graph over ``len(memberships)`` variables: one
    cell box per variable with one :data:`MESSAGE` port per unit it
    belongs to and all three loops, one unit box per constraint unit with
    one :data:`MESSAGE` port per member, and a wire from each variable to
    each of its units.

    Every variable must belong to the same number of units and every unit
    must have the same number of members, so that one shared cell box and
    one shared unit box fill every site.

    Parameters:
        memberships : Per variable, the tuple of indices of the units it
                      belongs to; units are numbered ``0`` to the number
                      of unit boxes minus one.
    """
    n_vars = len(memberships)
    degree = len(memberships[0])
    assert all(len(units) == degree for units in memberships), \
        "variables belong to different numbers of units"
    n_units = 1 + max(max(units) for units in memberships)
    size = [0] * n_units
    for units in memberships:
        for unit_index in units:
            size[unit_index] += 1
    assert len(set(size)) == 1, "units have different numbers of members"

    cell = Box(
        "cell", Ty(), MESSAGE ** degree @ STATE ** 2 @ CLUE ** 2
        @ ANSWER ** 2)
    unit = Box("unit", Ty(), MESSAGE ** size[0])

    free = [0] * n_units
    wires: list = []
    for index in range(n_vars):
        for position, unit_index in enumerate(memberships[index]):
            wires.append(
                ((index, position), (n_vars + unit_index, free[unit_index])))
            free[unit_index] += 1
        add_loops(wires, index, cell.cod)
    assert all(slot == size[0] for slot in free), "a unit box is not full"
    return CMap.from_wiring((cell, ) * n_vars + (unit, ) * n_units, wires)


def clique(peers: tuple) -> CMap:
    """
    The pairwise clique over ``len(peers)`` variables: one cell box per
    variable with one :data:`PEER` port per peer plus state and clue
    loops, and a wire between each pair of peers.  No unit boxes and no
    answer loop.

    The peer relation must be symmetric and every variable must have the
    same number of peers, so that one shared cell box fills every site.

    Parameters:
        peers : Per variable, the tuple of indices of its peers.
    """
    n_vars = len(peers)
    n_peers = len(peers[0])
    assert all(len(others) == n_peers for others in peers), \
        "variables have different numbers of peers"
    cell = Box("cell", Ty(), PEER ** n_peers @ STATE ** 2 @ CLUE ** 2)

    wires: list = []
    for index in range(n_vars):
        for other in peers[index]:
            assert index in peers[other], "the peer relation is asymmetric"
            if index < other:
                wires.append(((index, peers[index].index(other)),
                              (other, peers[other].index(index))))
        add_loops(wires, index, cell.cod)
    return CMap.from_wiring((cell, ) * n_vars, wires)
