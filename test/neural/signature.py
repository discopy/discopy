# -*- coding: utf-8 -*-

"""
The port layout of one generator, and the wirings drawn out of a family's
combinatorics: ``Signature.slices`` reproduces the hand-written cursor
arithmetic of the cells it replaced, the builders draw frobenius maps whose
loops are traces, and every layout question has one answer.
"""

from pytest import raises

from discopy import frobenius
from discopy.frobenius import CMap, Ty
from discopy.neural import (
    Orbit, Signature, Sym, from_incidence, from_relation)
from discopy.neural.signature import leg_generators


MESSAGE, PEER, STATE = Ty("message"), Ty("peer"), Ty("state")
HIDDEN, MEMORY = Ty("hidden"), Ty("memory")
CLUE, ANSWER = Ty("clue"), Ty("answer")


def cell(degree: int = 3) -> Signature:
    """
    A cell of a factor graph: one message port per unit it belongs to, plus
    a state, a clue and an answer loop.
    """
    return Signature((
        Orbit(MESSAGE, degree, Sym.PERM), Orbit(STATE, traced=True),
        Orbit(CLUE, traced=True), Orbit(ANSWER, traced=True)))


def peer_cell(peers: int = 20) -> Signature:
    """
    A cell of a clique: one port per peer, plus one traced loop carrying
    both states of an ``LSTMCell`` and one carrying the clue.
    """
    return Signature((
        Orbit(PEER, peers, Sym.PERM), Orbit(HIDDEN @ MEMORY, traced=True),
        Orbit(CLUE, traced=True)))


def test_slices_match_the_old_cursor():
    """
    ``Signature.slices`` reproduces the hand-written cursor arithmetic of
    the cells it replaced, asserted on the actual old offsets.  The factor
    graph cell read ``cursor = n_message * dim``, its state at ``cursor``,
    then ``cursor += 2 * state_dim``, its clue there, ``cursor += 2 * dim``
    and its answer there.  The clique cell read ``cursor = n_peers *
    state_dim``, its hidden state at ``cursor`` and its memory at ``cursor
    + state_dim``, then ``cursor += 4 * state_dim`` and its clue there.
    """
    dim, state_dim, y_dim = 24, 96, 48
    places = cell(3).slices(
        {MESSAGE: dim, STATE: state_dim, CLUE: dim, ANSWER: y_dim})
    cursor = 3 * dim
    assert places[MESSAGE] == slice(0, cursor)
    assert places[STATE] == slice(cursor, cursor + state_dim)
    cursor += 2 * state_dim
    assert places[CLUE] == slice(cursor, cursor + dim)
    cursor += 2 * dim
    assert places[ANSWER] == slice(cursor, cursor + y_dim)

    peers = 20
    places = peer_cell(peers).slices(
        {PEER: state_dim, HIDDEN: state_dim, MEMORY: state_dim, CLUE: dim})
    cursor = peers * state_dim
    assert places[PEER] == slice(0, cursor)
    assert places[HIDDEN] == slice(cursor, cursor + state_dim)
    assert places[MEMORY] == slice(
        cursor + state_dim, cursor + 2 * state_dim)
    cursor += 4 * state_dim
    assert places[CLUE] == slice(cursor, cursor + dim)


def test_both_builders_draw_frobenius_maps():
    """
    A triangle from a relation and a square with two units from an
    incidence are closed maps in ``frobenius``, the one category where a
    wire between two codomain ports type checks, with every port wired
    exactly once.
    """
    node = Signature((Orbit(PEER, 2, Sym.PERM), Orbit(STATE, traced=True)))
    triangle = from_relation(((1, 2), (0, 2), (0, 1)), node)
    assert [len(box.cod) for box in triangle.boxes] == [4, 4, 4]
    unit = Signature((Orbit(PEER, 3, Sym.PERM), ))
    square = from_incidence(((0, 1), ) * 4, node, unit)
    assert [len(box.cod) for box in square.boxes] == [4] * 6
    assert [box.name for box in square.boxes] == ["cell"] * 4 + ["unit"] * 2
    for shape, n_ports in ((triangle, 12), (square, 24)):
        assert isinstance(shape, frobenius.CMap)
        assert shape.n_ports == n_ports
        assert not len(shape.dom) and not len(shape.cod)
        assert shape.edges.is_fixpoint_free_involution()
    assert isinstance(unit.box("unit"), frobenius.Box)
    with raises(TypeError):
        unit.box("unit", category=frobenius)


def test_the_relation_must_be_symmetric():
    node = Signature((Orbit(PEER, 1), Orbit(STATE, traced=True)))
    with raises(ValueError, match="not symmetric"):
        from_relation(((1, ), (0, ), (0, )), node)
    with raises(ValueError, match="left unwired"):
        from_relation(((0, ), ), node)


def test_incidence_names_and_signatures():
    """
    Relations may carry one signature each, under one name each; the names
    must be one per relation; and a relation nobody belongs to is a box
    with no ports, a scalar of the map.
    """
    node = Signature((Orbit(MESSAGE, 2, Sym.PERM), Orbit(STATE, traced=True)))
    unit = Signature((Orbit(MESSAGE, 3, Sym.PERM), ))
    readout = Signature((Orbit(MESSAGE, 1), Orbit(CLUE, traced=True)))
    shape = from_incidence(
        ((0, 1), (0, 1)), node, {"unit": unit, "readout": readout},
        relation_name=("unit", "readout"))
    assert [(box.name, len(box.cod)) for box in shape.boxes] \
        == [("cell", 4), ("cell", 4), ("unit", 2), ("readout", 4)]
    with raises(ValueError, match="2 names for 1 relations"):
        from_incidence(((0, ), ), node, unit, relation_name=("a", "b"))
    memberless = from_incidence(((1, ), ), node, unit)
    assert [len(box.cod) for box in memberless.boxes] == [3, 0, 1]


def test_resize_keeps_a_composite_leg_single():
    clique = peer_cell(2)
    assert clique.resize(PEER, 5).orbits[0].arity == 5
    assert clique.resize(HIDDEN, 1) == clique
    with raises(ValueError, match="several roles"):
        clique.resize(HIDDEN, 2)
    with raises(ValueError):
        Orbit(PEER, -1)


def test_slices_with_an_erased_and_a_repeated_role():
    """
    A role of width zero has no slice, as ``Dim(0)`` has no port, and a
    role in two orbits has no single slice: it raises, erased or not.
    """
    widths = {MESSAGE: 4, STATE: 8, CLUE: 4, ANSWER: 0}
    places = cell(3).slices(widths)
    assert ANSWER not in places and places[CLUE] == slice(28, 32)
    assert cell(3).width(widths) == 36 == places[CLUE].stop + 4
    twice = Signature((Orbit(STATE, traced=True), Orbit(STATE, traced=True)))
    assert twice.positions(STATE) == (0, 1, 2, 3)
    for width in (4, 0):
        with raises(ValueError, match="two orbits"):
            twice.slices({STATE: width})


def test_generators_act_on_legs_and_on_every_copy():
    """
    A leg permutation acts on every atom of a leg and on every copy of the
    orbit alike, the identity on the other orbits: a cyclic orbit has one
    generator, a permutation-symmetric one two, a single leg none.
    """
    cyclic = Signature((Orbit(PEER, 3, Sym.CYCLIC), Orbit(STATE, traced=True)))
    assert [tuple(p.inside) for p in cyclic.generators()] == [(1, 2, 0, 3, 4)]
    traced = Signature((
        Orbit(CLUE, 1), Orbit(STATE, 3, Sym.PERM, traced=True),
        Orbit(PEER, 1)))
    assert [tuple(p.inside) for p in traced.generators()] \
        == [(0, 2, 1, 3, 5, 4, 6, 7), (0, 2, 3, 1, 5, 6, 4, 7)]
    assert leg_generators(Sym.CYCLIC, 2) == [(1, 0)]
    assert cell(1).generators() == [] == leg_generators(Sym.PERM, 0)


def test_repr_round_trips():
    orbit = Orbit(PEER, 3, Sym.PERM, traced=True)
    scope = {"Orbit": Orbit, "Signature": Signature, "Sym": Sym,
             "frobenius": frobenius}
    assert eval(repr(orbit), scope) == orbit
    assert eval(repr(cell(2)), scope) == cell(2)
    assert eval(repr(Sym.CYCLIC), scope) is Sym.CYCLIC
    assert repr(Sym.NONE) == "Sym.NONE" and str(Sym.NONE) == "none"


def test_loop_wires_are_the_traces_of_a_box():
    node = Signature((Orbit(PEER, 1), Orbit(STATE, traced=True)))
    assert node.loop_wires(2) == [((2, 1), (2, 2))]
    assert peer_cell(2).loop_wires(0) \
        == [((0, 2), (0, 4)), ((0, 3), (0, 5)), ((0, 6), (0, 7))]
    box = node.box("cell")
    assert from_relation(((1, ), (0, )), node) == CMap.from_wiring(
        (box, box),
        [((0, 0), (1, 0))] + node.loop_wires(0) + node.loop_wires(1))
