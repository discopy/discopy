# -*- coding: utf-8 -*-

"""
Tests for :mod:`discopy.quantum.gates`.

Every box that narrows :code:`Box.__init__` needs a :code:`from_tree` of its
own, and where the tree does not hold what its constructor reads, a
:code:`to_tree` to put it there (#780).
"""

from pytest import mark

from discopy.quantum.gates import (
    CCX, CCZ, CRx, CRz, CU1, CX, CY, CZ, Bits, Bra, Controlled, Copy, Digits,
    Discard, Encode, H, Ket, Match, Measure, MixedScalar, MixedState, Rx, Ry,
    Rz, S, SWAP, Scalar, Sqrt, T, U1, X, Y, Z)
from discopy.utils import from_tree


BOXES = [
    CCX, CCZ, CX, CY, CZ, H, S, SWAP, T, X, Y, Z,
    Rx(.25), Ry(.25), Rz(.25), U1(.25),
    CRx(.25), CRz(.25), CU1(.25), CRz(.25, distance=3),
    Controlled(X, distance=-2), Controlled(CX, distance=2),
    Scalar(.5), Scalar(.5, is_mixed=True), MixedScalar(.5), Sqrt(2),
    Ket(1, 0), Bra(1, 0), Ket(0).dagger(), Copy(), Match(),
    Bits(1, 0), Bits(1, 0).dagger(), Digits(2, dim=4),
    Discard(), Discard(2), MixedState(), MixedState(2),
    Measure(), Measure(2, destructive=False), Measure(2, override_bits=True),
    Encode(), Encode(2, constructive=False), Encode(2, reset_bits=True),
]


@mark.parametrize("box", BOXES, ids=repr)
def test_to_from_tree(box):
    assert from_tree(box.to_tree()) == box


def test_to_from_tree_circuit():
    circuit = Ket(0, 0) >> CX >> Rz(.25) @ H >> Scalar(.5) @ Bra(0, 0)
    assert from_tree(circuit.to_tree()) == circuit


def test_Controlled_from_tree_reads_back_through_the_base():
    """
    A subclass such as :code:`CRz` takes the phase of the rotation it controls
    where :code:`Controlled` takes the gate itself, and the two are equal, so
    the tree of either reads back as a :code:`Controlled`.
    """
    assert CRz(.25) == Controlled(Rz(.25), distance=1)
    assert from_tree(CRz(.25).to_tree()) == CRz(.25)
    assert isinstance(from_tree(CRz(.25).to_tree()), Controlled)


def test_Measure_Encode_trees_carry_their_flags():
    assert Measure(2, destructive=False).to_tree()["destructive"] is False
    assert Measure(2, override_bits=True).to_tree()["override_bits"] is True
    assert Encode(2, constructive=False).to_tree()["constructive"] is False
    assert Encode(2, reset_bits=True).to_tree()["reset_bits"] is True


def test_Scalar_tree_carries_is_mixed():
    assert Scalar(.5, is_mixed=True).to_tree()["is_mixed"] is True
    assert from_tree(Scalar(.5, is_mixed=True).to_tree()).is_mixed is True
    assert from_tree(Scalar(.5).to_tree()).is_mixed is False
