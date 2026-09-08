# -*- coding: utf-8 -*-

"""
Batching over heterogeneous diagrams is the monoidal product of their maps:
running a batch gives, member for member, what running each member alone
gives, the flat state of the product is the concatenation of the members'
states, and a model keys its cache on the members so that a fresh batch of
interned diagrams costs no interpretation.
"""

from pytest import fixture, importorskip, raises

from discopy import cmap, frobenius
from discopy.frobenius import Ty
from discopy.neural import (
    Batch, Dim, Orbit, Signature, Sym, bucket, from_relation)

torch = importorskip("torch")
MapNN = importorskip("discopy.neural.model").MapNN


PEER, STATE, CLUE = Ty("peer"), Ty("state"), Ty("clue")

NODE = Signature((
    Orbit(PEER, 1, Sym.PERM), Orbit(STATE, traced=True),
    Orbit(CLUE, traced=True)))

OB = {PEER: Dim(3), STATE: Dim(4), CLUE: Dim(2)}


@fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class Mix(torch.nn.Module):
    """ Mixes every port with every other, whatever the width of the box. """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.5, dtype=torch.double))

    def forward(self, x):
        return torch.tanh(x + self.weight * x.flip(-1))


def shapes():
    """ A pair and a path, degrees ``(1, 1)`` and ``(1, 2, 1)``. """
    return (from_relation(((1, ), (0, )), NODE),
            from_relation(((1, ), (0, 2), (1, )), NODE))


def small_model(**kwargs) -> MapNN:
    torch.manual_seed(0)
    return MapNN(OB, {"cell": Mix()}, rounds=3, **kwargs).double()


def test_a_batch_is_the_product_of_its_members():
    """
    Running ``[a, b]`` gives, member for member, what running ``a`` and
    ``b`` alone gives: the monoidal product through the whole model.  One
    module call covers every site of the same degree rather than one per
    member: the degrees here are one and two, hence two groups.
    """
    pair, path = shapes()
    model = small_model()
    torch.manual_seed(1)
    x_pair = torch.randn(2, 2, 2, dtype=torch.double)
    x_path = torch.randn(2, 3, 2, dtype=torch.double)
    with torch.no_grad():
        alone = [model.read(shape, model(shape, {("cell", CLUE): x}),
                            ("cell", STATE))
                 for shape, x in ((pair, x_pair), (path, x_path))]
        batch = Batch([pair, path])
        state = model(batch, {("cell", CLUE): torch.cat([x_pair, x_path], 1)})
        pieces = batch.split(
            model.read(batch, state, ("cell", STATE)), ("cell", STATE))
    assert batch.sizes(("cell", STATE)) == (2, 3)
    assert batch.widths(model.ob) == tuple(
        sum(model.interpret(shape).cmap.port_widths) for shape in (pair, path))
    assert len(pieces) == 2
    for expected, found in zip(alone, pieces):
        assert torch.allclose(expected, found, atol=1e-13)
    assert len(model.interpret(batch).cmap.routing["groups"]) == 2


def test_a_batch_runs_its_members_exactly():
    """
    A batch of two shapes, padded or not, gives every member's state
    exactly: the flat state of the product is the concatenation of the
    members', so ``join`` and ``split_state`` are inverse, and running the
    joined state through the batch is running each member alone.
    """
    pair, path = shapes()
    members = (pair, path, pair)
    model = small_model()
    torch.manual_seed(1)
    states = [torch.randn(2, sum(model.interpret(shape).cmap.port_widths),
                          dtype=torch.double) for shape in members]
    for pad in (False, True):
        batch = Batch(members, pad=pad)
        joined = batch.join(states)
        assert joined.shape == (2, sum(batch.widths(OB, padded=True)))
        assert all(torch.equal(expected, found) for expected, found
                   in zip(states, batch.split_state(joined, OB)))
        with torch.no_grad():
            alone = [model(shape, state)
                     for shape, state in zip(members, states)]
            together = batch.split_state(model(batch, joined), OB)
        assert len(together) == 3
        assert all(torch.allclose(expected, found, atol=1e-12)
                   for expected, found in zip(alone, together))
    with raises(ValueError, match="expected 3"):
        batch.join(states[:1])


def test_a_batch_drops_its_padding():
    pair, path = shapes()
    batch = Batch([pair, path, pair], pad=True)
    assert len(batch.parts) == 4 and batch.given == 3
    model = small_model()
    x = torch.zeros(1, 2 + 3 + 2 + 2, 2, dtype=torch.double)
    with torch.no_grad():
        state = model(batch, {("cell", CLUE): x})
        pieces = batch.split(
            model.read(batch, state, ("cell", STATE)), ("cell", STATE))
    assert [piece.shape[1] for piece in pieces] == [2, 3, 2]
    assert [bucket(n) for n in (1, 3, 5, 9, 2000)] == [1, 4, 8, 16, 2000]
    with raises(ValueError, match="at least one"):
        Batch([])


def test_a_fresh_batch_of_interned_parts_hits_the_cache():
    """
    The cache key of a batch is the identity of its members, so a batch
    built anew from the same interned diagrams is a hit, and one built from
    a fresh diagram of the same shape is a miss.
    """
    pair, path = shapes()
    model = small_model()
    first = model.interpret(Batch([pair, path]))
    assert model.interpret(Batch([pair, path])) is first
    assert model.cache_stats() == {
        "hits": 1, "misses": 1, "held": 1, "capacity": 128}
    assert Batch([pair, path]).cache_key() == ("batch", id(pair), id(path))
    assert Batch([pair, path, pair], pad=True).cache_key() \
        == ("batch", id(pair), id(path), id(pair), id(pair))
    model.interpret(Batch([shapes()[0], path]))
    assert model.cache_stats()["misses"] == 2


def test_widths_read_integers_and_dims_alike():
    """
    A width is read through the functor of the interpretation, so an
    integer-valued ``ob`` and a ``Dim``-valued one give the same widths,
    and they are the widths of the compiled members.
    """
    pair, path = shapes()
    batch = Batch([pair, path])
    model = small_model()
    assert batch.widths({PEER: 3, STATE: 4, CLUE: 2}) == batch.widths(OB) \
        == tuple(sum(model.interpret(shape).cmap.port_widths)
                 for shape in (pair, path)) == (30, 48)
    assert batch.widths({PEER: 3, STATE: 4, CLUE: 0}) == (22, 36)


def test_len_repr_and_sizes():
    pair, path = shapes()
    batch = Batch([pair, path, pair], pad=True)
    assert len(batch) == 3 and len(batch.parts) == 4 and batch.pad
    assert repr(batch) == f"Batch({(pair, path, pair)!r}, pad=True)"
    scope = {"Batch": Batch, "cmap": cmap, "frobenius": frobenius}
    assert eval(repr(batch), scope) == batch
    assert eval(repr(batch), scope) != Batch([pair, path, pair])
    assert batch.sizes(("cell", STATE)) == (2, 3, 2)
    assert batch.sizes(("cell", STATE), padded=True) == (2, 3, 2, 2)
    assert batch.sizes(("unit", STATE)) == (0, 0, 0)
    assert batch.diagram is batch.diagram and len(batch.diagram.boxes) == 9
