# -*- coding: utf-8 -*-

"""
What the library can do, independently of any task: non-uniform degree,
one batched call per group of boxes sharing a module -- pinned against a
one-call-per-box oracle written out here -- and heterogeneous batching
through :class:`~discopy.neural.MapNN`.
"""

import pytest

torch = pytest.importorskip("torch")

from discopy.frobenius import Ty
from discopy.neural import (
    Batch, Dim, MapNN, Orbit, Signature, Sym, box_ports, bucket,
    from_incidence, from_relation, interpret)


MESSAGE, PEER = Ty("message"), Ty("peer")
STATE, CLUE = Ty("state"), Ty("clue")


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class Mix(torch.nn.Module):
    """
    Mixes every port with every other, whatever the width of the box: an
    all-port module that is not elementwise, so that a wrong gather or
    scatter would show.
    """
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.5, dtype=torch.double))
        self.bias = torch.nn.Parameter(torch.tensor(.1, dtype=torch.double))

    def forward(self, x):
        return torch.tanh(x + self.weight * x.flip(-1) + self.bias)


def node_signature(degree: int = 1, role=PEER) -> Signature:
    return Signature((
        Orbit(role, degree, Sym.PERM), Orbit(STATE, traced=True),
        Orbit(CLUE, traced=True)))


OB = {PEER: Dim(3), MESSAGE: Dim(3), STATE: Dim(4), CLUE: Dim(2)}


def oracle(cmap, state, rounds):
    """ One call per box per round, port by port. """
    widths = cmap.port_widths
    offsets = [sum(widths[:i]) for i in range(len(widths))]
    for _ in range(rounds):
        outgoing = torch.zeros_like(state)
        for index, box in enumerate(cmap.boxes):
            ports = box_ports(cmap, index)
            value = box.module(torch.cat(
                [state[:, offsets[p]:offsets[p] + widths[p]] for p in ports],
                -1))
            for port, chunk in zip(ports, torch.split(
                    value, [widths[p] for p in ports], -1)):
                outgoing[:, offsets[port]:offsets[port] + widths[port]] = chunk
        state = torch.cat([
            outgoing[:, offsets[cmap.edges[p]]:
                     offsets[cmap.edges[p]] + widths[p]]
            for p in range(len(widths))], -1)
    return state


def test_nonuniform_relation_forward_matches_oracle():
    """
    A path graph -- degrees 1, 2, 1 -- runs, one call per group of boxes
    of the same degree, and agrees with the one-call-per-box oracle.
    """
    path = from_relation(((1, ), (0, 2), (1, )), node_signature(1))
    assert [len(box.cod) for box in path.boxes] == [5, 6, 5]
    cmap = interpret(path, OB, {"cell": Mix()})
    assert len(cmap.routing["groups"]) == 2
    torch.manual_seed(1)
    init = torch.randn(2, sum(cmap.port_widths), dtype=torch.double)
    with torch.no_grad():
        fast = cmap(init=init, n_rounds=3, inject=False, return_flat=True)
    assert torch.allclose(fast, oracle(cmap, init, 3), atol=1e-12)


def test_nonuniform_incidence_forward_matches_oracle():
    """
    Mixed node degrees and mixed relation sizes through ``from_incidence``,
    two modules, against the oracle.
    """
    node = node_signature(1, MESSAGE)
    unit = Signature((Orbit(MESSAGE, 2, Sym.PERM), ))
    shape = from_incidence(((0, 1), (0, 1), (1, )), node, unit)
    assert [len(box.cod) for box in shape.boxes[3:]] == [2, 3]
    cmap = interpret(shape, OB, {"cell": Mix(), "unit": Mix()})
    assert len(cmap.routing["groups"]) == 4
    torch.manual_seed(1)
    init = torch.randn(2, sum(cmap.port_widths), dtype=torch.double)
    with torch.no_grad():
        fast = cmap(init=init, n_rounds=3, inject=False, return_flat=True)
    assert torch.allclose(fast, oracle(cmap, init, 3), atol=1e-12)


def small_model(rounds: int = 3, **kwargs) -> MapNN:
    torch.manual_seed(0)
    return MapNN(OB, {"cell": Mix()}, rounds=rounds, **kwargs).double()


def test_a_graph_level_readout_is_a_generator():
    """
    One extra relation wired to every node, under its own name, gives a
    graph-level readout with no change to the model: its per-leg emissions
    are readable as an ordinary port family.
    """
    node = node_signature(1, MESSAGE)
    unit = Signature((Orbit(MESSAGE, 2, Sym.PERM), ))
    shape = from_incidence(((0, 1), (0, 1), (1, )), node, unit,
                           relation_name=("unit", "readout"))
    assert [box.name for box in shape.boxes] \
        == ["cell", "cell", "cell", "unit", "readout"]
    model = MapNN(OB, {"cell": Mix(), "unit": Mix(), "readout": Mix()},
                  rounds=2).double()
    assert model.sites(shape, ("readout", MESSAGE)) == 3
    with torch.no_grad():
        state = model(shape)
    assert model.read(shape, state, ("readout", MESSAGE)).shape == (1, 3, 3)


def test_one_model_two_shapes_one_set_of_weights():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    model = small_model()
    before = [(name, id(value)) for name, value in model.named_parameters()]
    with torch.no_grad():
        one, other = model(pair), model(path)
    assert one.shape == (1, sum(model.compile(pair)[0].port_widths))
    assert other.shape == (1, sum(model.compile(path)[0].port_widths))
    assert [(name, id(value))
            for name, value in model.named_parameters()] == before
    assert model.compile(pair)[0] is model.compile(pair)[0]
    assert model.cache_stats()["misses"] == 2


def test_the_compilation_cache_is_bounded():
    node = node_signature(1)
    model = small_model(cache=2)
    shapes = [from_relation(((1, ), (0, )), node) for _ in range(4)]
    for shape in shapes:
        model.compile(shape)
    assert model.cache_stats(reset=True) == {
        "hits": 0, "misses": 4, "held": 2, "capacity": 2}
    assert model.cache_stats()["misses"] == 0


def test_deep_supervises_every_round():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    model = small_model(rounds=4)
    cmap, _, _ = model.compile(pair)
    torch.manual_seed(1)
    state = torch.randn(2, sum(cmap.port_widths), dtype=torch.double)
    with torch.no_grad():
        every = model(pair, state, deep=True)
        last = model(pair, state)
        two = model(pair, state, rounds=2)
    assert len(every) == 4 and torch.equal(last, every[-1])
    assert torch.equal(two, every[1])
    assert torch.equal(every[0], model(pair, state, rounds=1))


def test_writing_a_family_writes_every_copy_of_its_trace():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    model = small_model()
    values = torch.arange(2 * 2 * 4, dtype=torch.double).reshape(2, 2, 4)
    state = model.initial(pair, {("cell", STATE): values})
    assert torch.equal(model.read(pair, state, ("cell", STATE)), values)
    every = model.read(pair, state, ("cell", STATE), every=True)
    assert every.shape == (2, 4, 4)
    assert torch.equal(every[:, 0::2], values)
    assert torch.equal(every[:, 1::2], values)
    assert model.read(pair, state, ("cell", CLUE)).abs().sum() == 0
    assert model.initial(pair, rows=3).shape == (3, state.shape[1])


def test_writing_a_family_with_several_legs():
    """
    A traced orbit with two legs lays out both heads before both tails, so
    a value written on a head lands on the tail its wire loops back to,
    not on the port next to it.
    """
    node = Signature((
        Orbit(PEER, 1, Sym.PERM), Orbit(STATE, 2, traced=True),
        Orbit(CLUE, traced=True)))
    pair = from_relation(((1, ), (0, )), node)
    model = small_model()
    cmap, ports, heads = model.compile(pair)
    assert len(heads["cell", STATE]) == 4 and len(ports["cell", STATE]) == 8
    values = torch.arange(4 * 4, dtype=torch.double).reshape(1, 4, 4)
    state = model.initial(pair, {("cell", STATE): values})
    assert torch.equal(model.read(pair, state, ("cell", STATE)), values)
    for head in heads["cell", STATE]:
        assert torch.equal(cmap.read(state, (head, )),
                           cmap.read(state, (cmap.edges[head], )))


def test_a_batch_is_the_product_of_its_members():
    """
    Running ``[a, b]`` gives, member for member, what running ``a`` and
    ``b`` alone gives: the monoidal product through the whole model.
    """
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
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
        sum(model.compile(shape)[0].port_widths) for shape in (pair, path))
    for expected, found in zip(alone, pieces):
        assert torch.allclose(expected, found, atol=1e-13)
    # one module call covers every site of the same degree, not one per
    # member: here the degrees are 1 and 2, hence two groups
    assert len(model.compile(batch)[0].routing["groups"]) == 2


def test_a_batch_drops_its_padding():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
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
    with pytest.raises(ValueError, match="at least one"):
        Batch([])


def test_a_batch_state_splits_back():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    model = small_model()
    batch = Batch([pair, path])
    torch.manual_seed(1)
    states = [torch.randn(2, sum(model.compile(shape)[0].port_widths),
                          dtype=torch.double) for shape in (pair, path)]
    for expected, found in zip(
            states, batch.split_state(batch.join(states), model.ob)):
        assert torch.equal(expected, found)
    with pytest.raises(ValueError, match="expected 2"):
        batch.join(states[:1])
