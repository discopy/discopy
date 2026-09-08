# -*- coding: utf-8 -*-

"""
What the library can do, independently of any task: non-uniform degree,
one batched call per group of boxes sharing a module -- pinned against a
one-call-per-box oracle written out here -- training through
:class:`~discopy.neural.MapNN`, and one set of weights serving every
degree once the cell reads its layout off its signature.
"""

from pytest import fixture, importorskip, raises

from discopy.frobenius import Ty
from discopy.neural import (
    Dim, Orbit, Signature, Sym, from_incidence, from_relation, interpret)

torch = importorskip("torch")
neural_model = importorskip("discopy.neural.model")
MapNN, Interpretation = neural_model.MapNN, neural_model.Interpretation


MESSAGE, PEER = Ty("message"), Ty("peer")
STATE, CLUE = Ty("state"), Ty("clue")


@fixture(autouse=True)
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

WIDTHS = {PEER: 3, STATE: 4, CLUE: 2}


class Pooled(torch.nn.Module):
    """
    A cell built on ``Linear`` layers that serves every degree: it reads
    its degree off the width it is handed and its layout off the signature,
    mean-pools the peer block so that its input scale does not depend on
    the degree, and writes the next state on the incoming copy of the loop
    its old state arrived on.
    """
    def __init__(self):
        super().__init__()
        self.update = torch.nn.Linear(3 + 4 + 2, 4)
        self.emit = torch.nn.Linear(4, 3)

    def forward(self, x):
        degree = (x.shape[-1] - node_signature(0).width(WIDTHS)) \
            // WIDTHS[PEER]
        places = node_signature(degree).slices(WIDTHS)
        peers = x[:, places[PEER]].reshape(len(x), degree, -1).mean(1)
        boundary = torch.cat([peers, x[:, places[STATE]], x[:, places[CLUE]]],
                             -1)
        new = torch.tanh(self.update(boundary))
        out = torch.zeros_like(x)
        out[:, places[STATE].stop:places[STATE].stop + WIDTHS[STATE]] = new
        out[:, places[PEER]] = self.emit(new).repeat(1, degree)
        return out


def oracle(cmap, state, rounds):
    """
    One call per box per round, port by port.  It reads the port order with
    ``box_ports`` as the implementation does, so a wrong un-reversal of the
    clockwise storage would cancel out here: the port order is pinned
    independently by the ``edges`` of the ``interpret`` doctest.
    """
    widths = cmap.port_widths
    offsets = [sum(widths[:i]) for i in range(len(widths))]
    for _ in range(rounds):
        outgoing = torch.zeros_like(state)
        for index, box in enumerate(cmap.boxes):
            ports = cmap.box_ports(index)
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


def test_an_interpretation_is_a_named_triple():
    """
    ``MapNN.interpret`` is the package's name for compiling a diagram, and
    leaves ``torch.nn.Module.compile`` as torch defines it; what it returns
    is addressed by name rather than by position.
    """
    pair = from_relation(((1, ), (0, )), node_signature(1))
    model = small_model()
    found = model.interpret(pair)
    assert isinstance(found, Interpretation) and found.cmap is found[0]
    assert set(found.ports) == set(found.heads) == {
        ("cell", PEER), ("cell", STATE), ("cell", CLUE)}
    assert MapNN.compile is torch.nn.Module.compile


def test_one_model_two_shapes_one_set_of_weights():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    model = small_model()
    before = [(name, id(value)) for name, value in model.named_parameters()]
    with torch.no_grad():
        one, other = model(pair), model(path)
    assert one.shape == (1, sum(model.interpret(pair).cmap.port_widths))
    assert other.shape == (1, sum(model.interpret(path).cmap.port_widths))
    assert [(name, id(value))
            for name, value in model.named_parameters()] == before
    assert model.interpret(pair).cmap is model.interpret(pair).cmap
    assert model.cache_stats()["misses"] == 2


def test_a_cell_reads_its_degree_through_the_signature():
    """
    One set of ``Linear`` weights serves a pair and a path, degrees one and
    two, because the cell reads its degree off the width it is handed and
    its layout off ``Signature.slices``; a ``Linear`` on the whole boundary
    of one degree cannot run the other.
    """
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    torch.manual_seed(0)
    model = MapNN(OB, {"cell": Pooled()}, rounds=3).double()
    clues = [torch.randn(2, sites, 2, dtype=torch.double) for sites in (2, 3)]
    states = [model.read(shape, model(shape, {("cell", CLUE): clue}),
                         ("cell", STATE))
              for shape, clue in zip((pair, path), clues)]
    assert [state.shape for state in states] == [(2, 2, 4), (2, 3, 4)]
    assert all(state.abs().sum() > 0 for state in states)
    states[1].sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    wide = MapNN(OB, {"cell": torch.nn.Linear(15, 15)}, rounds=1).double()
    assert wide(pair).shape == (1, 30)
    with raises(RuntimeError):
        wide(path)


def test_training_copies_a_clue_across_a_pair():
    """
    Two nodes, each handed a clue and asked to answer with its peer's: a
    few hundred Adam steps on an MLP cell bring the loss down by an order
    of magnitude, and every parameter of the model receives a gradient.
    """
    answer = Ty("answer")
    node = Signature((
        Orbit(PEER, 1, Sym.PERM), Orbit(STATE, traced=True),
        Orbit(CLUE, traced=True), Orbit(answer, traced=True)))
    widths = {PEER: 4, STATE: 4, CLUE: 2, answer: 2}
    pair = from_relation(((1, ), (0, )), node)
    torch.manual_seed(0)
    cell = torch.nn.Sequential(
        torch.nn.Linear(node.width(widths), 16), torch.nn.Tanh(),
        torch.nn.Linear(16, node.width(widths)))
    model = MapNN({role: Dim(n) for role, n in widths.items()},
                  {"cell": cell}, rounds=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(200):
        clue = torch.randn(32, 2, 2)
        state = model(pair, {("cell", CLUE): clue})
        loss = torch.nn.functional.mse_loss(
            model.read(pair, state, ("cell", answer)), clue.flip(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] / 10
    assert all(p.grad.abs().sum() > 0 for p in model.parameters())
    assert model.cache_stats()["misses"] == 1


def test_injection_readds_the_initial_state():
    """
    ``MapNN(inject=True)`` runs the affine transition ``sigma(Phi(s)) + i``
    of the map itself, and differs from the linear one.
    """
    pair = from_relation(((1, ), (0, )), node_signature(1))
    model = small_model(inject=True)
    torch.manual_seed(1)
    clue = torch.randn(2, 2, 2, dtype=torch.double)
    with torch.no_grad():
        found = model(pair, {("cell", CLUE): clue})
        init = model.initial(pair, {("cell", CLUE): clue})
        expected = model.interpret(pair).cmap(
            init=init, n_rounds=3, inject=True, return_flat=True)
        plain = model(pair, {("cell", CLUE): clue}, inject=False)
    assert torch.equal(found, expected) and not torch.equal(found, plain)


def test_a_missing_role_or_generator_is_an_error():
    pair = from_relation(((1, ), (0, )), node_signature(1))
    with raises(KeyError):
        MapNN({PEER: Dim(3), CLUE: Dim(2)}, {"cell": Mix()}).interpret(pair)
    with raises(KeyError):
        MapNN(OB, {"unit": Mix()}).interpret(pair)
    erased = MapNN({**OB, CLUE: Dim(0)}, {"cell": Mix()}).double()
    with raises(KeyError):
        erased.read(pair, erased(pair), ("cell", CLUE))


def test_initial_follows_the_like_tensor():
    pair = from_relation(((1, ), (0, )), node_signature(1))
    model = small_model()
    total = sum(model.interpret(pair).cmap.port_widths)
    state = model.initial(pair, rows=3, like=torch.zeros(1))
    assert state.shape == (3, total) and state.dtype == torch.float32
    assert model.initial(pair).dtype == torch.double


def test_compile_rounds_records_its_kwargs_for_every_map():
    """
    ``compile_rounds`` hands the per-round step of every map, interpreted
    before or after, to ``torch.compile``; the compilation itself is lazy,
    on the first forward pass of each map.
    """
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    model = small_model()
    before = model.interpret(pair).cmap
    assert model.compile_rounds(mode="reduce-overhead") is model
    after = model.interpret(path).cmap
    assert before.compile_kwargs == after.compile_kwargs \
        == {"mode": "reduce-overhead"}


def test_the_compilation_cache_is_bounded():
    node = node_signature(1)
    model = small_model(cache=2)
    shapes = [from_relation(((1, ), (0, )), node) for _ in range(4)]
    for shape in shapes:
        model.interpret(shape)
    assert model.cache_stats(reset=True) == {
        "hits": 0, "misses": 4, "held": 2, "capacity": 2}
    assert model.cache_stats()["misses"] == 0


def test_deep_supervises_every_round():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    model = small_model(rounds=4)
    cmap = model.interpret(pair).cmap
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
    cmap, ports, heads = model.interpret(pair)
    assert len(heads["cell", STATE]) == 4 and len(ports["cell", STATE]) == 8
    values = torch.arange(4 * 4, dtype=torch.double).reshape(1, 4, 4)
    state = model.initial(pair, {("cell", STATE): values})
    assert torch.equal(model.read(pair, state, ("cell", STATE)), values)
    for head in heads["cell", STATE]:
        assert torch.equal(cmap.read(state, (head, )),
                           cmap.read(state, (cmap.edges[head], )))
