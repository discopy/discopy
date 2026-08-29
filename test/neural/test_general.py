# -*- coding: utf-8 -*-

"""
What the library can do, independently of any task: non-uniform degree,
heterogeneous batching through :class:`~discopy.neural.MapNN`, the four
cell shapes, and the four solvers.

``test_equivalence.py`` holds the recorded sudoku models bitwise; this file
holds the capabilities to the same standard -- the fused forward against
the one-call-per-box oracle, exact equivariance in float64, and exact
agreement with the code paths they generalise wherever the general case
degenerates to the special one.
"""

import pytest
import torch

from discopy.frobenius import Ty
from discopy.neural import (
    ACT, Batch, Dim, FixedPoint, HaltHead, Interaction, Iterate, MapNN, Mode,
    Orbit, Recursion, Refresh, Relation, Signature, Site, Sym, bucket,
    check_equivariant, from_incidence, from_relation, interpret)
from discopy.neural.cells import POOL, Cyclic, Gate
from discopy.utils import AxiomError

MESSAGE, PEER = Ty("message"), Ty("peer")
STATE, CLUE, ANSWER = Ty("state"), Ty("clue"), Ty("answer")


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def node_signature(degree: int = 1, role=PEER) -> Signature:
    return Signature((
        Orbit(role, degree, Sym.PERM), Orbit(STATE, traced=True),
        Orbit(CLUE, traced=True)))


def make_site(signature, widths, **kwargs):
    return Site(signature, widths, {STATE: Mode.STATE, CLUE: Mode.INPUT},
                hidden=8, **kwargs)


# --- non-uniform degree -----------------------------------------------------

def test_nonuniform_relation_forward_matches_reference():
    """A path graph -- degrees 1, 2, 1 -- runs, and the fused forward
    agrees with the one-call-per-box oracle."""
    node = node_signature(1)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    assert [len(box.cod) for box in path.boxes] == [5, 6, 5]
    torch.manual_seed(0)
    module = make_site(node, {PEER: 3, STATE: 4, CLUE: 2}).double()
    found = interpret(path, {PEER: Dim(3), STATE: Dim(4), CLUE: Dim(2)},
                      {"cell": module})
    # one module, two degrees: exactly two batched groups
    assert len(found.cmap._fused_routing["metas"]) == 2
    torch.manual_seed(1)
    init = torch.randn(2, found.total, dtype=torch.double)
    with torch.no_grad():
        fast = found.cmap(init=init, n_rounds=3, inject=False)
        slow = found.cmap.forward_reference(init=init, n_rounds=3,
                                            inject=False)
    for emitted, expected in zip(fast, slow):
        assert torch.allclose(emitted, expected, atol=1e-12)


def test_nonuniform_incidence_forward_matches_reference():
    """Mixed node degrees and mixed relation sizes through
    ``from_incidence``, against the oracle."""
    node = node_signature(1, MESSAGE)
    unit = Signature((Orbit(MESSAGE, 2, Sym.PERM), ))
    # nodes 0, 1 in relations 0 and 1; node 2 only in relation 1.
    shape = from_incidence(((0, 1), (0, 1), (1, )), node, unit)
    assert [len(box.cod) for box in shape.boxes[3:]] == [2, 3]
    torch.manual_seed(0)
    site = make_site(node, {MESSAGE: 3, STATE: 4, CLUE: 2}).double()
    relation = Relation(unit, {MESSAGE: 3}, hidden=8).double()
    found = interpret(shape, {MESSAGE: Dim(3), STATE: Dim(4), CLUE: Dim(2)},
                      {"cell": site, "unit": relation})
    torch.manual_seed(1)
    init = torch.randn(2, found.total, dtype=torch.double)
    with torch.no_grad():
        fast = found.cmap(init=init, n_rounds=3, inject=False)
        slow = found.cmap.forward_reference(init=init, n_rounds=3,
                                            inject=False)
    for emitted, expected in zip(fast, slow):
        assert torch.allclose(emitted, expected, atol=1e-12)


def test_a_graph_level_readout_is_a_generator():
    """One extra relation wired to every node, under its own name, gives a
    graph-level readout with no solver change: its per-leg emissions are
    readable as an ordinary port family."""
    node = node_signature(1, MESSAGE)
    unit = Signature((Orbit(MESSAGE, 2, Sym.PERM), ))
    shape = from_incidence(((0, 1), (0, 1), (1, )), node, unit,
                           relation_name=("unit", "readout"))
    assert [box.name for box in shape.boxes] \
        == ["cell", "cell", "cell", "unit", "readout"]
    torch.manual_seed(0)
    site = make_site(node, {MESSAGE: 3, STATE: 4, CLUE: 2}).double()
    constraint = Relation(unit, {MESSAGE: 3}, hidden=8).double()
    readout = Relation(unit, {MESSAGE: 3}, hidden=8).double()
    found = interpret(
        shape, {MESSAGE: Dim(3), STATE: Dim(4), CLUE: Dim(2)},
        {"cell": site, "unit": constraint, "readout": readout})
    assert found.sites(("readout", MESSAGE)) == 3   # one leg per node
    torch.manual_seed(1)
    init = torch.randn(2, found.total, dtype=torch.double)
    with torch.no_grad():
        flat = found.advance(init, 2)
    assert found.read(flat, ("readout", MESSAGE)).shape == (2, 3, 3)


def test_a_wire_may_only_be_erased_whole():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    with pytest.raises(ValueError, match="non-atomic"):
        interpret(pair, {PEER: Dim(3, 3), STATE: Dim(4), CLUE: Dim(2)},
                  {"cell": None})


# --- the four cell shapes ---------------------------------------------------

def test_cyclic_cell_is_cyclic_not_symmetric():
    leg = Ty("leg")
    planar = Signature((Orbit(leg, 5, Sym.CYCLIC), ))
    torch.manual_seed(0)
    box = Cyclic(planar, {leg: 3}, hidden=8).double()
    assert check_equivariant(box, planar, {leg: 3})[leg] < 1e-15
    # the same module under a PERM signature breaks: rotation is not the
    # whole symmetric group, so the checker must reject it.
    with pytest.raises(AxiomError, match="perm"):
        check_equivariant(box, Signature((Orbit(leg, 5, Sym.PERM), )),
                          {leg: 3})


def test_cyclic_cell_is_arity_fixed():
    leg = Ty("leg")
    torch.manual_seed(0)
    box = Cyclic(Signature((Orbit(leg, 4, Sym.CYCLIC), )), {leg: 2},
                 hidden=4)
    with pytest.raises(ValueError, match="arity-fixed"):
        box(torch.zeros(1, 6))


def test_gate_cell_none_only():
    wire = Ty("wire")
    torch.manual_seed(0)
    box = Gate(Signature((Orbit(wire, 2), )), {wire: 3}, hidden=8).double()
    assert box(torch.zeros(2, 6, dtype=torch.double)).shape == (2, 6)
    # NONE declares nothing, so nothing to check and nothing broken
    assert check_equivariant(
        box, Signature((Orbit(wire, 2), )), {wire: 3}) == {}
    with pytest.raises(AxiomError, match="perm"):
        check_equivariant(
            box, Signature((Orbit(wire, 2, Sym.PERM), )), {wire: 3})
    with pytest.raises(ValueError, match="arity-fixed"):
        box(torch.zeros(1, 9, dtype=torch.double))


def test_per_leg_emission():
    node = node_signature(3)
    widths = {PEER: 3, STATE: 4, CLUE: 2}
    torch.manual_seed(0)
    site = make_site(node, widths, per_leg=True).double()
    # still permutation-equivariant: the state pools, each leg answers its
    # own message
    assert check_equivariant(site, node, widths)[PEER] < 1e-12
    torch.manual_seed(1)
    row = torch.randn(1, site.signature.width(widths), dtype=torch.double)
    out = site(row)[:, :3 * widths[PEER]].reshape(1, 3, widths[PEER])
    assert not torch.allclose(out[0, 0], out[0, 1])
    # whereas the broadcast site answers every leg alike
    torch.manual_seed(0)
    broadcast = make_site(node, widths).double()
    out = broadcast(row)[:, :3 * widths[PEER]].reshape(1, 3, widths[PEER])
    assert torch.equal(out[0, 0], out[0, 1])


def test_per_leg_needs_emit():
    with pytest.raises(ValueError, match="per-leg"):
        make_site(node_signature(2), {PEER: 3, STATE: 4, CLUE: 2},
                  per_leg=True, emit=False)


def test_max_pooling_keeps_an_extremum_exactly():
    """
    A max pool is the reduction a change of degree leaves alone: adding a
    member it dominates changes nothing, whereas a mean divides by the
    members and a sum grows with them.  It also reduces without reordering
    a floating-point sum, so a site pooling with it is equivariant to
    within an ulp of the build's kernels rather than up to accumulation
    order.
    """
    node = node_signature(3)
    widths = {PEER: 3, STATE: 4, CLUE: 2}
    torch.manual_seed(0)
    site = make_site(node, widths, pool="max").double()
    assert check_equivariant(site, node, widths)[PEER] < 1e-15

    orbit = torch.tensor([[[1., 5.], [3., 2.]]])
    grown = torch.cat([orbit, torch.tensor([[[0., -1.]]])], dim=1)
    assert torch.equal(POOL["max"](orbit), POOL["max"](grown))
    assert not torch.equal(POOL["mean"](orbit), POOL["mean"](grown))
    assert not torch.equal(POOL["sum"](orbit), POOL["sum"](grown))


# --- the model --------------------------------------------------------------

def small_model(solver=None, dtype=torch.double) -> MapNN:
    node = node_signature(1)
    torch.manual_seed(0)
    cell = make_site(node, {PEER: 3, STATE: 4, CLUE: 2})
    return MapNN({PEER: Dim(3), STATE: Dim(4), CLUE: Dim(2)},
                 {"cell": cell},
                 solver=solver or Iterate(rounds=3)).to(dtype)


def test_one_model_two_shapes_one_set_of_weights():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    model = small_model()
    before = [(name, id(value)) for name, value in model.named_parameters()]
    with torch.no_grad():
        one, other = model(pair), model(path)
    assert one.shape == (1, model.compile(pair).total)
    assert other.shape == (1, model.compile(path).total)
    assert [(name, id(value))
            for name, value in model.named_parameters()] == before
    # compiling is cached by the identity of the diagram
    assert model.compile(pair) is model.compile(pair)
    assert isinstance(model.compile(pair), Interaction)
    assert model.compile(model.compile(pair)) is model.compile(pair)


def test_the_compilation_cache_is_bounded():
    node = node_signature(1)
    model = small_model()
    model.cache = 2
    shapes = [from_relation(((1, ), (0, )), node) for _ in range(4)]
    for shape in shapes:
        model.compile(shape)
    assert len(model._compiled) == 2


def test_a_batch_is_the_product_of_its_members():
    """Running ``[a, b]`` gives, member for member, what running ``a`` and
    ``b`` alone gives: the monoidal product through the whole model."""
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
    assert batch.widths(model.ob) == (
        model.compile(pair).total, model.compile(path).total)
    for expected, found in zip(alone, pieces):
        assert torch.allclose(expected, found, atol=1e-13)
    # one module call covers every site of the same degree, not one per
    # member: here the degrees are 1 and 2, hence two groups
    assert len(model.compile(batch).cmap._fused_routing["metas"]) == 2


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
    states = [torch.randn(2, model.compile(shape).total, dtype=torch.double)
              for shape in (pair, path)]
    for expected, found in zip(
            states, batch.split_state(batch.join(states), model.ob)):
        assert torch.equal(expected, found)
    with pytest.raises(ValueError, match="expected 2"):
        batch.join(states[:1])


# --- the solvers ------------------------------------------------------------

def test_iterate_supervises_every_round():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    model = small_model(Iterate(rounds=4, inject=False))
    found = model.compile(pair)
    torch.manual_seed(1)
    state = torch.randn(2, found.total, dtype=torch.double)
    with torch.no_grad():
        last, every = model.run(pair, state, deep=True)
    assert len(every) == 4 and torch.equal(last, every[-1])
    with torch.no_grad():
        assert torch.equal(every[1], found.advance(state, 2))
    assert Iterate(rounds=4).depth == 4


def test_fixed_point_stops_on_the_residual():
    """A map that settles is detected as settled; the differentiation
    policy decides whether the iteration is in the graph."""
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    still = torch.nn.Linear(15, 15).double()
    with torch.no_grad():
        still.weight.zero_(), still.bias.fill_(0.5)
    model = MapNN({PEER: Dim(3), STATE: Dim(4), CLUE: Dim(2)},
                  {"cell": still},
                  solver=FixedPoint(rounds=16, tol=1e-9)).double()
    found = model.compile(pair)
    state = found.zeros(2, dtype=torch.double)
    settled, every = model.run(pair, state, deep=True)
    assert float(found.residual(settled).max()) < 1e-9
    assert len(every) < 16          # it stopped early

    # "last" differentiates one round from a detached limit, "full" every
    # round actually run.
    unrolled = MapNN(model.ob, {"cell": still},
                     solver=FixedPoint(rounds=4, tol=None,
                                       backward="full")).double()
    with torch.enable_grad():
        state = found.zeros(2, dtype=torch.double).requires_grad_(True)
        model.run(pair, state)[0].sum().backward()
        assert state.grad is None or float(state.grad.abs().sum()) == 0.0
        state = found.zeros(2, dtype=torch.double).requires_grad_(True)
        unrolled.run(pair, state)[0].sum().backward()
        assert state.grad is not None
    with pytest.raises(ValueError):
        FixedPoint(backward="implicit")


def refresh_model(cycles: int = 2, steps: int = 2, halt: str = None):
    node = Signature((
        Orbit(PEER, 1, Sym.PERM), Orbit(STATE, traced=True),
        Orbit(CLUE, traced=True), Orbit(ANSWER, traced=True)))
    torch.manual_seed(0)
    cell = Site(node, {PEER: 3, STATE: 4, CLUE: 2, ANSWER: 2},
                {STATE: Mode.STATE, CLUE: Mode.INPUT, ANSWER: Mode.CARRY},
                hidden=8, resumable=True)
    refresh = Refresh(torch.nn.GRUCell(4, 2), torch.nn.LayerNorm(2),
                      source=("cell", STATE), target=("cell", ANSWER))
    solver = Recursion(1, cycles, steps, refresh=refresh) if halt is None \
        else ACT(1, cycles, steps, refresh=refresh,
                 halt=HaltHead(2, halt))
    model = MapNN({PEER: Dim(3), STATE: Dim(4), CLUE: Dim(2),
                   ANSWER: Dim(2)}, {"cell": cell}, solver=solver).double()
    return node, model


def test_recursion_refreshes_only_its_target():
    node, model = refresh_model()
    pair = from_relation(((1, ), (0, )), node)
    found = model.compile(pair)
    torch.manual_seed(1)
    state = torch.randn(2, found.total, dtype=torch.double)
    with torch.no_grad():
        refreshed = model.solver.refresh(found, state)
    assert not torch.equal(found.read(refreshed, ("cell", ANSWER)),
                           found.read(state, ("cell", ANSWER)))
    for key in (("cell", STATE), ("cell", CLUE), ("cell", PEER)):
        assert torch.equal(found.read(refreshed, key),
                           found.read(state, key))
    # every copy of the loop gets the same value written to it
    every = found.read(refreshed, ("cell", ANSWER), every=True)
    assert torch.equal(every[:, 0::2],
                       found.read(refreshed, ("cell", ANSWER)))
    assert Recursion(6, 3, 8).depth == 144
    with pytest.raises(ValueError, match="`steps`"):
        model.run(pair, state, rounds=2)


@pytest.mark.parametrize("cycles,differentiated", [(1, True), (2, False)])
def test_only_the_last_cycle_is_differentiated(cycles, differentiated):
    """The detach boundary, pinned exactly: with more than one cycle no
    gradient reaches the state a step started from."""
    node, model = refresh_model(cycles=cycles)
    pair = from_relation(((1, ), (0, )), node)
    found = model.compile(pair)
    state = found.zeros(2, dtype=torch.double).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model.solver.step(found, state, cycles=cycles).sum().backward()
    assert (state.grad is not None) == differentiated
    assert model.ar["cell"].encode[0].weight.grad is not None


def test_act_reads_the_trace_it_halts_on():
    node, model = refresh_model(halt="softmin")
    pair = from_relation(((1, ), (0, )), node)
    found = model.compile(pair)
    state = found.zeros(2, dtype=torch.double)
    with torch.no_grad():
        after, answer, halt = model.solver.step(found, state)
        plain, _ = model.run(pair, state)
    assert answer.shape == (2, 2, 2) and halt.shape == (2, 2)
    assert model.solver.halt.logit(halt).shape == (2, )
    assert torch.equal(answer, found.read(after, ("cell", ANSWER)))
    assert plain.shape == state.shape
    correct = torch.zeros(2, 2, dtype=torch.bool)
    assert float(model.solver.halt.loss(halt, correct)) > 0.0
    with pytest.raises(ValueError, match="trace to halt on"):
        ACT(1, 2, 2, halt=HaltHead(2))
