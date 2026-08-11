# -*- coding: utf-8 -*-

"""
The generality fixes on top of the frozen sudoku behaviour: non-uniform
degree, structural batching through the engine, the NONE and CYCLIC
cells, per-leg emission and verifier-based selection.

Everything here is *additive*: ``test_equivalence.py`` holds the old
behaviour bitwise, this file holds the new capabilities to the same
standard -- the fused forward against the reference oracle, exact
equivariance in float64, and exact agreement with the code paths they
generalise wherever the general case degenerates to the old one.
"""

import pytest
import torch

from discopy.frobenius import Ty
from discopy.neural import Dim, Orbit, Signature, Sym, check_equivariant
from discopy.neural.cells import Cyclic, Gate, Mode, Relation, Site
from discopy.neural.engine import (
    Engine, Schedule, evaluate_act, evaluate_selected)
from discopy.neural.functor import Interpretation, interpret
from discopy.neural.skeleton import from_incidence, from_relation
from discopy.utils import AxiomError

MESSAGE, PEER = Ty("message"), Ty("peer")
STATE, CLUE = Ty("state"), Ty("clue")


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def node_signature(degree: int = 1) -> Signature:
    return Signature((
        Orbit(PEER, degree, Sym.PERM), Orbit(STATE, traced=True),
        Orbit(CLUE, traced=True)))


def factor_node(degree: int = 1) -> Signature:
    return Signature((
        Orbit(MESSAGE, degree, Sym.PERM), Orbit(STATE, traced=True),
        Orbit(CLUE, traced=True)))


def make_site(signature, widths, **kwargs):
    return Site(signature, widths,
                {STATE: Mode.STATE, CLUE: Mode.INPUT},
                hidden=8, **kwargs)


# --- P2: non-uniform degree -------------------------------------------------

def test_nonuniform_relation_forward_matches_reference():
    """A path graph -- degrees 1, 2, 1 -- runs, and the fused forward
    agrees with the one-call-per-box oracle."""
    node = node_signature(1)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    assert [path.signature(i).orbits[0].arity for i in range(3)] == [1, 2, 1]
    torch.manual_seed(0)
    widths = {PEER: 3, STATE: 4, CLUE: 2}
    module = make_site(node, widths).double()
    wiring = interpret(Interpretation(
        {PEER: Dim(3), STATE: Dim(4), CLUE: Dim(2)},
        {"cell": module}), path)
    # one module, two degrees: exactly two batched groups
    assert len(wiring.cmap._fused_routing["metas"]) == 2
    torch.manual_seed(1)
    init = torch.randn(2, wiring.router.total, dtype=torch.double)
    with torch.no_grad():
        fast = wiring.cmap(init=init, n_rounds=3, inject=False)
        slow = wiring.cmap.forward_reference(
            init=init, n_rounds=3, inject=False)
    for emitted, expected in zip(fast, slow):
        assert torch.allclose(emitted, expected, atol=1e-12)


def test_nonuniform_incidence_forward_matches_reference():
    """Mixed node degrees and mixed relation sizes through
    ``from_incidence``, against the oracle."""
    node = factor_node(1)
    unit = Signature((Orbit(MESSAGE, 2, Sym.PERM), ))
    # nodes 0, 1 in relations 0 and 1; node 2 only in relation 1.
    shape = from_incidence(((0, 1), (0, 1), (1, )), node, unit)
    sizes = [shape.signature(i).orbits[0].arity for i in range(3, 5)]
    assert sizes == [2, 3]
    torch.manual_seed(0)
    widths = {MESSAGE: 3, STATE: 4, CLUE: 2}
    site = make_site(node, widths).double()
    relation = Relation(unit, {MESSAGE: 3}, hidden=8).double()
    wiring = interpret(Interpretation(
        {MESSAGE: Dim(3), STATE: Dim(4), CLUE: Dim(2)},
        {"cell": site, "unit": relation}), shape)
    torch.manual_seed(1)
    init = torch.randn(2, wiring.router.total, dtype=torch.double)
    with torch.no_grad():
        fast = wiring.cmap(init=init, n_rounds=3, inject=False)
        slow = wiring.cmap.forward_reference(
            init=init, n_rounds=3, inject=False)
    for emitted, expected in zip(fast, slow):
        assert torch.allclose(emitted, expected, atol=1e-12)


def test_uniform_path_unchanged():
    """On uniform combinatorics the builders produce the same wiring as
    before: same box types, same involution."""
    node = factor_node(3)
    unit = Signature((Orbit(MESSAGE, 3, Sym.PERM), ))
    grid = from_incidence(
        ((0, 1, 2), (0, 1, 2), (0, 1, 2)), node, unit)
    assert all(grid.signature(i) == node for i in range(3))
    assert all(grid.signature(i) == unit for i in range(3, 6))
    # the declared signature object itself, not a resized copy
    assert grid.signature(0) is grid.signatures["cell"]


def test_wrong_type_still_rejected():
    """A box whose type is not a resizing of its signature's first orbit
    is still an error."""
    from discopy import frobenius
    from discopy.neural.skeleton import Skeleton
    node = node_signature(1)
    bad = frobenius.Box("cell", frobenius.Ty(),
                        frobenius.Ty("peer", "peer"))
    with pytest.raises(ValueError, match="type of its signature"):
        Skeleton(frobenius.CMap.from_wiring(
            (bad, ), [((0, 0), (0, 1))]), {"cell": node})


def test_product_mixes_canonical_degrees():
    """Two skeletons declaring one name at different canonical arities --
    two grid sizes of one task -- still form a product; a genuinely
    different signature still does not."""
    node = factor_node(1)
    small = from_incidence(
        ((0, ), (0, )), node, Signature((Orbit(MESSAGE, 2, Sym.PERM), )))
    large = from_incidence(
        ((0, ), (0, ), (0, )), node,
        Signature((Orbit(MESSAGE, 3, Sym.PERM), )))
    both = small @ large
    assert len(both.boxes) == len(small.boxes) + len(large.boxes)
    other = from_incidence(
        ((0, ), (0, )), node,
        Signature((Orbit(MESSAGE, 2, Sym.CYCLIC), )))
    with pytest.raises(ValueError, match="two different signatures"):
        small @ other


# --- P3b: a graph-level readout is a generator ------------------------------

def test_graph_readout_is_a_relation():
    """One extra relation wired to every node, under its own name, gives
    a graph-level readout with no engine change: its per-leg emissions
    are readable as an ordinary port family."""
    node = factor_node(1)
    unit = Signature((Orbit(MESSAGE, 2, Sym.PERM), ))
    shape = from_incidence(
        ((0, 1), (0, 1), (1, )), node, unit,
        relation_name=("unit", "readout"))
    assert [box.name for box in shape.boxes] \
        == ["cell", "cell", "cell", "unit", "readout"]
    torch.manual_seed(0)
    widths = {MESSAGE: 3, STATE: 4, CLUE: 2}
    site = make_site(node, widths).double()
    constraint = Relation(unit, {MESSAGE: 3}, hidden=8).double()
    readout = Relation(unit, {MESSAGE: 3}, hidden=8).double()
    wiring = interpret(Interpretation(
        {MESSAGE: Dim(3), STATE: Dim(4), CLUE: Dim(2)},
        {"cell": site, "unit": constraint, "readout": readout}), shape)
    family = wiring.ports[("readout", MESSAGE)]
    assert len(family) == 3  # one leg per node
    torch.manual_seed(1)
    init = torch.randn(2, wiring.router.total, dtype=torch.double)
    with torch.no_grad():
        flat = wiring.cmap(init=init, n_rounds=2, inject=False,
                           return_flat=True)
    assert wiring.router.read(flat, family).shape == (2, 3, 3)


# --- P4: the NONE and CYCLIC cells ------------------------------------------

def test_cyclic_cell_is_cyclic_not_symmetric():
    leg = Ty("leg")
    planar = Signature((Orbit(leg, 5, Sym.CYCLIC), ))
    torch.manual_seed(0)
    box = Cyclic(planar, {leg: 3}, hidden=8).double()
    assert check_equivariant(box, planar, {leg: 3})[leg] == 0.0
    # the same module under a PERM signature breaks: rotation is not
    # the whole symmetric group, so the checker must reject it.
    symmetric = Signature((Orbit(leg, 5, Sym.PERM), ))
    with pytest.raises(AxiomError, match="perm"):
        check_equivariant(box, symmetric, {leg: 3})


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
    # but the same MLP under a PERM signature is rejected
    with pytest.raises(AxiomError, match="perm"):
        check_equivariant(
            box, Signature((Orbit(wire, 2, Sym.PERM), )), {wire: 3})
    with pytest.raises(ValueError, match="arity-fixed"):
        box(torch.zeros(1, 9, dtype=torch.double))


# --- P5: per-leg emission ----------------------------------------------------

def test_per_leg_emission():
    node = node_signature(3)
    widths = {PEER: 3, STATE: 4, CLUE: 2}
    torch.manual_seed(0)
    site = make_site(node, widths, per_leg=True).double()
    # still permutation-equivariant: the state pools, each leg answers
    # its own message
    assert check_equivariant(site, node, widths)[PEER] < 1e-12
    # and genuinely per-leg: distinct messages get distinct answers
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


# --- P1: the engine binds a batch of skeletons -------------------------------

def small_engine(skeleton, n_classes: int = 4):
    class ConstantEncoder(torch.nn.Embedding):
        pass

    widths = {PEER: Dim(3), STATE: Dim(4), CLUE: Dim(2)}
    return Engine(
        skeleton, widths,
        parts={
            "embedding": lambda: ConstantEncoder(n_classes + 1, 2),
            "cell": lambda: make_site(
                node_signature(1), {PEER: 3, STATE: 4, CLUE: 2}),
            "readout": lambda: torch.nn.Linear(4, n_classes)},
        sites={"cell": "cell"},
        schedule=Schedule(rounds=3, cycles=1, steps=1,
                          inject=True, supervise="round"),
        clue=("cell", CLUE), state=("cell", STATE), n_classes=n_classes)


def test_bind_matches_members():
    """Binding [a, b] gives, member for member, the logits of binding
    [a] and [b] alone: the monoidal product through the whole engine."""
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    torch.manual_seed(0)
    engine = small_engine(pair).double()
    torch.manual_seed(1)
    x_pair = torch.randint(0, 5, (2, 2))
    x_path = torch.randint(0, 5, (2, 3))

    engine.bind([pair])
    with torch.no_grad():
        alone_pair = engine(x_pair)
    engine.bind([path])
    assert engine.n_cells == 3 and engine.member_cells == (3, )
    with torch.no_grad():
        alone_path = engine(x_path)

    engine.bind([pair, path])
    assert engine.member_cells == (2, 3)
    with torch.no_grad():
        both = engine(torch.cat([x_pair, x_path], dim=1))
    left, right = engine.split_cells(both)
    assert torch.allclose(left, alone_pair, atol=1e-13)
    assert torch.allclose(right, alone_path, atol=1e-13)

    engine.unbind()
    assert engine.n_cells == 2 and engine.batch is None
    with torch.no_grad():
        assert torch.allclose(engine(x_pair), alone_pair, atol=1e-13)
    with pytest.raises(ValueError, match="no batch"):
        engine.split_cells(both)


def test_bind_padding_is_dropped():
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    torch.manual_seed(0)
    engine = small_engine(pair).double()
    batch = engine.bind([pair, path, pair], pad=True)
    assert len(batch.parts) == 4 and batch.given == 3
    assert engine.member_cells == (2, 3, 2, 2)
    x = torch.randint(0, 5, (1, sum(engine.member_cells)))
    with torch.no_grad():
        pieces = engine.split_cells(engine(x))
    assert [piece.shape[1] for piece in pieces] == [2, 3, 2]


def test_bind_shares_parameters():
    """Rebinding never touches the weights: the same parameter tensors,
    in the same order, before, during and after."""
    node = node_signature(1)
    pair = from_relation(((1, ), (0, )), node)
    path = from_relation(((1, ), (0, 2), (1, )), node)
    torch.manual_seed(0)
    engine = small_engine(pair)
    before = [(name, id(value)) for name, value
              in engine.named_parameters()]
    engine.bind([path, pair])
    assert [(name, id(value)) for name, value
            in engine.named_parameters()] == before
    engine.unbind()
    assert [(name, id(value)) for name, value
            in engine.named_parameters()] == before


# --- P6: verifier-based selection -------------------------------------------

class TinySplit:
    def __init__(self, puzzles, solutions):
        self.puzzles, self.solutions = puzzles, solutions

    def __len__(self):
        return len(self.puzzles)


@pytest.fixture(scope="module")
def act_model_and_split():
    import sys
    from pathlib import Path
    neural = Path(__file__).resolve().parents[2] / "docs" / "neural"
    if str(neural) not in sys.path:
        sys.path.insert(0, str(neural))
    pytest.importorskip("pandas")
    import numpy as np
    from sudoku import models as zoo
    torch.manual_seed(0)
    np.random.seed(0)
    model = zoo.act(rounds=1, cycles=2, n_sup=2, halt_head="softmin")
    arrays = np.load(neural / "golden" / "act.npz")
    split = TinySplit(arrays["clues"][:8], arrays["target"][:8] + 1)
    return model, split


def decode_rule(logits, clues):
    predicted = logits.argmax(-1) + 1
    return torch.where(clues > 0, clues, predicted)


def test_selected_degenerates_to_act(act_model_and_split):
    """With one rollout and no noise, the selected evaluation *is* the
    ACT evaluation, number for number."""
    model, split = act_model_and_split
    plain = evaluate_act(model, split, decode_rule, batch_size=4)
    selected = evaluate_selected(
        model, split, decode_rule, rollouts=1, sigma=0.0, batch_size=4)
    for key in ("cell", "board", "depth"):
        assert selected[key] == plain[key], key
    assert selected["board_mean"] == plain["board"]
    assert selected["board_oracle"] == plain["board"]


def test_selected_is_bounded_and_deterministic(act_model_and_split):
    model, split = act_model_and_split
    one = evaluate_selected(model, split, decode_rule,
                            rollouts=3, sigma=0.5, batch_size=4, seed=7)
    two = evaluate_selected(model, split, decode_rule,
                            rollouts=3, sigma=0.5, batch_size=4, seed=7)
    assert one == two
    assert one["board_oracle"] >= one["board"] >= 0.0
    assert one["board_oracle"] >= one["board_mean"]


def test_act_engine_refuses_bind(act_model_and_split):
    model, _ = act_model_and_split
    with pytest.raises(NotImplementedError, match="halt head"):
        model.bind([model.skeleton])
