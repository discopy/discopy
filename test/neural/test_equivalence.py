# -*- coding: utf-8 -*-

"""
The refactor of ``discopy.neural`` into a category-generic engine changed
no arithmetic, and this is where that claim is checked.

Every test compares against ``docs/neural/golden/``, recorded by
``docs/neural/golden/record.py`` against the code as it stood *before* the
refactor and committed first.  The comparisons are ``torch.equal`` --
bitwise, not ``allclose`` -- on the CPU in float32 and float64, with one
thread, because a multi-threaded reduction adds its partial sums back in a
different order.

If one of these fails, the fix is the line that changed the arithmetic,
never the tolerance.

The one deliberate structural change is recorded in
:func:`test_ports_refined`: the clique cell's ``[h, c]`` concatenation
became two named roles, ``hidden`` and ``memory``, on one traced loop.
That splits two of its ports in two and renames the bookkeeping around
them -- ``edges``, ``port_widths``, ``layout``, ``inverse`` -- while
leaving the routing permutation, the box-order layout and every number
untouched, which is what the tests below assert.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from discopy.frobenius import Ty
from discopy.neural import Dim, Orbit, Signature, Sym, check_equivariant
from discopy.neural.batch import Batch
from discopy.neural.cells import Mode, Relation, Site
from discopy.neural.functor import Interpretation, interpret
from discopy.neural.skeleton import from_relation
from discopy.utils import AxiomError

GOLDEN = Path(__file__).resolve().parents[2] / "docs" / "neural" / "golden"
NEURAL = GOLDEN.parent

pytest.importorskip("pandas")
if str(NEURAL) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(NEURAL))

record = pytest.importorskip("golden.record")           # noqa: E402
from sudoku import models as zoo                        # noqa: E402
from sudoku import signature as roles                   # noqa: E402
from sudoku import skeleton as grids                    # noqa: E402

MODELS = tuple(record.MODELS)

#: The clique cell is the one whose ports were refined; the others must
#: match the golden port-level bookkeeping exactly.
REFINED = ("rrn", )


def fingerprint(name: str) -> tuple[dict, dict]:
    """ The recorded JSON and arrays of one model. """
    return (json.loads((GOLDEN / f"{name}.json").read_text()),
            dict(np.load(GOLDEN / f"{name}.npz")))


@pytest.fixture(autouse=True)
def deterministic():
    """ The conditions the goldens were recorded under. """
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(record.THREADS)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(False)


def build(name: str, dtype=torch.float32):
    """ One golden model, freshly seeded, in the given precision. """
    return record.build(record.MODELS[name]).to(dtype)


def batch(arrays: dict):
    """ The golden batch of puzzles and their solutions. """
    return (torch.as_tensor(arrays["clues"], dtype=torch.long),
            torch.as_tensor(arrays["target"], dtype=torch.long))


# --- T1: the wiring --------------------------------------------------------

@pytest.mark.parametrize("name", MODELS)
def test_computation_fingerprint(name):
    """
    T1a. What the arithmetic is a function of -- the total width, the
    routing permutation, the box-order permutation and the groups sharing
    a module -- is unchanged, for every model including the refined one.
    """
    golden, arrays = fingerprint(name)
    fresh: dict = {}
    structure = record.structure(build(name), fresh)
    assert structure["computation"] == golden["structure"]["computation"]
    for key in ("src", "perm"):
        assert np.array_equal(fresh[key], arrays[key]), key


@pytest.mark.parametrize(
    "name", [name for name in MODELS if name not in REFINED])
def test_port_fingerprint(name):
    """
    T1b. The port-level bookkeeping -- the involution, the port widths,
    the layout and its inverse, the port families a solver reads -- is
    unchanged for every model whose roles were not refined.
    """
    golden, arrays = fingerprint(name)
    fresh: dict = {}
    structure = record.structure(build(name), fresh)
    assert structure["counts"] == golden["structure"]["counts"]
    assert structure["families"] == golden["structure"]["families"]
    for key in ("edges", "port_widths", "layout", "inverse"):
        assert np.array_equal(fresh[key], arrays[key]), key


def test_ports_refined():
    """
    T1c. The clique's refinement is a refinement and nothing else: merging
    each ``(hidden, memory)`` pair back into one port recovers the golden
    port widths exactly, and the flat width is unchanged.
    """
    golden, arrays = fingerprint("rrn")
    fresh: dict = {}
    structure = record.structure(build("rrn"), fresh)
    peers = len(grids.peers_of()[0])
    merged = []
    for start in range(0, len(fresh["port_widths"]), peers + 6):
        block = list(fresh["port_widths"][start:start + peers + 6])
        # clockwise: the two clue ports, then the four halves of the loop,
        # then the peers; each adjacent pair of halves was one port.
        merged += block[:2] + [block[2] + block[3], block[4] + block[5]] \
            + block[6:]
    assert merged == list(arrays["port_widths"])
    assert structure["counts"]["router_total"] \
        == golden["structure"]["counts"]["router_total"]
    assert structure["counts"]["parameters"] \
        == golden["structure"]["counts"]["parameters"]


# --- T2: the parameters ----------------------------------------------------

@pytest.mark.parametrize("name", MODELS)
def test_parameters(name):
    """
    T2. The parameters have the same shapes in the same order, the same
    values under the same seed, and the golden ``state_dict`` loads
    through the rename map with nothing missing and nothing unexpected.
    """
    from migration import rename, translate             # noqa: E402
    golden, _ = fingerprint(name)
    model = build(name)
    fresh = record.parameters(model)
    assert fresh["sha256"] == golden["parameters"]["sha256"]
    old = golden["parameters"]["named"]
    assert [shape for _, shape in fresh["named"]] \
        == [shape for _, shape in old]
    assert [rename(name, key) for key, _ in old] \
        == [key for key, _ in fresh["named"]]

    # a checkpoint as it was stored before the refactor: the model's own
    # weights, under the golden's key names.  Loading it back through the
    # rename map has to be strict and total.
    inverse = {rename(name, key): key
               for key in golden["parameters"]["state_dict_keys"]}
    assert len(inverse) == len(golden["parameters"]["state_dict_keys"])
    stored = {inverse[key]: value
              for key, value in model.state_dict().items()}
    assert set(stored) == set(golden["parameters"]["state_dict_keys"])
    report = model.load_state_dict(translate(name, stored), strict=True)
    assert not report.missing_keys and not report.unexpected_keys


# --- T3, T4, T5: the numbers ----------------------------------------------

@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("dtype,tag", [(torch.float32, "f32"),
                                       (torch.float64, "f64")])
def test_forward(name, dtype, tag):
    """ T3. The logits are bitwise equal at every supervised checkpoint. """
    _, arrays = fingerprint(name)
    clues, _ = batch(arrays)
    fresh = torch.stack(record.forward(build(name, dtype), clues))
    assert torch.equal(fresh, torch.as_tensor(arrays[f"logits_{tag}"]))


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("dtype,tag", [(torch.float32, "f32"),
                                       (torch.float64, "f64")])
def test_backward(name, dtype, tag):
    """ T4. Every parameter gradient is bitwise equal. """
    from migration import rename                        # noqa: E402
    _, arrays = fingerprint(name)
    clues, target = batch(arrays)
    fresh = record.backward(build(name, dtype), clues, target)
    golden = {rename(name, key[len(f"grad_{tag}/"):]): value
              for key, value in arrays.items()
              if key.startswith(f"grad_{tag}/")}
    assert set(golden) == set(fresh)
    for key, value in golden.items():
        assert torch.equal(fresh[key], torch.as_tensor(value)), key


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("dtype,tag", [(torch.float32, "f32"),
                                       (torch.float64, "f64")])
def test_trajectory(name, dtype, tag):
    """ T5. The loss after each of 20 optimizer steps is bitwise equal. """
    golden, arrays = fingerprint(name)
    clues, target = batch(arrays)
    fresh = record.trajectory(build(name, dtype), clues, target)
    assert fresh == golden[f"trajectory_{tag}"]


# --- T6: the reference oracle ---------------------------------------------

@pytest.mark.parametrize("name", ["goi", "trm"])
def test_forward_reference(name):
    """
    T6. The vectorized forward still agrees with the one-call-per-box
    reference implementation, which the refactor never touched.
    """
    _, arrays = fingerprint(name)
    clues, _ = batch(arrays)
    model = build(name, torch.float64)
    inject = model.schedule.inject
    with torch.no_grad():
        state = model.initial(clues[:2])
        fast = model.grid(init=state, n_rounds=2, inject=inject)
        slow = model.grid.forward_reference(
            init=state, n_rounds=2, inject=inject)
    assert len(fast) == len(slow) == len(model.grid.boxes)
    for emitted, expected in zip(fast, slow):
        assert torch.allclose(emitted, expected, atol=1e-12)


# --- T7: the layout --------------------------------------------------------

def test_slices_match_the_old_cursor():
    """
    T7. ``Signature.slices`` reproduces the hand-written cursor arithmetic
    of the cells it replaced, asserted on the actual old offsets.
    """
    dim, state_dim, y_dim = 24, 96, 48
    places = roles.cell(3).slices(
        {roles.MESSAGE: dim, roles.STATE: state_dim,
         roles.CLUE: dim, roles.ANSWER: y_dim})
    # GoICell.forward: cursor = n_message * dim; state at cursor; cursor +=
    # 2 * state_dim; clue at cursor; cursor += 2 * dim; answer at cursor.
    cursor = 3 * dim
    assert places[roles.MESSAGE] == slice(0, cursor)
    assert places[roles.STATE] == slice(cursor, cursor + state_dim)
    cursor += 2 * state_dim
    assert places[roles.CLUE] == slice(cursor, cursor + dim)
    cursor += 2 * dim
    assert places[roles.ANSWER] == slice(cursor, cursor + y_dim)

    peers = 20
    places = roles.peer_cell(peers).slices(
        {roles.PEER: state_dim, roles.HIDDEN: state_dim,
         roles.MEMORY: state_dim, roles.CLUE: dim})
    # RRNCell.forward: cursor = n_peers * state_dim; hidden at cursor,
    # memory at cursor + state_dim; cursor += 4 * state_dim; clue there.
    cursor = peers * state_dim
    assert places[roles.PEER] == slice(0, cursor)
    assert places[roles.HIDDEN] == slice(cursor, cursor + state_dim)
    assert places[roles.MEMORY] == slice(
        cursor + state_dim, cursor + 2 * state_dim)
    cursor += 4 * state_dim
    assert places[roles.CLUE] == slice(cursor, cursor + dim)


# --- T8: the equations -----------------------------------------------------

def test_check_equivariant():
    """
    T8. The declared symmetry holds for the cells that declare it, and a
    module that breaks it is rejected -- without the negative control the
    checker would be vacuous.
    """
    message = roles.MESSAGE
    unit = roles.unit(9)
    relation = Relation(unit, {message: 4}, hidden=8).double()
    assert check_equivariant(relation, unit, {message: 4})[message] < 1e-12

    cell = roles.cell(3)
    widths = {roles.MESSAGE: 4, roles.STATE: 8, roles.CLUE: 4,
              roles.ANSWER: 0}
    site = Site(cell, widths,
                {roles.STATE: Mode.STATE, roles.CLUE: Mode.INPUT,
                 roles.ANSWER: Mode.CARRY}, hidden=8).double()
    assert check_equivariant(site, cell, widths)[message] < 1e-12

    class Skew(torch.nn.Module):
        """ Weighs each port differently, so it cannot commute. """
        def forward(self, x):
            return x * torch.arange(1., 1 + x.shape[-1], dtype=x.dtype)

    with pytest.raises(AxiomError):
        check_equivariant(Skew(), unit, {message: 4})


def test_fusion_is_lax():
    """
    A learned relation does *not* fuse: the diagnostic reports how far it
    is, and the answer is not zero.  This is the honest version of the
    claim that Frobenius structure is preserved -- it is not.
    """
    from discopy.neural.cells import fusion_residual
    unit = roles.unit(4)
    relation = Relation(unit, {roles.MESSAGE: 2}, hidden=4).double()
    assert fusion_residual(relation, unit, {roles.MESSAGE: 2}) > 1e-3


# --- T9: the monoidal product ---------------------------------------------

def test_batch_splits_back():
    """
    T9. Batching over variable structure is the monoidal product of maps:
    ``a @ b`` runs both at once and splits back into exactly ``a`` and
    ``b``.

    The decomposition is exact -- the split sizes are the members' own
    flat widths, the round trip is the identity, and the product's ports
    are the members' ports in order.  The *values* agree to a few units in
    the last place rather than bitwise, because a batched module call over
    six sites is not the same kernel as one over two: this is the same
    rounding freedom :meth:`discopy.neural.CMap.compile` documents, and it
    is why a batch of mixed shapes is a new capability rather than a
    silent change to an existing run.
    """
    peer, state = Ty("peer"), Ty("state")
    node = Signature((Orbit(peer, 1, Sym.PERM), Orbit(state, traced=True)))
    ob = {peer: Dim(3), state: Dim(5)}
    torch.manual_seed(0)
    module = Site(node, {peer: 3, state: 5},
                  {state: Mode.STATE}, hidden=6).double()
    meaning = Interpretation(ob, {"cell": module})

    pair = from_relation(((1, ), (0, )), node)
    triple = from_relation(((1, ), (0, ), (3, ), (2, )), node)
    one, other = interpret(meaning, pair), interpret(meaning, triple)

    together = Batch((pair, triple), meaning)
    assert len(together.wiring.cmap.boxes) \
        == len(pair.boxes) + len(triple.boxes)
    assert together.totals == (one.router.total, other.router.total)
    assert together.wiring.cmap.port_widths \
        == one.cmap.port_widths + other.cmap.port_widths
    # one module call for every site of both members, not one per member
    assert len(together.wiring.cmap._fused_routing["metas"]) == 1

    torch.manual_seed(1)
    states = [torch.randn(2, part.router.total, dtype=torch.double)
              for part in (one, other)]
    for state, found in zip(states, together.split(together.join(states))):
        assert torch.equal(state, found)

    apart = [part(init=state, n_rounds=3, return_flat=True)
             for part, state in zip((one, other), states)]
    run = together.wiring(init=together.join(states), n_rounds=3,
                          return_flat=True)
    for expected, found in zip(apart, together.split(run)):
        assert expected.shape == found.shape
        assert torch.allclose(expected, found, rtol=0, atol=1e-14)


# --- T10: the erasure invariant -------------------------------------------

def test_answer_erasure():
    """
    T10. Sending the answer role to ``Dim(0)`` erases its ports and its
    wires, so the recursion's skeleton interpreted without an answer is
    structurally the factor graph of the plain model.
    """
    torch.manual_seed(0)
    plain = zoo.goi(rounds=2)
    torch.manual_seed(0)
    erased = interpret(
        Interpretation(roles.factor_widths(zoo.WIDTHS["goi"], 0),
                       {"cell": plain.cell, "unit": plain.factor}),
        grids.factor_graph())
    assert erased.cmap.port_widths == plain.grid.port_widths
    assert list(erased.cmap.edges) == list(plain.grid.edges)
    assert erased.ports == plain.wiring.ports


# --- T11: the halt head is free -------------------------------------------

def test_halt_head_is_free():
    """
    T11. An engine with a halt head and one without, built under the same
    seed, have bitwise identical weights: the head is initialised to
    constants and built last, so it draws the same numbers and no others.
    """
    torch.manual_seed(0)
    plain = zoo.trm(rounds=2, cycles=2, n_sup=3)
    torch.manual_seed(0)
    halting = zoo.act(rounds=2, cycles=2, n_sup=3, halt_head="softmin")
    shared = dict(plain.named_parameters())
    found = dict(halting.named_parameters())
    assert set(found) - set(shared) == {"q_head.weight", "q_head.bias"}
    for key, value in shared.items():
        assert torch.equal(value, found[key]), key
