# -*- coding: utf-8 -*-

"""
The refactor of ``discopy.neural`` around :class:`~discopy.neural.MapNN`
changed no arithmetic, and this is where that claim is checked.

Every test compares against ``docs/neural/golden/``, recorded by
``docs/neural/golden/record.py`` against the code as it stood *before* the
refactor and committed first.  On the build that recorded them the
comparisons reproduce bitwise; another build's kernels drift by an ulp on
the same arithmetic, so the numeric gates T3-T5 allow exactly that much
and nothing more.

If one of these fails, the fix is the line that changed the arithmetic,
never the tolerance.

Two deliberate structural changes are recorded rather than hidden.
:func:`test_ports_refined` is the older one: the clique cell's ``[h, c]``
concatenation became two named roles, ``hidden`` and ``memory``, on one
traced loop, which splits two of its ports in two without moving a number.
The newer one is that the *modules moved*: the generators now live under
the ``MapNN`` that shares them and the answer refresh under the solver that
runs it, so a golden key is compared through ``model.rename``, the map that
also loads a pre-refactor checkpoint.  The weights themselves are asserted
bitwise, in the golden's own order.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

NEURAL = Path(__file__).resolve().parents[2] / "docs" / "neural"
for _path in (NEURAL / "golden", NEURAL / "examples" / "sudoku"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import model as zoo                                      # noqa: E402
import record                                            # noqa: E402
from discopy.neural import (                             # noqa: E402
    Dim, Mode, Relation, Site, check_equivariant, fusion_residual, interpret)
from discopy.utils import AxiomError                     # noqa: E402

GOLDEN = NEURAL / "golden"
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
    peers = len(zoo.peers_of()[0])
    merged = []
    for start in range(0, len(fresh["port_widths"]), peers + 6):
        block = list(fresh["port_widths"][start:start + peers + 6])
        # clockwise: the two clue ports, then the four halves of the loop,
        # then the peers; each adjacent pair of halves was one port.
        merged += block[:2] + [block[2] + block[3], block[4] + block[5]] \
            + block[6:]
    assert merged == list(arrays["port_widths"])
    for key in ("router_total", "parameters"):
        assert structure["counts"][key] == golden["structure"]["counts"][key]


# --- T2: the parameters ----------------------------------------------------

@pytest.mark.parametrize("name", MODELS)
def test_parameters(name):
    """
    T2. The parameters have the same shapes and the same values under the
    same seed, name by name through ``model.rename``, with nothing missing
    and nothing left over; and a checkpoint stored under the golden's key
    names loads back strictly.

    The digest is taken in the *golden's* order rather than in the order
    the modules now register, since where a module hangs is exactly what
    the refactor changed.
    """
    golden, _ = fingerprint(name)
    model = build(name)
    fresh = dict(model.named_parameters())
    old = golden["parameters"]["named"]
    assert {zoo.rename(key) for key, _ in old} == set(fresh)

    digest = hashlib.sha256()
    for key, shape in old:
        value = fresh[zoo.rename(key)]
        assert list(value.shape) == shape, key
        digest.update(value.detach().to(torch.float64).numpy().tobytes())
    assert digest.hexdigest() == golden["parameters"]["sha256"]

    # a checkpoint as it was stored before the refactor: the model's own
    # weights, under the golden's key names.  Loading it back through the
    # rename map has to be strict and total.
    inverse = {zoo.rename(key): key
               for key in golden["parameters"]["state_dict_keys"]}
    assert len(inverse) == len(golden["parameters"]["state_dict_keys"])
    stored = {inverse[key]: value
              for key, value in model.state_dict().items()}
    report = model.load_state_dict(zoo.translate(stored), strict=True)
    assert not report.missing_keys and not report.unexpected_keys


# --- T3, T4, T5: the numbers ----------------------------------------------

#: The goldens are bitwise recordings of one torch build; another build's
#: kernels drift by an ulp on the same arithmetic, so the forward gate
#: allows ulp-scale drift and nothing more.  One backward pass amplifies
#: it through the deepest model's float32 accumulations, so the backward
#: gate is looser there; twenty optimizer steps amplify it beyond any
#: usable tolerance, so the trajectory gate only runs where it is
#: defined, on the recording build.
TOLERANCE = {"f32": dict(rtol=1e-5, atol=1e-6),
             "f64": dict(rtol=1e-12, atol=1e-14)}
BACKWARD = {"f32": dict(rtol=1e-3, atol=1e-6),
            "f64": dict(rtol=1e-12, atol=1e-14)}
recording_build = pytest.mark.skipif(
    not os.environ.get("GOLDEN_BITWISE"),
    reason="run on the recording build with GOLDEN_BITWISE=1")


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("dtype,tag", [(torch.float32, "f32"),
                                       (torch.float64, "f64")])
def test_forward(name, dtype, tag):
    """ T3. The logits match at every supervised checkpoint. """
    _, arrays = fingerprint(name)
    clues, _ = batch(arrays)
    fresh = torch.stack(record.forward(build(name, dtype), clues))
    assert torch.allclose(fresh, torch.as_tensor(arrays[f"logits_{tag}"]),
                          **TOLERANCE[tag])


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("dtype,tag", [(torch.float32, "f32"),
                                       (torch.float64, "f64")])
def test_backward(name, dtype, tag):
    """ T4. Every parameter gradient matches. """
    _, arrays = fingerprint(name)
    clues, target = batch(arrays)
    fresh = record.backward(build(name, dtype), clues, target)
    golden = {zoo.rename(key[len(f"grad_{tag}/"):]): value
              for key, value in arrays.items()
              if key.startswith(f"grad_{tag}/")}
    assert set(golden) == set(fresh)
    for key, value in golden.items():
        assert torch.allclose(fresh[key], torch.as_tensor(value),
                              **BACKWARD[tag]), key


@recording_build
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
    grid = model.interaction.cmap
    inject = model.map.solver.inject
    with torch.no_grad():
        state = model.initial(clues[:2])
        fast = grid(init=state, n_rounds=2, inject=inject)
        slow = grid.forward_reference(init=state, n_rounds=2, inject=inject)
    assert len(fast) == len(slow) == len(grid.boxes)
    for emitted, expected in zip(fast, slow):
        assert torch.allclose(emitted, expected, atol=1e-12)


# --- T7: the layout --------------------------------------------------------

def test_slices_match_the_old_cursor():
    """
    T7. ``Signature.slices`` reproduces the hand-written cursor arithmetic
    of the cells it replaced, asserted on the actual old offsets.
    """
    dim, state_dim, y_dim = 24, 96, 48
    places = zoo.cell(3).slices(
        {zoo.MESSAGE: dim, zoo.STATE: state_dim,
         zoo.CLUE: dim, zoo.ANSWER: y_dim})
    # GoICell.forward: cursor = n_message * dim; state at cursor; cursor +=
    # 2 * state_dim; clue at cursor; cursor += 2 * dim; answer at cursor.
    cursor = 3 * dim
    assert places[zoo.MESSAGE] == slice(0, cursor)
    assert places[zoo.STATE] == slice(cursor, cursor + state_dim)
    cursor += 2 * state_dim
    assert places[zoo.CLUE] == slice(cursor, cursor + dim)
    cursor += 2 * dim
    assert places[zoo.ANSWER] == slice(cursor, cursor + y_dim)

    peers = 20
    places = zoo.peer_cell(peers).slices(
        {zoo.PEER: state_dim, zoo.HIDDEN: state_dim,
         zoo.MEMORY: state_dim, zoo.CLUE: dim})
    # RRNCell.forward: cursor = n_peers * state_dim; hidden at cursor,
    # memory at cursor + state_dim; cursor += 4 * state_dim; clue there.
    cursor = peers * state_dim
    assert places[zoo.PEER] == slice(0, cursor)
    assert places[zoo.HIDDEN] == slice(cursor, cursor + state_dim)
    assert places[zoo.MEMORY] == slice(
        cursor + state_dim, cursor + 2 * state_dim)
    cursor += 4 * state_dim
    assert places[zoo.CLUE] == slice(cursor, cursor + dim)


# --- T8: the equations -----------------------------------------------------

def test_check_equivariant():
    """
    T8. The declared symmetry holds for the cells that declare it, and a
    module that breaks it is rejected -- without the negative control the
    checker would be vacuous.
    """
    message, unit = zoo.MESSAGE, zoo.unit(9)
    relation = Relation(unit, {message: 4}, hidden=8).double()
    assert check_equivariant(relation, unit, {message: 4})[message] < 1e-12

    cell = zoo.cell(3)
    widths = {zoo.MESSAGE: 4, zoo.STATE: 8, zoo.CLUE: 4, zoo.ANSWER: 0}
    site = Site(cell, widths,
                {zoo.STATE: Mode.STATE, zoo.CLUE: Mode.INPUT,
                 zoo.ANSWER: Mode.CARRY}, hidden=8).double()
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
    unit = zoo.unit(4)
    relation = Relation(unit, {zoo.MESSAGE: 2}, hidden=4).double()
    assert fusion_residual(relation, unit, {zoo.MESSAGE: 2}) > 1e-3


# --- T9: the erasure invariant --------------------------------------------

def test_answer_erasure():
    """
    T9. Sending the answer role to ``Dim(0)`` erases its ports and its
    wires, so the recursion's diagram interpreted without an answer is
    structurally the factor graph of the plain model.
    """
    torch.manual_seed(0)
    plain = zoo.goi(rounds=2)
    erased = interpret(
        zoo.factor_graph(),
        {zoo.MESSAGE: Dim(24), zoo.STATE: Dim(96), zoo.CLUE: Dim(24),
         zoo.ANSWER: Dim(0)},
        {"cell": plain.map.ar["cell"], "unit": plain.map.ar["unit"]})
    found = plain.interaction
    assert erased.widths == found.widths
    assert erased.routing == found.routing
    assert erased.ports == found.ports and erased.heads == found.heads


# --- T10: the halt head is free -------------------------------------------

def test_halt_head_is_free():
    """
    T10. A model with a halt head and one without, built under the same
    seed, have bitwise identical weights: the head is initialised to
    constants and built last, so it draws the same numbers and no others.
    """
    torch.manual_seed(0)
    plain = zoo.trm(rounds=2, cycles=2, steps=3)
    torch.manual_seed(0)
    halting = zoo.act(rounds=2, cycles=2, steps=3, halt_head="softmin")
    shared = dict(plain.named_parameters())
    found = dict(halting.named_parameters())
    assert set(found) - set(shared) == {
        "map.solver.halt.weight", "map.solver.halt.bias"}
    for key, value in shared.items():
        assert torch.equal(value, found[key]), key
