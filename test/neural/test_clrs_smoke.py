# -*- coding: utf-8 -*-

"""
End to end, small: the real CLRS example on a tiny model.

``test_sudoku_smoke.py`` does this for a study whose diagram is fixed and
whose samples are two arrays.  The CLRS example is the other shape: **every
batch is its own diagram**, drawn out of the adjacency matrices of its
members, and a sample is a whole execution trace -- inputs, a hint per step,
a trajectory length and outputs.  This file holds the pieces that only that
shape has: the incidence, the layout of the site axis, the two clocks (a
round of message passing versus a step of the algorithm), the per-sample
length masking of the hint loss, and the depth sweep.

Everything is the production path:
``docs/neural/examples/CLRS_small``'s own ``dataset``, ``model``, ``train``
and ``evaluate``.  Only the *size* is a test's: four trajectories of the
committed ``val`` splits, at widths small enough that a whole training loop
is a second.

The assertions are deterministic on purpose -- shapes, box names, which
parameters receive a gradient, a checkpoint round trip -- rather than an
accuracy that would make CI a coin flip.  The one optimisation claim, that
the loss goes down on a fixed batch, is a smoke assertion beside them.
"""

import doctest
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from discopy.neural import cells

EXAMPLE = Path(__file__).resolve().parents[2] / "docs" / "neural" \
    / "examples" / "CLRS_small"

#: The module names the examples of ``docs/neural`` share.  They are
#: imported as top-level modules -- an example is a directory of scripts,
#: not a package -- so two examples in one pytest session would shadow each
#: other's ``model``.  :func:`_example` therefore imports under a saved
#: ``sys.modules`` and puts back whatever was there.
SHARED = ("config", "dataset", "model", "train", "evaluate")


def _example(directory: Path) -> dict:
    """ The modules of one example, imported without leaving a trace. """
    saved = {name: sys.modules.pop(name, None) for name in SHARED}
    sys.path.insert(0, str(directory))
    try:
        return {name: importlib.import_module(name) for name in SHARED}
    finally:
        sys.path.remove(str(directory))
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


if not (EXAMPLE / "data" / "bfs-val.npz").exists():
    pytest.skip("the CLRS-30 cache is not built; see examples/CLRS_small",
                allow_module_level=True)

CLRS = _example(EXAMPLE)
config, dataset = CLRS["config"], CLRS["dataset"]
zoo, training, evaluations = CLRS["model"], CLRS["train"], CLRS["evaluate"]

#: Small enough that a whole training loop is a second, large enough that
#: every module of every model is exercised at its real shape.
TINY = config.Widths(dim=4, state_dim=8, hidden=8, edge_dim=4, graph_dim=8)

#: Four rounds is two algorithm steps, since a node reaches a node through
#: a box; the two clocks are what several tests below pin down.  A *fixed*
#: depth, unlike the campaigns: the trajectory rule would run
#: ``dag_shortest_paths`` for a hundred rounds on data whose only job here
#: is to exercise a shape.  ``test_the_depth_is_the_trajectory`` is where
#: the rule itself is checked.
ROUNDS, BATCH = 4, 4

BUDGET = config.Budget(name="smoke", epochs=1, batch_size=2,
                       eval_batch_size=2, rounds=ROUNDS, n_train=BATCH,
                       n_wide=2, eval_every=1, seeds=(0, ), sweep=(1.0, 2.0))

#: The two algorithms whose diagram is the complete graph and whose probes
#: live on pairs: H1's showcases.
SHOWCASE = ("floyd_warshall", "matrix_chain_order")


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def splits():
    """ Four trajectories of the committed ``val`` split, per algorithm. """
    return {name: dataset.load(name, "val").subsample(BATCH)
            for name in config.ALGORITHMS}


@pytest.fixture(scope="module")
def batches(splits):
    return {name: zoo.Batch.of(split) for name, split in splits.items()}


def build(algorithm: str, seed: int = 0, widths=TINY, **kwargs):
    """ One tiny model, freshly seeded. """
    torch.manual_seed(seed)
    np.random.seed(seed)
    kwargs.setdefault("steps", ROUNDS // zoo.HOPS)
    return zoo.build(algorithm, widths, **kwargs)


def finite(model) -> bool:
    return all(torch.isfinite(value).all()
               for value in model.state_dict().values())


def test_the_examples_own_doctests(splits):
    """
    The example is documentation as much as it is code, and its docstrings
    carry runnable claims -- what a signature's ports are, what the two
    clocks map to, what a decoder's shape is, at which width the two arms
    of H1 have the same parameter count.  ``pytest`` collects the
    library's doctests and not an example's, so they are run here.

    The example's modules are put back under their own names for the
    duration: :class:`doctest.DocTestFinder` decides whether a class
    belongs to the module it was handed by looking its ``__module__`` up
    in ``sys.modules``, so with the *sudoku* example's ``config`` sitting
    there -- which is exactly what a whole-suite run leaves behind -- every
    class of this one is skipped and the file passes having run nothing.
    ``found.attempted`` is the assertion that catches that.

    The collision is **planted here** rather than waited for.  It arrives
    naturally only when some other module has already claimed the name,
    so a guard that relies on that is a guard whose failure depends on
    what else pytest collected and in which order -- and this whole test
    exists because a silent zero once passed for a green suite.  Putting
    a decoy in ``sys.modules`` first makes the assertion fail whenever
    the restore is missing, running alone or last.
    """
    modules = (config, dataset, zoo, training, evaluations)
    saved = {name: sys.modules.get(name) for name in SHARED}
    for name in SHARED:
        sys.modules[name] = ModuleType(name)
    sys.modules.update(dict(zip(SHARED, modules)))
    sys.path.insert(0, str(EXAMPLE))  # a docstring may import a sibling
    try:
        for module in modules:
            found = doctest.testmod(module, verbose=False)
            assert found.failed == 0, module.__name__
            assert found.attempted, module.__name__
    finally:
        sys.path.remove(str(EXAMPLE))
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


# --- the data --------------------------------------------------------------

@pytest.mark.parametrize("algorithm", config.ALGORITHMS)
def test_the_cache_is_the_benchmark(algorithm, splits):
    """
    The cached arrays pass the example's own verification: the shapes
    agree with the spec, the graph is undirected with a full diagonal, and
    the outputs are what the reference algorithm computes.
    """
    dataset.check(splits[algorithm], samples=2, log=lambda _: None)


def test_a_split_round_trips_through_its_cache(tmp_path, splits, monkeypatch):
    """ Saving and reading a split gives the same arrays back. """
    monkeypatch.setattr(dataset, "DATA_DIR", tmp_path)
    dataset.save(splits["bfs"])
    again = dataset.read("bfs", "val")
    assert np.array_equal(again.lengths, splits["bfs"].lengths)
    for key, value in splits["bfs"].outputs.items():
        assert np.array_equal(again.outputs[key], value)


# --- the diagram -----------------------------------------------------------

def test_the_incidence_is_the_graph():
    """
    One node box per node, one edge box per undirected edge, one readout
    relation per sample -- and the readout is what gives an isolated node
    a degree, which is what an Erdos-Renyi sample needs.
    """
    pairs = [np.array([[0, 1], [1, 2]]), np.zeros((0, 2), dtype=np.int64)]
    lists, names = zoo.incidence(pairs, 3)
    assert names == ("edge", "edge", "readout", "readout")
    assert lists == ((0, 2), (0, 1, 2), (1, 2), (3, ), (3, ), (3, ))
    assert all(entry for entry in lists), "a node of degree zero"
    drawn = zoo.graph(pairs, 3)
    assert [box.name for box in drawn.boxes] == \
        ["node"] * 6 + ["edge", "edge", "readout", "readout"]


def test_a_batch_is_the_disjoint_union(splits):
    """ Twice the trajectories is twice the node sites, one map. """
    model = build("bfs")
    sizes = []
    for count in (2, 4):
        batch = zoo.Batch.of(splits["bfs"].subsample(count))
        zoo.check_layout(model, batch)
        sizes.append(model.map.sites(batch.diagram, model.answer))
    assert sizes == [2 * splits["bfs"].n, 4 * splits["bfs"].n]


def test_minimum_has_no_edges_and_no_edge_module(batches):
    """
    ``minimum`` is the non-graph sanity check: its nodes meet only in the
    readout relation, so the diagram has no edge box and the
    interpretation carries no edge module -- the readout alone is what can
    solve it.
    """
    batch = batches["minimum"]
    assert batch.n_edges == 0
    assert {box.name for box in batch.diagram.boxes} == {"node", "readout"}
    model = build("minimum")
    assert set(model.map.ar) == {"node", "readout"}
    assert ("edge", zoo.ESTATE) not in model.map.compile(batch.diagram).heads


@pytest.mark.parametrize("algorithm", config.ALGORITHMS)
def test_the_layout_is_what_the_decoders_assume(algorithm, batches):
    zoo.check_layout(build(algorithm), batches[algorithm])


@pytest.mark.parametrize("algorithm", SHOWCASE)
def test_the_showcase_diagram_is_the_complete_graph(
        algorithm, splits, batches):
    """
    ``floyd_warshall`` answers about every pair and ``matrix_chain_order``
    about every interval, so every pair is a box -- and since that wiring
    depends on the size alone, two batches of a size are the *same*
    diagram and a whole split compiles once.
    """
    split, batch = splits[algorithm], batches[algorithm]
    size, samples = split.n, len(split)
    assert batch.n_edges == samples * size * (size - 1) // 2
    assert batch.diagram is zoo.dense_graph(size, samples)
    assert zoo.Batch.of(split).diagram is batch.diagram


@pytest.mark.parametrize("algorithm", SHOWCASE)
def test_the_pair_grid_is_the_wiring(algorithm, batches):
    """
    Every edge decoder indexes a pair grid, and that the grid's ``(i, j)``
    is the box wired to nodes ``i`` and ``j`` is a property of
    ``from_incidence``'s box order rather than of anything declared.  So
    it is checked against the wiring -- and a permuted grid is rejected,
    which is what makes the check a check.
    """
    model, batch = build(algorithm), batches[algorithm]
    zoo.check_layout(model, batch)
    shuffled = replace(batch, pairs=batch.pairs.flip(-1))
    with pytest.raises(AssertionError):
        zoo.check_pairs(model.map.compile(batch.diagram), shuffled)


def test_an_edge_probe_is_decoded_off_the_pair_that_answers_it(batches):
    """
    The shapes of Part 2's new heads: a pair scalar and mask are ``n x n``,
    a pair pointer is ``n x n x n`` -- one candidate node per pair -- and
    the pair latents they read are symmetric, since an undirected edge is
    one box that both orders of its endpoints see.
    """
    model, batch = build("floyd_warshall"), batches["floyd_warshall"]
    with torch.no_grad():
        latents = model.latents(batch, model.initial(batch))
        found = model(batch)
    assert latents.pairs.shape[:3] == (BATCH, batch.size, batch.size)
    assert torch.equal(latents.pairs, latents.pairs.transpose(1, 2))
    assert torch.equal(latents.pairs[:, 0, 0], torch.zeros_like(
        latents.pairs[:, 0, 0])), "the diagonal is no box"
    assert found["D"].shape == (BATCH, batch.size, batch.size)
    assert found["Pi"].shape == (BATCH, ) + (batch.size, ) * 3


@pytest.mark.parametrize("algorithm", SHOWCASE)
def test_the_h1_arms_differ_only_in_the_edge_state(algorithm, batches):
    """
    H1's ablation: the same diagram with ``ESTATE`` sent to ``Dim(0)``.
    The wiring is identical, the messages still pass through the edge
    boxes, the edge decoders still answer -- from the two node states
    alone -- and the parameter counts are matched within the 10% the
    study's discipline asks for.
    """
    batch = batches[algorithm]
    with_state, without = build(algorithm), build(
        algorithm, edge_state=False)
    arms = [one.map.compile(batch.diagram) for one in (with_state, without)]
    assert len(arms[0].cmap.boxes) == len(arms[1].cmap.boxes), \
        "one diagram, two interpretations of it"
    assert arms[1].total < arms[0].total, "the state ports are erased"
    assert ("edge", zoo.ESTATE) not in arms[1].heads
    with torch.no_grad():
        assert without.latents(batch, without.initial(batch)).pairs is None
        found = without(batch)
    for name, value in found.items():
        assert torch.isfinite(value).all(), name


def test_the_h1_arms_are_parameter_matched():
    """
    ``config.WIDTHS["paired"]`` is the width the node-only arm is widened
    to, and it is the *recorded result* of ``model.matched`` rather than a
    number someone chose: a cell without a recurrent state is a smaller
    cell, so leaving the widths alone would compare memory against
    capacity.
    """
    for algorithm in SHOWCASE:
        arms = [zoo.count_parameters(build(
            algorithm, widths=config.WIDTHS[key], edge_state=state))
            for key, state in (("mpnn", True), ("paired", False))]
        assert abs(arms[0] - arms[1]) / arms[0] < 0.1, (algorithm, arms)


# --- the two ends of the task ----------------------------------------------

def test_a_probe_is_decoded_by_a_head_of_its_own(batches):
    """
    ``minimum`` decodes three ``node/mask_one`` probes -- the answer
    ``min``, the running answer ``min_h`` and the loop counter ``i`` --
    and they are three different questions asked of one state.  One head
    per *type* would ask a single logit per node to be three
    distributions at once; one head per *probe* is both representable and
    what ``clrs._src.nets`` does.
    """
    model, batch = build("minimum"), batches["minimum"]
    mask_one = [name for name in zoo.decoded("minimum")
                if dataset.kind("minimum", name)[1] == "mask_one"]
    assert mask_one == ["min_h", "i", "min"]
    assert len({id(model.decoders[name]) for name in mask_one}) == 3
    with torch.no_grad():
        found = model(batch)
    assert not torch.equal(found["min"], found["i"])


def test_the_survey_knows_which_decoders_exist():
    """
    ``dataset.DECODABLE`` is what ``dataset.survey`` subtracts to answer
    "what is left to build". It is a plain tuple because ``dataset`` must
    import without torch, so this is what keeps it equal to the decoders
    that actually exist -- and Part 2's answer is that nothing is left:
    every probe of every one of the eight algorithms is decodable.
    """
    assert set(dataset.DECODABLE) == set(zoo.DECODERS)
    assert set(config.ALGORITHMS) == set(dataset.PROJECT)
    assert {(location, type_) for algorithm in config.ALGORITHMS
            for stage, location, type_ in dataset.SPECS[algorithm].values()
            if stage != "input"} <= set(dataset.DECODABLE)


def test_the_edge_inputs_are_what_the_encoders_expect(splits):
    """
    ``dataset.edge_features`` says what an edge box is given and the
    encoders are built from it; ``Split.edge_inputs`` is what actually
    arrives.  They are two lists in two files, so they are asserted equal
    rather than kept equal by hand.
    """
    for algorithm in config.ALGORITHMS:
        split = splits[algorithm]
        names = dataset.edge_features(algorithm)
        assert all(set(one) == set(names) for one in split.edge_inputs)
        assert [name for name, where in zoo.encoded(algorithm)
                if where == "edge"] == list(names)


def test_a_directed_edge_keeps_its_orientation(splits, batches):
    """
    ``dag_shortest_paths`` is the one directed task: a pair is one box, so
    which way it points is an input, and it says what ``A`` says.

    The bit is written under the assumption that ``from_incidence`` fills
    an edge's slots in node order, i.e. that the first message port is the
    lower-indexed endpoint's -- so the assumption is checked against the
    wiring too, since otherwise every direction would flip silently.
    """
    model, batch = build("dag_shortest_paths"), batches["dag_shortest_paths"]
    zoo.check_edges(model.map.compile(batch.diagram), batch)
    split = splits["dag_shortest_paths"]
    matrix = np.asarray(split.inputs["A"])
    for index, (pairs, given) in enumerate(
            zip(split.edges, split.edge_inputs)):
        one, other = pairs[:, 0], pairs[:, 1]
        forward = matrix[index][one, other] > 0
        assert np.array_equal(given["orient"] > 0, forward)
        assert np.array_equal(
            given["A"], np.where(forward, matrix[index][one, other],
                                 matrix[index][other, one]))
        assert not (forward & (matrix[index][other, one] > 0)).any(), \
            "a DAG has no pair joined in both directions"


def test_only_the_probes_asked_for_are_decoded(batches):
    """
    Scoring wants the outputs alone, and the decoders it does not run are
    the hint heads: a prediction is the same tensor either way.
    """
    model, batch = build("bellman_ford"), batches["bellman_ford"]
    with torch.no_grad():
        every, one = model(batch), model(batch, names=["pi"])
    assert set(one) == {"pi"} and set(every) == set(zoo.decoded(
        "bellman_ford"))
    assert torch.equal(every["pi"], one["pi"])


# --- the two clocks --------------------------------------------------------

def path_graph(size: int = 6) -> dataset.Split:
    """
    ``bfs`` on the path ``0 - 1 - ... - n-1`` from node ``0``, written out
    by hand: at step ``t`` the ``t``-hop ball is exactly ``{0, ..., t}``
    and node ``j > 0`` is reached from ``j - 1``.

    No transcription of any algorithm is involved -- the answers are the
    ones a reader can check on the page -- which is what makes it a
    control for the mapping and not a re-test of the reference.
    """
    adjacency = np.eye(size)
    for node in range(size - 1):
        adjacency[node, node + 1] = adjacency[node + 1, node] = 1.0
    reach = np.array([[1.0 * (node <= step) for node in range(size)]
                      for step in range(size)])
    parent = np.array([[float(max(node - 1, 0)) if node <= step else
                        float(node) for node in range(size)]
                       for step in range(size)])
    return dataset.Split(
        "bfs", "path",
        {"pos": np.arange(size)[None] / size,
         "s": np.eye(size)[0][None], "A": adjacency[None] - np.eye(size),
         "adj": adjacency[None]},
        {"reach_h": reach[:, None], "pi_h": parent[:, None]},
        {"pi": parent[-1][None]}, np.array([size]))


def test_a_checkpoint_is_the_algorithm_step_it_says_it_is():
    """
    The one piece of arithmetic the whole protocol hangs on, pinned on a
    case whose answer is computable by hand.

    On a path graph the ``k``-hop ball is ``{0, ..., k}``, so the target
    the ``k``-th checkpoint is supervised on must be that set for ``k + 1``
    -- ``hints[0]`` being the initial condition -- and that checkpoint must
    be read after ``HOPS * (k + 1)`` rounds, since a node reaches a node
    through a box.  Both columns come from ``model.alignment`` and both are
    checked here, because an off-by-one in either is invisible in a loss
    curve: the model half-learns the shifted target and every probe is
    uniformly mediocre.
    """
    split = path_graph(6)
    batch, model = zoo.Batch.of(split), build("bfs", steps=None)
    table = zoo.alignment(zoo.rounds_of(batch.steps))
    assert len(table) == 6
    for step, found, hint in table:
        assert found == zoo.HOPS * (step + 1) and hint == step + 1
        if hint >= batch.steps:  # the last checkpoint answers, it hints not
            assert model.hint_targets(batch, step) == {}
            continue
        truth, alive = model.hint_targets(batch, step)["reach_h"]
        assert bool(alive.all())
        assert truth.tolist() == [
            [1.0 * (node <= hint) for node in range(6)]], step

    with torch.no_grad():
        every = model.run(batch, deep=True)
    assert len(every) == batch.steps == 6
    for step, found, _ in table:
        with torch.no_grad():
            alone = model.map(batch.diagram, model.initial(batch),
                              rounds=found)
        assert torch.equal(every[step], alone), step


def test_the_benchmarks_hints_are_the_hop_balls(splits):
    """
    The other half of the same claim, on the benchmark's own arrays rather
    than on a hand-made one: ``reach_h[t]`` *is* the ``t``-hop ball of the
    sampled graph, computed here by repeated adjacency application, so the
    convention ``model.hint_of`` encodes is the benchmark's and not a
    guess about it.
    """
    split = splits["bfs"]
    adjacency = np.asarray(split.inputs["adj"]) > 0.5
    source = np.asarray(split.inputs["s"]).argmax(-1)
    reach = np.asarray(split.hints["reach_h"]) > 0.5
    for index in range(len(split)):
        ball = np.zeros(split.n, bool)
        ball[source[index]] = True
        for step in range(int(split.lengths[index])):
            assert np.array_equal(reach[step, index], ball), (index, step)
            ball = ball | adjacency[index][ball].any(0)


@pytest.mark.parametrize("algorithm", config.ALGORITHMS)
def test_the_depth_is_the_trajectory(algorithm, splits):
    """
    Part 2's protocol: a run is ``HOPS`` rounds per step of the sampled
    execution, so there is one checkpoint per step of the longest
    trajectory of the batch and the output supervision never clamps --
    which is what makes "reach the answer and stay there" trainable rather
    than hoped for.
    """
    split = splits[algorithm]
    batch = zoo.Batch.of(split)
    model = build(algorithm, steps=None)
    assert model.steps_of(batch) == int(split.lengths.max()) == batch.steps
    assert model.rounds_for(batch) == zoo.HOPS * batch.steps
    assert model.rounds_for(batch, 3.0) == 3 * zoo.HOPS * batch.steps
    with torch.no_grad():
        every = model.run(batch, deep=True)
    settled = (batch.lengths - 1).clamp(max=len(every) - 1)
    assert torch.equal(settled, batch.lengths - 1), "the clamp binds"


@pytest.mark.parametrize("algorithm", config.ALGORITHMS)
def test_forward_of_every_algorithm(algorithm, batches):
    """
    The shared interface: one prediction per decoded *probe*, and one
    checkpoint per *algorithm step*, which is ``HOPS`` rounds.
    """
    model, batch = build(algorithm), batches[algorithm]
    size = batch.size
    with torch.no_grad():
        last = model(batch)
        every = model(batch, deep=True)
    assert len(every) == ROUNDS // zoo.HOPS
    assert set(last) == set(zoo.decoded(algorithm))
    for name, found in last.items():
        # a decoded probe has the shape of its target, plus one axis when
        # it is a softmax over the nodes: a pointer scores every candidate.
        tail = dataset.shape_of(algorithm, name, size)
        if dataset.kind(algorithm, name)[1] == "pointer":
            tail = tail + (size, )
        assert found.shape == (BATCH, ) + tail, name
        assert torch.isfinite(found).all()
    for name in last:
        assert torch.equal(every[-1][name], last[name])


def test_a_deeper_run_is_a_longer_run(batches):
    """
    Test-time compute is a keyword: more rounds is more checkpoints and the
    same weights, and running twice gives the same numbers.
    """
    model = build("bfs")
    with torch.no_grad():
        deep = model(batches["bfs"], deep=True, rounds=3 * ROUNDS)
        again = model(batches["bfs"], rounds=3 * ROUNDS)
    assert len(deep) == 3 * ROUNDS // zoo.HOPS
    for pair, found in again.items():
        assert torch.equal(deep[-1][pair], found)


def test_a_shallow_run_ends_where_the_deep_one_does_only_on_a_boundary():
    """
    The documented caveat of the two clocks: a deep run reads its
    checkpoints at hop boundaries, so an odd number of rounds ends half a
    hop past the last of them.  Every protocol here sweeps multiples of
    ``HOPS``; the assertion is that the caveat is what it says it is.
    """
    assert zoo.HOPS == 2
    split = dataset.load("bfs", "val").subsample(2)
    model, batch = build("bfs"), zoo.Batch.of(split)
    with torch.no_grad():
        odd = model(batch, rounds=2 * zoo.HOPS + 1)
        deep = model(batch, deep=True, rounds=2 * zoo.HOPS + 1)
        even = model(batch, rounds=2 * zoo.HOPS)
    assert len(deep) == 2
    for pair in odd:
        assert not torch.equal(odd[pair], deep[-1][pair])
        assert torch.equal(even[pair], deep[-1][pair])


# --- the solvers, and the cell Part 3's grid cannot have --------------------

def test_a_full_fixed_point_is_an_iterate(batches):
    """
    With no tolerance to stop at and no re-injection, Picard iteration
    *is* finite iteration, bitwise -- so ``backward="full"`` is a rename
    and only ``"last"`` is a row.  ``PART3.md`` rests on this: it is why
    a fixed point cannot be trained with a per-checkpoint hint loss and
    still be a different solver.
    """
    one, other = build("bfs"), build("bfs", solver="fixedpoint",
                                     backward="full")
    other.load_state_dict(one.state_dict())
    with torch.no_grad():
        here = one.run(batches["bfs"], deep=True)
        there = other.run(batches["bfs"], deep=True)
    assert len(here) == len(there)
    assert all(torch.equal(a, b) for a, b in zip(here, there))


def test_a_fixed_point_differentiates_the_terminal_checkpoint_alone(batches):
    """
    The Jacobian-free one-step gradient: every round but the last runs
    under ``no_grad`` and the last is differentiated from the detached
    limit.  Half of why ``FixedPoint`` x ``no-settle`` is an empty cell.
    """
    model = build("bfs", solver="fixedpoint", backward="last")
    found = [state.requires_grad
             for state in model.run(batches["bfs"], deep=True)]
    assert found == [False] * (len(found) - 1) + [True]
    assert build("bfs").run(batches["bfs"], deep=True)[0].requires_grad


def test_the_terminal_checkpoint_needs_a_terminal_settle(batches):
    """
    The other half, and a defect in ``settle`` as Part 2 implemented it.

    A run is ``batch.steps`` checkpoints long and the ``k``-th is
    supervised on hint ``k + 1``, so the last one asks for a hint index
    the batch does not define -- and :meth:`Model.hint_targets` refuses
    it *before* it consults ``settle``.  So a hold that stops at the
    interior trains a basin everywhere except at the state a fixed point
    converges to, which is the one place H2 reads.

    Together with the two tests above: a ``FixedPoint(last)`` arm
    differentiates only the checkpoint that neither ``None`` nor
    ``"interior"`` supervises, so it gets no hint gradient at all and
    ``settle`` is a no-op for it.
    """
    model, batch = build("bfs"), batches["bfs"]
    last = batch.steps - 1
    assert model.hint_targets(batch, last, settle=None) == {}
    assert model.hint_targets(batch, last, settle="interior") == {}
    held = model.hint_targets(batch, last, settle="terminal")
    assert held and all(int(alive.sum()) == len(batch)
                        for _, alive in held.values())
    # and the interior hold is not a no-op where it does reach: it lifts
    # the checkpoint before the last from the survivors to everyone.
    alive = int((last < batch.lengths).sum())
    assert alive < len(batch)
    assert int(model.hint_targets(
        batch, last - 1, settle=None)["reach_h"][1].sum()) == alive
    assert int(model.hint_targets(
        batch, last - 1, settle="interior")["reach_h"][1].sum()) == len(batch)


def test_the_librarys_fixed_point_cannot_train_an_encoder(batches):
    """
    Why ``config.H2_FIXEDPOINT`` names :class:`model.Grounded` and not
    the library's solver.

    ``backward="last"`` differentiates one round from ``state.detach()``.
    That keeps the inputs in the graph when an interaction re-injects
    them; these cells are *resumable* instead, so the inputs ride on
    traced loops inside the state and detaching it detaches them.  Every
    encoder parameter is then a dead parameter, and an arm built on it
    would differ from an ``Iterate`` one in the differentiation policy
    *and* in whether its encoders were trained at all.
    """
    model = build("bfs", solver="fixedpoint", backward="last")
    model.loss(batches["bfs"])[0].backward()
    dead = [name for name, one in model.named_parameters()
            if one.grad is None]
    assert dead and all(name.startswith("encoders.") for name in dead)
    assert set(dead) == {name for name, _ in model.named_parameters()
                         if name.startswith("encoders.")}


def test_a_grounded_fixed_point_is_the_same_forward(batches):
    """
    The repair is free because the roles it writes back are *carried*: a
    site re-emits ``FEAT`` and ``WEIGHT`` unchanged, so the limit already
    holds bitwise what the encoders wrote there and re-attaching them
    changes no number.
    """
    one, other = build("bfs", solver="fixedpoint", backward="last"), \
        build("bfs", solver="grounded")
    other.load_state_dict(one.state_dict())
    with torch.no_grad():
        here = one.run(batches["bfs"], deep=True)
        there = other.run(batches["bfs"], deep=True)
    assert len(here) == len(there)
    assert all(torch.equal(a, b) for a, b in zip(here, there))


def test_a_grounded_fixed_point_trains_its_encoders(batches):
    """ And it differentiates the terminal checkpoint and no other. """
    model = build("bfs", solver="grounded")
    states = model.run(batches["bfs"], deep=True)
    assert [one.requires_grad for one in states] == \
        [False] * (len(states) - 1) + [True]
    model.loss(batches["bfs"])[0].backward()
    for name, one in model.named_parameters():
        assert one.grad is not None, name


def test_a_probe_fits_the_hint_heads_and_leaves_the_interaction(batches):
    """
    What an output-only arm of Part 3 is, and why it is not
    ``hint_weight = 0``.

    Detaching the state the hints are decoded from puts the hint
    decoders back in the graph -- without them the per-head split every
    Part 3 table owes would be read off untrained heads -- while leaving
    the interaction's gradient bitwise alone, which is the axis.
    """
    def gradients(**kwargs):
        model = build("bfs", **kwargs)
        model.loss(batches["bfs"])[0].backward()
        return {name: one.grad for name, one in model.named_parameters()}
    plain, probed = gradients(), gradients(probe=True)
    hints = [name for name in plain if "reach_h" in name]
    assert hints and all(one.any() for one in (plain[name] for name in hints))
    assert all(torch.equal(plain[name], probed[name]) for name in hints)
    assert any(not torch.equal(plain[name], probed[name])
               for name in plain if name.startswith("map."))
    # and under a grounded fixed point run *at the trajectory's depth*
    # the arm is output-only whether or not it says so: every round but
    # the last is outside the graph already, and the last is the
    # checkpoint no hint index reaches.  So `O` has to declare the probe
    # and `F` gets it for free, which is what makes the two comparable.
    # It is a property of the trajectory rule and not of the solver: at
    # Part 1's shorter fixed depth the terminal checkpoint is inside the
    # trajectory, is supervised, and the two do differ.
    bare = gradients(solver="grounded", steps=None)
    asked = gradients(solver="grounded", probe=True, steps=None)
    for name, found in bare.items():
        assert torch.equal(found, asked[name]), name


# --- the loss --------------------------------------------------------------

def test_the_hint_loss_is_masked_by_the_trajectory_length(batches):
    """
    A trajectory of ``length`` steps defines ``hints[0] ... hints[length -
    1]`` and ``hints[0]`` is the initial condition, so the ``k``-th
    checkpoint is supervised only on the trajectories still running -- and
    past the longest one, on none.
    """
    model, batch = build("bfs"), batches["bfs"]
    longest = int(batch.lengths.max())
    for step in range(longest - 1):
        found = model.hint_targets(batch, step)
        assert found, step
        for _, (truth, alive) in found.items():
            assert len(truth) == int(alive.sum()) == \
                int((step + 1 < batch.lengths).sum())
    assert model.hint_targets(batch, batch.steps) == {}


@pytest.mark.parametrize("algorithm", config.ALGORITHMS)
def test_the_loss_reaches_every_parameter(algorithm, batches):
    model, batch = build(algorithm), batches[algorithm]
    loss, parts = model.loss(batch)
    loss.backward()
    assert np.isfinite(loss.item())
    assert set(parts) == {"output", "hint"} | {
        f"probe/{name}" for name in zoo.decoded(algorithm)}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("algorithm", config.ALGORITHMS)
def test_the_loss_is_reported_per_probe(algorithm, batches):
    """
    A total hides a head. The stage terms are the sums of the probe terms
    that feed them, so a probe that is failing while its neighbours carry
    the sum is visible -- which is what a shared decoder was not, for a
    whole campaign; see ``NOTES.md``.
    """
    model, batch = build(algorithm), batches[algorithm]
    _, parts = model.loss(batch)
    stages = {"output": dataset.probes(algorithm, "output"),
              "hint": dataset.probes(algorithm, "hint")}
    for stage, names in stages.items():
        assert parts[stage] == pytest.approx(
            sum(parts[f"probe/{name}"] for name in names), rel=1e-5), stage


def test_only_the_output_is_supervised_without_hints(batches):
    """ ``hint_weight = 0`` is the ablation, and it is the same run. """
    model = build("bfs", hint_weight=0.0)
    loss, parts = model.loss(batches["bfs"])
    assert float(loss.detach()) == pytest.approx(parts["output"], rel=1e-6)


def test_loss_decreases_on_a_fixed_batch(batches):
    """
    A smoke assertion beside the deterministic ones: the model does learn
    something on a batch it sees over and over.
    """
    model, batch = build("bfs"), batches["bfs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(8):
        loss, _ = model.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


# --- the harness -----------------------------------------------------------

def test_train_epoch(splits):
    """ One epoch of the production harness: one optimizer step per batch. """
    model = build("bfs")
    before = [value.detach().clone() for value in model.parameters()]
    batches = zoo.Batches(splits["bfs"], BUDGET.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    stats = training.train_epoch(model, batches, optimizer,
                                 np.random.default_rng(0))
    assert stats["opt_steps"] == len(batches) == BATCH // BUDGET.batch_size
    assert np.isfinite(stats["loss"]) and finite(model)
    assert any(not torch.equal(one, other)
               for one, other in zip(before, model.parameters()))


def test_a_warm_epoch_compiles_nothing(splits):
    """
    A batch *is* a diagram and a diagram is compiled once, so every epoch
    after the first must be all hits. An LRU one diagram too small turns
    that one-off setup cost into a per-epoch one, and the only place the
    difference shows is the wall clock -- hence ``fit_cache``, which sizes
    it from the batches that exist, and ``cache_stats``, which says so.
    """
    model = build("bfs", cache=1)
    batches = zoo.Batches(splits["bfs"], 2)
    assert zoo.fit_cache(model, batches) == len(batches) > 1
    assert zoo.fit_cache(model, batches) == len(batches), "never shrinks"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training.train_epoch(model, batches, optimizer)
    cold = model.map.cache_stats(reset=True)
    assert cold["misses"] == len(batches) == cold["held"]

    training.train_epoch(model, batches, optimizer)
    warm = model.map.cache_stats()
    assert warm["misses"] == 0 and warm["hits"] == cold["hits"] + cold[
        "misses"]


def test_an_artefact_of_another_depth_is_refused(tmp_path, splits,
                                                 monkeypatch):
    """
    A trained artefact is reused by filename, and the two depth regimes
    file under different tags -- but an artefact can predate a protocol,
    which is how a fixed-depth checkpoint once turned up under a
    trajectory-rule name.  Reuse checks what the record says it was trained
    under and refuses the mismatch, because a table that mixes the two is
    not a table.
    """
    monkeypatch.setattr(training, "ARTIFACTS", tmp_path)
    budget = replace(BUDGET, epochs=1)
    given = {"train": splits["bfs"], "val": splits["bfs"]}
    training.train_model("bfs", budget, seed=0, widths=TINY,
                         device=torch.device("cpu"), splits=given,
                         log=lambda _: None)
    path = training.artifact_of("bfs", budget, 0)
    again = training.train_model(
        "bfs", budget, seed=0, widths=TINY, device=torch.device("cpu"),
        splits=given, log=lambda _: None)[1]
    assert again["depth"] == f"fixed:{ROUNDS}"

    stale = torch.load(path, map_location="cpu", weights_only=False)
    stale["depth"] = "trajectory"
    torch.save(stale, path)
    with pytest.raises(ValueError, match="depth"):
        training.train_model("bfs", budget, seed=0, widths=TINY,
                             device=torch.device("cpu"), splits=given,
                             log=lambda _: None)


def test_the_score_of_a_split_is_pooled_not_averaged(splits):
    """
    Scoring a split in one batch and in two gives the same number, which
    is what pooling the predictions buys and what averaging per-batch F1s
    would not.
    """
    model = build("bfs")
    scores = [zoo.evaluate_split(model, zoo.Batches(splits["bfs"], size))
              for size in (BATCH, BATCH // 2)]
    assert scores[0]["score"] == pytest.approx(scores[1]["score"], abs=1e-12)
    assert 0.0 <= scores[0]["score"] <= 1.0


def test_the_depth_sweep_is_bounded_and_reproducible(splits):
    """
    The sweep is a multiple of the trained depth rather than a round
    count: under the trajectory rule the trained depth is the length of
    the sample's own execution, so only a multiple asks the same question
    at two sizes.
    """
    model = build("bellman_ford")
    batches = zoo.Batches(splits["bellman_ford"], BATCH)
    found = evaluations.sweep(model, batches, BUDGET)
    assert list(found) == ["x1", "x2"]
    assert found == evaluations.sweep(model, batches, BUDGET)
    for scores in found.values():
        assert 0.0 <= scores["score"] <= 1.0
    assert found["x2"] == zoo.evaluate_split(
        model, batches, rounds=2 * ROUNDS)


def test_the_hint_curve_is_one_score_per_step(splits):
    """
    Where in the trajectory the imitation comes apart, per probe: the
    diagnostic Part 2 asks for.  One curve per hint probe, one point per
    step, and no point past the longest trajectory of the split.
    """
    model = build("bellman_ford", steps=None)
    batches = zoo.Batches(splits["bellman_ford"], BATCH)
    found = evaluations.hint_curve(model, batches)
    assert set(found) == set(dataset.probes("bellman_ford", "hint"))
    longest = int(splits["bellman_ford"].lengths.max())
    for name, curve in found.items():
        assert 0 < len(curve) <= longest, name
        assert all(np.isfinite(value) for value in curve), name


def test_the_algorithm_settles_where_the_hints_stop_moving():
    """
    The other half of H2's sentence, on the same hand-written case as the
    alignment: on the path ``0 - ... - 5`` from node ``0`` the ball grows
    by one node at every step and therefore changes at every step, so the
    last change is at step ``5`` and the round it is read at is
    ``HOPS * 5 = 10``.  A trajectory whose length is six and whose
    settling step is five is a trajectory that never stops early, which is
    what makes it a control: any measurement that reported convergence
    *before* the end here would be reading the benchmark's padding.
    """
    found = evaluations.settling(zoo.Batches(path_graph(6), 1, "cpu"))
    assert found["steps"] == [5] and found["median"] == 2 * 5 == zoo.HOPS * 5
    assert set(found["per_probe"]) == {"reach_h", "pi_h"}
    assert all(one["rounds"] == [10] for one in found["per_probe"].values())


def test_a_padded_trajectory_does_not_read_as_convergence():
    """
    The benchmark repeats a trajectory's last state to the width of the
    array, so a settling step measured over the whole array would be the
    same number for every sample -- the padding boundary -- rather than a
    property of the execution.  Cutting a path graph's length in half must
    move the answer with it.
    """
    split = path_graph(6)
    cut = dataset.Split(split.algorithm, split.name, split.inputs,
                        split.hints, split.outputs, np.array([3]))
    found = evaluations.settling(zoo.Batches(cut, 1, "cpu"))
    assert found["steps"] == [2], "a length is a length"


# --- what the diagram promises ---------------------------------------------

def test_the_cells_keep_the_equations_their_signatures_declare():
    """
    Every cell pools its message orbit symmetrically, so permutation
    equivariance holds up to the reordering of a float64 reduction.  The
    residual is measured, not assumed.
    """
    model = build("bfs")
    found = evaluations.equivariance(model, TINY)
    assert set(found) == {"node", "edge", "readout"}
    for name, residuals in found.items():
        assert residuals, name
        assert max(residuals.values()) < 1e-9, name


def test_the_equivariance_residual_is_exactly_zero_under_max():
    """
    H4's independent variable, and why H4 as written cannot be asked.

    A ``max`` is order-invariant *in floating point*, so under the
    aggregator the primary campaign uses the equivariance law is strict
    rather than lax: the residual is not small, it is zero on every cell
    of every seed, and a variable with no variance correlates with
    nothing.  Under ``mean`` it is machine epsilon over the width of a
    reduction -- a fact about orbit sizes, not about learned weights.
    """
    for pool, exact in (("max", True), ("mean", False)):
        found = evaluations.equivariance(build("bfs", pool=pool), TINY)
        residuals = [one for cell in found.values() for one in cell.values()]
        assert residuals
        assert all(one == 0.0 for one in residuals) is exact, pool
    assert evaluations.correlate([0.0] * 8, list(range(8)))["r"] is None


@pytest.mark.parametrize("algorithm", config.ALGORITHMS)
def test_every_output_probe_selects_over_the_node_set(algorithm):
    """
    The measurement that forces H4's amendment: every algorithm of this
    study has exactly one output probe, and on all eight it is a
    ``pointer`` or a ``mask_one``.  So the benchmark's own micro-F1 --
    and both published anchors -- is 100 % order-dependent mass, an
    *order-free output* drop is an empty column on every row, and there
    is no order-dependent output mass to partial out because it is a
    constant.  ``PART3.md`` repairs both by scoring over the hints too.
    """
    assert len(dataset.probes(algorithm, "output")) == 1
    assert evaluations.head_mass(algorithm)["output"] == {
        "free": 0.0, "dependent": 1.0, "unpooled": 0.0}


@pytest.mark.parametrize("algorithm", config.ALGORITHMS)
def test_the_head_split_covers_every_probe_once(algorithm):
    """
    Part 3's second rule is only a rule if the partition is total: a
    probe that fell out of every class would be mass nobody reported.
    """
    found = evaluations.heads(algorithm)
    every = [name for names in found.values() for name in names]
    assert sorted(every) == sorted(zoo.decoded(algorithm))
    assert len(every) == len(set(every))


@pytest.mark.parametrize("algorithm", config.ALGORITHMS)
def test_an_edge_is_an_orbit_wherever_the_sampler_is_undirected(algorithm):
    """
    Which of the two signatures :func:`model.edge` gives a task is read off
    ``dataset.DIRECTED`` and nowhere else, so ``Sym.NONE`` is claimed by
    ``dag_shortest_paths`` alone and the other six edge cells are orbits
    that owe an equation.  Owing it is not enough -- H4 correlates a
    residual against a generalization drop, so there has to *be* a
    residual -- and this asserts that every one of them is in the set
    :func:`evaluate.equivariance` measures, rather than skipped for being
    a cell the example wrote itself.

    ``minimum`` is the eighth task and draws no edges at all, so it has no
    edge cell to ask about; that it has none is the assertion there.
    """
    directed = algorithm in dataset.DIRECTED
    assert zoo.edge(directed).orbits[0].sym \
        == (zoo.Sym.NONE if directed else zoo.Sym.PERM)
    found = evaluations.equivariance(build(algorithm), TINY)
    if not zoo.has_edges(algorithm):
        assert algorithm == "minimum" and "edge" not in found
        return
    assert "edge" in found, "an edge cell is measured like any other"
    assert (found["edge"] == {}) is directed
    if not directed:
        assert max(found["edge"].values()) < 1e-9


def test_a_directed_cell_declares_that_it_owes_nothing():
    """
    ``dag_shortest_paths`` reads its two legs as source and target, so its
    edge signature is ``Sym.NONE``, its group has no generators and its
    residual is the *empty* dictionary rather than a zero.  An honest
    signature is one no equation is owed against -- and the node and
    readout cells of the same model still owe theirs.
    """
    model = build("dag_shortest_paths")
    found = evaluations.equivariance(model, TINY)
    assert found["edge"] == {}, "a directed edge is not an orbit"
    assert found["node"] and found["readout"]
    assert max(found["node"].values()) < 1e-9

    with torch.no_grad():
        cell = model.map.ar["edge"]
        width = zoo.edge(True).width(zoo.widths_of(zoo.graph_ob(TINY)))
        one = torch.randn(3, width)
        legs = TINY.dim
        other = torch.cat([one[:, legs:2 * legs], one[:, :legs],
                           one[:, 2 * legs:]], -1)
    assert not torch.allclose(
        cell(one)[:, :legs], cell(other)[:, legs:2 * legs]), \
        "the directed cell answers its endpoints the same way"


def test_the_residual_is_a_number_not_a_promise(splits):
    """
    :meth:`Interaction.residual` is reported, never assumed to be zero: an
    untrained map does not settle and the study says so.
    """
    model = build("bfs")
    found = evaluations.residuals(
        model, zoo.Batches(splits["bfs"], BATCH))
    assert np.isfinite(found["max"]) and found["max"] >= found["mean"] >= 0.0


def test_the_residual_curve_is_one_number_per_round(splits):
    """
    The curve H2 will read: one residual per round, run past the trained
    depth, and its last point is the scalar :func:`residuals` reports at
    that depth.
    """
    model, batches = build("bfs"), zoo.Batches(splits["bfs"], BATCH)
    curve = evaluations.residual_curve(model, batches, factor=3.0)
    assert len(curve) == 3 * ROUNDS
    assert all(np.isfinite(value) and value >= 0.0 for value in curve)
    end = evaluations.residuals(model, batches, factor=3.0)
    assert curve[-1] == pytest.approx(end["mean"], rel=1e-6)


def test_the_wide_split_is_reported_with_an_interval(splits):
    """
    The primary number of every table: the mean over trajectories of the
    per-trajectory score, with the standard error a table of two rows
    needs before it can call a difference a difference.
    """
    model = build("bfs")
    batches = zoo.Batches(splits["bfs"], 2)
    found = evaluations.interval(model, batches)
    assert found["trajectories"] == BATCH
    assert 0.0 <= found["mean"] <= 1.0
    assert found["half_width"] == pytest.approx(1.96 * found["std_error"])
    pooled = zoo.evaluate_split(model, batches)["score"]
    assert found["mean"] == pytest.approx(pooled, abs=0.5), \
        "the two averages differ only by the non-linearity of an F1"


def test_max_pooling_is_a_knob_of_the_recipe(batches):
    """
    ``build(pool=...)`` reaches every cell.  It is the architectural knob
    a change of size can see -- a mean rescales with a node's degree, a
    max does not -- so a campaign fixes it and files its weights under a
    tag of its own; see :attr:`config.Budget.tag`.
    """
    model = build("bfs", pool="max")
    assert {name: cell.pooling for name, cell in model.map.ar.items()} == {
        name: cells.POOL["max"] for name in ("node", "edge", "readout")}
    with torch.no_grad():
        found = model(batches["bfs"])["pi"]
    assert torch.isfinite(found).all()
    assert replace(config.FULL, pool="max").tag == "full-max"


# --- a checkpoint is the model ---------------------------------------------

def test_checkpoint_round_trip(tmp_path, batches):
    """
    Saving and loading a state dict reproduces the predictions bitwise,
    which is what makes a trained artifact worth keeping.
    """
    trained, batch = build("bellman_ford"), batches["bellman_ford"]
    optimizer = torch.optim.Adam(trained.parameters(), lr=1e-2)
    loss, _ = trained.loss(batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.save(trained.state_dict(), tmp_path / "model.pt")

    fresh = build("bellman_ford", seed=1)
    with torch.no_grad():
        assert not torch.equal(fresh(batch)["pi"], trained(batch)["pi"])
    zoo.load_checkpoint(fresh, tmp_path / "model.pt")
    with torch.no_grad():
        for name, found in trained(batch).items():
            assert torch.equal(fresh(batch)[name], found), name
