# -*- coding: utf-8 -*-

"""
The benchmark: CLRS-30 trajectories, generated once and cached as arrays.

    python dataset.py --generate          # needs the `clrs` package
    python dataset.py --check             # needs numpy and nothing else

A sample is a whole *execution* of an algorithm: its inputs, the hint at
every step of the trajectory, how many steps that trajectory took, and its
outputs.  :data:`~config.CLRS30` says which samples -- the benchmark's own
counts, lengths and seeds -- so what is cached here is the benchmark's
training set and not a re-draw of its distribution.

Two environments, one cache
---------------------------

Generation imports :mod:`clrs`, which brings jax, haiku and tensorflow with
it; training imports :mod:`torch`.  Rather than ask one environment to hold
both, the ``npz`` cache is the interface: :func:`generate` writes it with
``clrs`` in scope and everything else reads it with numpy alone.  The price
is that the *spec* -- which probe is a scalar, which a pointer, where each
lives -- has to be written down here rather than imported, so it is, in
:data:`SPECS`, verbatim from ``clrs._src.specs.SPECS``;
:func:`check_spec` asserts the copy against the original whenever ``clrs``
is importable, and :func:`check` re-derives every cached output from a
transcription of the reference algorithm, so nothing is trusted twice.

What a graph sample is
----------------------

Every graph algorithm of CLRS-30 hands the model a dense adjacency matrix
``adj`` and a dense weight matrix ``A``.  :attr:`Split.edges` turns that
into the edge list of a diagram, and which edges those are is a property of
what the algorithm *answers* rather than of what it was sampled from:

* most read the graph off ``adj``, off its diagonal, as an undirected edge
  list -- CLRS's own samplers build all of them but one with
  ``directed=False``, so a symmetric edge is the honest encoding and no
  symmetrisation is needed.  (``project.md`` expects to have to symmetrise
  ``bellman_ford``; :func:`check` shows the expectation was wrong, which is
  why the note is here and not in a comment.)
* :data:`DIRECTED` -- ``dag_shortest_paths`` alone -- has an asymmetric
  ``adj``.  A pair is still *one* box, with the orientation on its carried
  input, since two boxes per pair would double the wiring to say what one
  bit says.
* :data:`DENSE` -- ``floyd_warshall`` and ``matrix_chain_order`` -- get
  **every** pair, because they answer about every pair.  The sampled graph
  is then an input (``adj``) rather than the wiring, and the diagram
  depends on the size alone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from config import ALGORITHMS, CLRS30, DATA_DIR, MIXED, SPLITS, WIDE

#: The probes of each algorithm, verbatim from ``clrs._src.specs.SPECS``:
#: ``name -> (stage, location, type)``.  Copied rather than imported so
#: that a training environment needs no ``clrs``; :func:`check_spec`
#: asserts the copy.
SPECS = {
    "minimum": {
        "pos": ("input", "node", "scalar"),
        "key": ("input", "node", "scalar"),
        "min": ("output", "node", "mask_one"),
        "pred_h": ("hint", "node", "pointer"),
        "min_h": ("hint", "node", "mask_one"),
        "i": ("hint", "node", "mask_one"),
    },
    "bfs": {
        "pos": ("input", "node", "scalar"),
        "s": ("input", "node", "mask_one"),
        "A": ("input", "edge", "scalar"),
        "adj": ("input", "edge", "mask"),
        "pi": ("output", "node", "pointer"),
        "reach_h": ("hint", "node", "mask"),
        "pi_h": ("hint", "node", "pointer"),
    },
    "bellman_ford": {
        "pos": ("input", "node", "scalar"),
        "s": ("input", "node", "mask_one"),
        "A": ("input", "edge", "scalar"),
        "adj": ("input", "edge", "mask"),
        "pi": ("output", "node", "pointer"),
        "pi_h": ("hint", "node", "pointer"),
        "d": ("hint", "node", "scalar"),
        "msk": ("hint", "node", "mask"),
    },
    "dijkstra": {
        "pos": ("input", "node", "scalar"),
        "s": ("input", "node", "mask_one"),
        "A": ("input", "edge", "scalar"),
        "adj": ("input", "edge", "mask"),
        "pi": ("output", "node", "pointer"),
        "pi_h": ("hint", "node", "pointer"),
        "d": ("hint", "node", "scalar"),
        "mark": ("hint", "node", "mask"),
        "in_queue": ("hint", "node", "mask"),
        "u": ("hint", "node", "mask_one"),
    },
    "mst_prim": {
        "pos": ("input", "node", "scalar"),
        "s": ("input", "node", "mask_one"),
        "A": ("input", "edge", "scalar"),
        "adj": ("input", "edge", "mask"),
        "pi": ("output", "node", "pointer"),
        "pi_h": ("hint", "node", "pointer"),
        "key": ("hint", "node", "scalar"),
        "mark": ("hint", "node", "mask"),
        "in_queue": ("hint", "node", "mask"),
        "u": ("hint", "node", "mask_one"),
    },
    "dag_shortest_paths": {
        "pos": ("input", "node", "scalar"),
        "s": ("input", "node", "mask_one"),
        "A": ("input", "edge", "scalar"),
        "adj": ("input", "edge", "mask"),
        "pi": ("output", "node", "pointer"),
        "pi_h": ("hint", "node", "pointer"),
        "d": ("hint", "node", "scalar"),
        "mark": ("hint", "node", "mask"),
        "topo_h": ("hint", "node", "pointer"),
        "topo_head_h": ("hint", "node", "mask_one"),
        "color": ("hint", "node", "categorical"),
        "s_prev": ("hint", "node", "pointer"),
        "u": ("hint", "node", "mask_one"),
        "v": ("hint", "node", "mask_one"),
        "s_last": ("hint", "node", "mask_one"),
        "phase": ("hint", "graph", "mask"),
    },
    "floyd_warshall": {
        "pos": ("input", "node", "scalar"),
        "A": ("input", "edge", "scalar"),
        "adj": ("input", "edge", "mask"),
        "Pi": ("output", "edge", "pointer"),
        "Pi_h": ("hint", "edge", "pointer"),
        "D": ("hint", "edge", "scalar"),
        "msk": ("hint", "edge", "mask"),
        "k": ("hint", "node", "mask_one"),
    },
    "matrix_chain_order": {
        "pos": ("input", "node", "scalar"),
        "p": ("input", "node", "scalar"),
        "s": ("output", "edge", "pointer"),
        "pred_h": ("hint", "node", "pointer"),
        "m": ("hint", "edge", "scalar"),
        "s_h": ("hint", "edge", "pointer"),
        "msk": ("hint", "edge", "mask"),
    },
}

#: What a run may do to the ``pos`` input; see
#: :meth:`Split.repositioned`.  ``"sampler"`` is the benchmark's own.
POS = ("sampler", "shuffled", "uniform")

#: The value CLRS writes where a probe is not defined, ``OutputClass.MASKED``.
MASKED = -1

#: The algorithms whose sampler draws a **directed** graph, i.e. whose
#: ``adj`` is not symmetric.  ``python dataset.py --survey`` reads this off
#: ``clrs`` itself and :func:`check` asserts it on every cached split:
#: exactly one of the eight is directed, which is why the directed-edge
#: cell buys one algorithm and not the shape of Part 2.
DIRECTED = ("dag_shortest_paths", )

#: The algorithms whose diagram is the **complete** graph on the nodes
#: rather than the sampled one.  Both decode an ``n x n`` edge probe --
#: ``Pi`` over all pairs, ``s`` over all intervals -- so a pair that the
#: sampled graph does not join still has an answer, and a model with one
#: box per *sampled* edge would have nowhere to keep it.  This is the
#: diagram H1 is asked on: one recurrent state per pair, or none.
DENSE = ("floyd_warshall", "matrix_chain_order")

#: The number of classes of each ``categorical`` probe.  CLRS one-hot
#: encodes them, so the array carries a trailing class axis that no other
#: node probe has; ``check`` asserts the width against this.
CLASSES = {("dag_shortest_paths", "color"): 3}


def probes(algorithm: str, stage: str) -> tuple[str, ...]:
    """
    The probes of one stage of one algorithm, in the spec's order.

    Parameters:
        algorithm : The algorithm.
        stage : ``"input"``, ``"hint"`` or ``"output"``.

    Example
    -------
    >>> probes("bfs", "output"), probes("bfs", "hint")
    (('pi',), ('reach_h', 'pi_h'))
    """
    return tuple(name for name, (where, _, _) in SPECS[algorithm].items()
                 if where == stage)


def kind(algorithm: str, name: str) -> tuple[str, str]:
    """ The ``(location, type)`` of a probe. """
    _, location, type_ = SPECS[algorithm][name]
    return location, type_


def complete(size: int) -> np.ndarray:
    """
    Every pair ``i < j``, as an ``(n (n - 1) / 2, 2)`` array: the edge list
    of a :data:`DENSE` algorithm's diagram.

    It depends on the size alone, so two samples of one size draw the same
    diagram and a whole split compiles once.

    Example
    -------
    >>> complete(3).tolist()
    [[0, 1], [0, 2], [1, 2]]
    """
    one, other = np.triu_indices(size, k=1)
    return np.stack([one, other], -1).astype(np.int64)


def shape_of(algorithm: str, name: str, size: int) -> tuple[int, ...]:
    """
    The shape of one trajectory's array of a probe, its leading sample and
    step axes aside: what its *location* says, plus a class axis for a
    ``categorical``, which CLRS one-hot encodes.

    Example
    -------
    >>> shape_of("bfs", "pi", 16), shape_of("floyd_warshall", "D", 16)
    ((16,), (16, 16))
    >>> shape_of("dag_shortest_paths", "color", 16)
    (16, 3)
    >>> shape_of("dag_shortest_paths", "phase", 16)
    ()
    """
    location, type_ = kind(algorithm, name)
    if location == "graph":
        return ()
    if location == "edge":
        return (size, size)
    if type_ == "categorical":
        return (size, CLASSES[algorithm, name])
    return (size, )


def edge_features(algorithm: str) -> tuple[str, ...]:
    """
    What every edge box of an algorithm is given, in encoder order; the
    keys of :attr:`Split.edge_inputs`.

    Example
    -------
    >>> edge_features("bfs"), edge_features("dag_shortest_paths")
    (('A',), ('A', 'orient'))
    >>> edge_features("floyd_warshall"), edge_features("matrix_chain_order")
    (('A', 'adj'), ())
    """
    found = ["A"] if "A" in SPECS[algorithm] else []
    if algorithm in DIRECTED:
        found.append("orient")
    if algorithm in DENSE and "adj" in SPECS[algorithm]:
        found.append("adj")
    return tuple(found)


# --- a split ---------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    """
    A stack of trajectories of one algorithm, as CLRS produced them.

    Parameters:
        algorithm : The algorithm the trajectories execute.
        name : What the split is, e.g. ``"train"`` or ``"test"``.
        inputs : Per input probe, an array of shape ``(samples, n)`` for a
                 node probe and ``(samples, n, n)`` for an edge one.
        hints : Per hint probe, the same with a leading step axis.
        outputs : Per output probe, as ``inputs``.
        lengths : The number of steps of each trajectory, i.e. how many
                  entries of ``hints`` are defined.

    Example
    -------
    >>> split = Split("bfs", "toy",
    ...               {"pos": np.zeros((2, 4)), "adj": np.ones((2, 4, 4))},
    ...               {"reach_h": np.zeros((3, 2, 4))},
    ...               {"pi": np.zeros((2, 4))}, np.array([3, 2]))
    >>> len(split), split.n, split.steps
    (2, 4, 3)
    """

    algorithm: str
    name: str
    inputs: dict
    hints: dict
    outputs: dict
    lengths: np.ndarray

    def __len__(self) -> int:
        return len(self.lengths)

    @property
    def n(self) -> int:
        """ The number of nodes of every trajectory of the split. """
        return next(iter(self.inputs.values())).shape[1]

    @property
    def steps(self) -> int:
        """ The length of the longest trajectory, i.e. the hint axis. """
        return next(iter(self.hints.values())).shape[0]

    def take(self, indices) -> Split:
        """
        The sub-split of a set of trajectory indices.

        Parameters:
            indices : The trajectories to keep.
        """
        indices = np.asarray(indices)
        return Split(
            self.algorithm, self.name,
            {key: value[indices] for key, value in self.inputs.items()},
            {key: value[:, indices] for key, value in self.hints.items()},
            {key: value[indices] for key, value in self.outputs.items()},
            self.lengths[indices])

    def subsample(self, count: int = None) -> Split:
        """ The first ``count`` trajectories, or all of them. """
        if count is None or count >= len(self):
            return self
        return self.take(np.arange(count))

    def repositioned(self, mode: str, seed: int = 0) -> Split:
        """
        The same split with a different ``pos`` input.

        The benchmark's ``pos`` is ``arange(n) / n``.  Two things are true
        of it at once and they must not be confused, which is what
        :data:`POS` exists to keep apart:

        * its **spacing** is ``1 / n``, so a value means a different node
          at a different size -- the size-dependent liability;
        * its **rank** is the node index, and CLRS's reference algorithms
          iterate in index order, so the *labels* are tie-broken by it.
          ``_bfs`` assigns ``parent[j] = i`` for the **first** reached
          neighbour ``i``, and at ``n = 64`` 69.7 % of assignments have
          more than one candidate.  Rank is therefore not a nuisance
          feature, it is part of the task's definition.

        ``"shuffled"`` permutes the values, which destroys both: among
        ``k`` tied candidates no model of any capacity can beat ``1 / k``,
        so the task becomes partially *unrealizable* rather than merely
        harder.  ``"uniform"`` samples fresh values, sorts them and
        assigns them in index order, which destroys the spacing and keeps
        the rank -- the task stays realizable and only the size-dependent
        part is gone.  That is what a randomised position scalar means in
        the reference, and the difference between the two is the
        difference between an ablation and a broken label.

        Parameters:
            mode : A key of :data:`POS`.
            seed : The generator's seed.

        Example
        -------
        >>> split = read("bfs", "val")
        >>> ranks = lambda one: np.argsort(np.argsort(one.inputs["pos"]))
        >>> shuffled = split.repositioned("shuffled")
        >>> bool((ranks(shuffled) == ranks(split)).all())
        False
        >>> uniform = split.repositioned("uniform")
        >>> bool((ranks(uniform) == ranks(split)).all())
        True
        >>> bool((uniform.inputs["pos"] == split.inputs["pos"]).all())
        False
        """
        generator = np.random.default_rng(seed)
        pos = np.asarray(self.inputs["pos"]).copy()
        for index in range(len(pos)):
            if mode == "shuffled":
                pos[index] = pos[index][generator.permutation(pos.shape[1])]
            elif mode == "uniform":
                pos[index] = np.sort(generator.random(pos.shape[1]))
            else:
                raise ValueError(f"{mode} is not one of {sorted(POS)}")
        return Split(self.algorithm, self.name, {**self.inputs, "pos": pos},
                     self.hints, self.outputs, self.lengths)

    def batches(self, size: int) -> list:
        """
        The trajectories cut into batches of ``size``, in order.

        A batch is run as **one diagram** -- the disjoint union of its
        members' graphs -- so it is compiled once and reused every epoch.
        That is why the batching is fixed rather than reshuffled: a
        reshuffle would draw a fresh diagram, and a diagram costs a
        compilation.

        Parameters:
            size : The trajectories per batch.
        """
        return [self.take(np.arange(start, min(start + size, len(self))))
                for start in range(0, len(self), size)]

    @cached_property
    def edges(self) -> tuple:
        """
        The edge list of each trajectory, as an integer array of shape
        ``(m, 2)`` with ``i < j``: one entry per box of the diagram.

        Three cases, and which one an algorithm is in is a property of what
        it *decodes*, not of what it was sampled from:

        * a :data:`DENSE` algorithm gets **every** pair, because it answers
          about every pair; the list is then the same for every sample of a
          size, which is what makes its diagram compile once for a whole
          split rather than once per batch;
        * a graph algorithm gets the edges of ``adj`` off its diagonal,
          symmetrised for the :data:`DIRECTED` one -- a directed edge is
          still one box, with its orientation on the carry;
        * a non-graph algorithm has no edges at all: its nodes meet only in
          the readout relation, which is the whole point of including one.
        """
        if self.algorithm in DENSE:
            return tuple(complete(self.n) for _ in range(len(self)))
        if "adj" not in self.inputs:
            return tuple(np.zeros((0, 2), dtype=np.int64)
                         for _ in range(len(self)))
        found = np.asarray(self.inputs["adj"]) > 0.5
        if self.algorithm in DIRECTED:
            found = found | np.transpose(found, (0, 2, 1))
        return tuple(np.stack(np.nonzero(one), -1).astype(np.int64)
                     for one in np.triu(found, k=1))

    @cached_property
    def edge_inputs(self) -> tuple:
        """
        Per trajectory, what every edge box is given, in :attr:`edges`
        order: a dictionary from feature name to an ``(m, )`` array.

        ``A`` is the weight, read in whichever direction carries it; a
        directed algorithm adds ``orient``, the bit that says whether the
        lower-indexed endpoint is the source, which is the only place the
        orientation of an edge can live once the pair is one box; a dense
        algorithm adds ``adj``, because on a complete-graph diagram the
        boxes no longer say where the sampled graph is -- the one place
        Part 1's "``adj`` is already in the wiring" stops holding.
        """
        matrix = self.inputs.get("A")
        found: list = []
        for index, pairs in enumerate(self.edges):
            one, other = pairs[:, 0], pairs[:, 1]
            given: dict = {}
            if matrix is not None:
                weights = np.asarray(matrix)[index]
                given["A"] = weights[one, other] + (
                    weights[other, one] if self.algorithm in DIRECTED
                    else 0.0)
                if self.algorithm in DIRECTED:
                    given["orient"] = np.where(
                        weights[one, other] > 0, 1.0, -1.0)
            if self.algorithm in DENSE and "adj" in self.inputs:
                given["adj"] = np.asarray(
                    self.inputs["adj"])[index][one, other]
            found.append({key: value.astype(np.float64)
                          for key, value in given.items()})
        return tuple(found)

    @cached_property
    def weights(self) -> tuple:
        """ The weight ``A`` of each edge, in :attr:`edges` order. """
        return tuple(
            given.get("A", np.zeros(len(pairs), dtype=np.float64))
            for given, pairs in zip(self.edge_inputs, self.edges))


# --- generation, with the `clrs` package -----------------------------------

def check_spec(log=print) -> None:
    """
    Assert :data:`SPECS` against ``clrs._src.specs.SPECS``.

    The copy exists so that a training environment needs no ``clrs``; this
    is what stops it from drifting, and it runs on every generation.
    """
    from clrs._src import specs
    for algorithm, spec in SPECS.items():
        theirs = specs.SPECS[algorithm]
        assert list(theirs) == list(spec), f"{algorithm}: probes differ"
        for name, (stage, location, type_) in spec.items():
            assert tuple(str(part) for part in theirs[name]) == \
                (stage, location, type_), f"{algorithm}.{name}: kind differs"
    log(f"  spec: {len(SPECS)} algorithms match clrs")


def spec_of(split: str) -> dict:
    """
    What the sampler is asked for, per split name.

    A ``trainN`` is one size of the mixed training split: the 1000
    trajectories of ``CLRS30["train"]`` divided evenly between
    :data:`~config.MIXED`, so the *budget* is the benchmark's and only its
    shape changes.  Each carries a seed of its own, offset by the size, so
    that two sizes are not the same graphs relabelled.

    Example
    -------
    >>> spec_of("train")["length"], spec_of("train")["num_samples"]
    (16, 1000)
    >>> spec_of("train8") == {"num_samples": 200, "length": 8, "seed": 9}
    True
    """
    if split in CLRS30:
        return CLRS30[split]
    if split == "wide":
        return WIDE
    size = int(split[len("train"):])
    return {"num_samples": CLRS30["train"]["num_samples"] // len(MIXED),
            "length": size, "seed": CLRS30["train"]["seed"] + size}


def generate(algorithm: str, split: str, log=print) -> Split:
    """
    One split of one algorithm, straight from the benchmark's sampler.

    ``Sampler.next()`` without a batch size returns the whole pre-generated
    pool, so what comes out is exactly the ``num_samples`` trajectories
    :data:`~config.CLRS30` asks for, at its length and under its seed --
    not a fresh draw of the same distribution.

    Parameters:
        algorithm : The algorithm to sample.
        split : ``"train"``, ``"val"``, ``"test"`` or ``"wide"``.
        log : Where to print progress.
    """
    import clrs
    setup = spec_of(split)
    sampler, _ = clrs.build_sampler(
        algorithm, num_samples=setup["num_samples"],
        length=setup["length"], seed=setup["seed"])
    feedback = sampler.next()
    features = feedback.features
    found = Split(
        algorithm, split,
        {point.name: np.asarray(point.data) for point in features.inputs},
        {point.name: np.asarray(point.data) for point in features.hints},
        {point.name: np.asarray(point.data) for point in feedback.outputs},
        np.asarray(features.lengths).astype(np.int64))
    log(f"  {algorithm}/{split}: {len(found)} trajectories, "
        f"n = {found.n}, up to {found.steps} steps")
    return found


# --- caching ---------------------------------------------------------------

def path_of(algorithm: str, split: str):
    """ Where a split is cached. """
    return DATA_DIR / f"{algorithm}-{split}.npz"


def save(split: Split) -> None:
    """ Cache a split as one ``npz``, one array per probe. """
    arrays = {"lengths": split.lengths}
    for stage, group in (("input", split.inputs), ("hint", split.hints),
                         ("output", split.outputs)):
        for name, value in group.items():
            arrays[f"{stage}__{name}"] = np.asarray(value, dtype=np.float32)
    np.savez_compressed(path_of(split.algorithm, split.name), **arrays)


def read(algorithm: str, split: str) -> Split:
    """ A cached split, or ``None`` when it has not been generated. """
    path = path_of(algorithm, split)
    if not path.exists():
        return None
    stored = np.load(path)
    groups: dict = {"input": {}, "hint": {}, "output": {}}
    for key in stored.files:
        if key == "lengths":
            continue
        stage, name = key.split("__", 1)
        groups[stage][name] = stored[key]
    return Split(algorithm, split, groups["input"], groups["hint"],
                 groups["output"], stored["lengths"].astype(np.int64))


def load(algorithm: str, split: str) -> Split:
    """
    A cached split, with the message that builds it when it is missing.

    Parameters:
        algorithm : The algorithm.
        split : The split name.
    """
    found = read(algorithm, split)
    if found is None:
        raise FileNotFoundError(
            f"{path_of(algorithm, split)} is missing; run "
            f"`python dataset.py --generate` in an environment with the "
            f"`clrs` package (`pip install dm-clrs`)")
    return found


def load_all(algorithm: str) -> dict:
    """ Every split of one algorithm. """
    return {split: load(algorithm, split) for split in SPLITS}


# --- the reference algorithms, for verification ----------------------------

def reference(algorithm: str, split: Split, index: int) -> dict:
    """
    The outputs of a trajectory, recomputed from its inputs by a
    transcription of the CLRS reference implementation, as
    ``{probe: array}``.

    This is what makes the cache trustworthy without ``clrs`` in scope: the
    arrays are not merely well-shaped, they are the answer to the question
    the inputs ask, decided again by the algorithm that was sampled.  Every
    branch below is ``clrs._src.algorithms`` line by line, tie-breaking
    included, because a shortest-path *tree* is not determined by the
    distances alone: two parents of equal cost are two different correct
    answers and only one of them is the label.

    Parameters:
        algorithm : The algorithm.
        split : The split the trajectory belongs to.
        index : Which trajectory.
    """
    if algorithm == "minimum":
        return {"min": int(np.argmin(split.inputs["key"][index]))}
    if algorithm == "matrix_chain_order":
        return {"s": _matrix_chain_order(
            np.asarray(split.inputs["p"][index], dtype=np.float64))}
    matrix = np.asarray(split.inputs["A"][index], dtype=np.float64)
    if algorithm == "floyd_warshall":
        return {"Pi": _floyd_warshall(matrix)}
    source = int(np.argmax(split.inputs["s"][index]))
    if algorithm == "bfs":
        return {"pi": _bfs(matrix, source)}
    if algorithm == "dijkstra":
        return {"pi": _extract_min(matrix, source, prim=False)}
    if algorithm == "mst_prim":
        return {"pi": _extract_min(matrix, source, prim=True)}
    if algorithm == "dag_shortest_paths":
        return {"pi": _dag_shortest_paths(matrix, source)}
    return {"pi": _bellman_ford(matrix, source)}


def _bfs(matrix: np.ndarray, source: int) -> np.ndarray:
    """ ``clrs._src.algorithms.graphs.bfs``. """
    size = len(matrix)
    reach, parent = np.zeros(size), np.arange(size)
    reach[source] = 1
    while True:
        before = np.copy(reach)
        for i in range(size):
            for j in range(size):
                if matrix[i, j] > 0 and before[i] == 1:
                    if parent[j] == j and j != source:
                        parent[j] = i
                    reach[j] = 1
        if np.all(reach == before):
            return parent


def _bellman_ford(matrix: np.ndarray, source: int) -> np.ndarray:
    """ ``clrs._src.algorithms.graphs.bellman_ford``. """
    size = len(matrix)
    distance, parent = np.zeros(size), np.arange(size)
    mask = np.zeros(size)
    mask[source] = 1
    while True:
        before, seen = np.copy(distance), np.copy(mask)
        for i in range(size):
            for j in range(size):
                if seen[i] == 1 and matrix[i, j] != 0:
                    if mask[j] == 0 or before[i] + matrix[i, j] < distance[j]:
                        distance[j] = before[i] + matrix[i, j]
                        parent[j] = i
                    mask[j] = 1
        if np.all(distance == before):
            return parent


def _extract_min(matrix: np.ndarray, source: int, prim: bool) -> np.ndarray:
    """
    ``clrs._src.algorithms.graphs.dijkstra`` and ``mst_prim``, which are
    one loop apart: the key a node is queued under is the distance from
    the source for the first and the weight of the edge that reached it
    for the second.
    """
    size = len(matrix)
    key, mark = np.zeros(size), np.zeros(size)
    queued, parent = np.zeros(size), np.arange(size)
    queued[source] = 1
    for _ in range(size):
        u = int(np.argsort(key + (1.0 - queued) * 1e9)[0])
        if queued[u] == 0:
            break
        mark[u], queued[u] = 1, 0
        for v in range(size):
            if matrix[u, v] != 0:
                found = matrix[u, v] if prim else key[u] + matrix[u, v]
                if mark[v] == 0 and (queued[v] == 0 or found < key[v]):
                    parent[v], key[v], queued[v] = u, found, 1
    return parent


def _floyd_warshall(matrix: np.ndarray) -> np.ndarray:
    """ ``clrs._src.algorithms.graphs.floyd_warshall``. """
    size = len(matrix)
    distance, mask = np.copy(matrix), (matrix > 0).astype(np.float64)
    mask[np.arange(size), np.arange(size)] = 1.0
    parent = np.repeat(np.arange(size)[:, None], size, axis=1).astype(float)
    for k in range(size):
        before, seen = np.copy(distance), np.copy(mask)
        for i in range(size):
            for j in range(size):
                if seen[i, k] > 0 and seen[k, j] > 0:
                    if mask[i, j] == 0 or \
                            before[i, k] + before[k, j] < distance[i, j]:
                        distance[i, j] = before[i, k] + before[k, j]
                        parent[i, j] = parent[k, j]
                    else:
                        distance[i, j] = before[i, j]
                    mask[i, j] = 1
    return parent


def _matrix_chain_order(sizes: np.ndarray) -> np.ndarray:
    """ ``clrs._src.algorithms.dynamic_programming.matrix_chain_order``. """
    size = len(sizes)
    cost = np.zeros((size, size))
    split = np.zeros((size, size))
    mask = np.zeros((size, size))
    for i in range(1, size):
        mask[i, i] = 1
    while True:
        before, seen = np.copy(cost), np.copy(mask)
        for i in range(1, size):
            for j in range(i + 1, size):
                flag = seen[i, j]
                for k in range(i, j):
                    if seen[i, k] == 1 and seen[k + 1, j] == 1:
                        mask[i, j] = 1
                        found = before[i, k] + before[k + 1, j] \
                            + sizes[i - 1] * sizes[k] * sizes[j]
                        if flag == 0 or found < cost[i, j]:
                            cost[i, j], split[i, j], flag = found, k, 1
        if np.all(before == cost):
            return split


def _dag_shortest_paths(matrix: np.ndarray, source: int) -> np.ndarray:
    """
    ``clrs._src.algorithms.graphs.dag_shortest_paths``: an iterative
    depth-first search building a topological order, then one relaxation
    pass along it.  Transcribed rather than replaced by "relax in
    topological order", because *which* order the search happens to build
    is what decides the parent of a node two equal-cost paths reach.
    """
    size = len(matrix)
    parent, distance = np.arange(size), np.zeros(size)
    mark, color = np.zeros(size), np.zeros(size, dtype=np.int64)
    topo, previous = np.arange(size), np.arange(size)
    head, last, u = 0, source, source
    while True:
        color[u] = 1
        for v in range(size):
            if matrix[u, v] != 0 and color[v] == 0:
                color[v], previous[v], last = 1, last, v
                break
        if last == u:
            color[u] = 2
            if color[head] == 2:
                topo[u] = head
            head = u
            if previous[u] == u:
                break
            before = previous[last]
            previous[last] = last
            last = before
        u = last
    distance[head], mark[head] = 0, 1
    while topo[head] != head:
        mark[head] = 1
        for j in range(size):
            if matrix[head, j] != 0.0:
                if mark[j] == 0 or distance[head] + matrix[head, j] \
                        < distance[j]:
                    distance[j] = distance[head] + matrix[head, j]
                    parent[j], mark[j] = head, 1
        head = topo[head]
    return parent


def check(split: Split, samples: int = 8, log=print) -> None:
    """
    Verify a split: the shapes agree with :data:`SPECS`, the trajectory
    lengths fit the hint axis, the graph is undirected with a full
    diagonal, and the outputs of the first ``samples`` trajectories are the
    ones the reference algorithm computes.

    Parameters:
        split : The split to verify.
        samples : How many trajectories to re-decide; the check is a
                  quadratic Python loop, so it is a sample rather than a
                  sweep.
        log : Where to print progress.
    """
    size = split.n
    for stage, group in (("input", split.inputs), ("hint", split.hints),
                         ("output", split.outputs)):
        assert set(group) == set(probes(split.algorithm, stage)), \
            f"{split.name}: {stage} probes are {sorted(group)}"
        for name, value in group.items():
            head = (split.steps, len(split)) if stage == "hint" \
                else (len(split), )
            assert value.shape == head + shape_of(
                split.algorithm, name, size), \
                f"{split.name}: {name} has shape {value.shape}"
    assert split.lengths.min() >= 1, f"{split.name}: an empty trajectory"
    assert split.lengths.max() <= split.steps, \
        f"{split.name}: a trajectory longer than the hint axis"

    if "adj" in split.inputs:
        adjacency = np.asarray(split.inputs["adj"]) > 0.5
        directed = split.algorithm in DIRECTED
        assert directed != np.array_equal(
            adjacency, np.transpose(adjacency, (0, 2, 1))), \
            f"{split.name}: adj is symmetric iff the sampler is undirected"
        assert adjacency[:, np.arange(size), np.arange(size)].all(), \
            f"{split.name}: adj has a hole in its diagonal"
        assert np.array_equal(
            adjacency, (np.asarray(split.inputs["A"]) > 0)
            | np.eye(size, dtype=bool)[None]), \
            f"{split.name}: adj is not the graph of A"

    for index in range(min(samples, len(split))):
        for name, found in reference(split.algorithm, split, index).items():
            truth = split.outputs[name][index]
            if kind(split.algorithm, name)[1] == "mask_one":
                truth = int(np.argmax(truth))
            assert np.array_equal(truth, found), \
                f"{split.name}: {name} disagrees with the reference " \
                f"on trajectory {index}"
    log(f"  {split.algorithm}/{split.name}: {len(split)} trajectories, "
        f"n = {size}, {samples} re-decided, verified")


# --- what the samplers draw ------------------------------------------------

#: The eight algorithms of ``project.md``, in the order the tables report
#: them.  Part 2 trains all of them, so this is now
#: :data:`config.ALGORITHMS`; the name is kept because :func:`survey` is
#: about *the benchmark's* eight rather than about a protocol's.
PROJECT = ALGORITHMS

#: The graph flags a sampler passes to ``_random_er_graph``.
FLAGS = ("directed", "acyclic", "weighted")

#: The ``(location, type)`` pairs the example can decode, i.e. the keys of
#: ``model.DECODERS``.  Kept here as a plain tuple so that this module
#: stays free of torch -- it is the one that needs ``clrs`` -- and pinned
#: against the real thing by ``test_clrs_smoke.py``.  Part 1 had the four
#: node rows; the edge rows are what ``floyd_warshall`` and
#: ``matrix_chain_order`` needed, and the last two are
#: ``dag_shortest_paths`` alone.
DECODABLE = (("node", "scalar"), ("node", "mask"), ("node", "mask_one"),
             ("node", "pointer"), ("node", "categorical"),
             ("edge", "scalar"), ("edge", "mask"), ("edge", "pointer"),
             ("graph", "mask"))


def survey(algorithms=PROJECT) -> dict:
    """
    What the CLRS samplers actually draw, read off ``clrs`` itself: the
    sampler class each algorithm resolves to, the graph flags its
    ``_sample_data`` passes, and the probes it locates on edges or on the
    graph rather than on nodes.

    This exists because whether a task is directed decides whether a
    directed-edge cell has to be written at all, and that question is
    answered by the sampler and by nothing else.  An algorithm whose
    sampler builds no graph reports no flags.

    Parameters:
        algorithms : The algorithms to survey.
    """
    import inspect
    import re

    from clrs._src import samplers, specs
    found = {}
    for algorithm in algorithms:
        sampler = samplers.SAMPLERS[algorithm]
        source = inspect.getsource(sampler._sample_data)
        graph = {flag: match.group(1) for flag in FLAGS
                 if (match := re.search(rf"\b{flag}=(\w+)", source))}
        spec = specs.SPECS[algorithm]
        kinds = {name: (str(stage), str(location), str(type_))
                 for name, (stage, location, type_) in spec.items()}
        found[algorithm] = {
            "sampler": sampler.__name__,
            "graph": graph,
            "located": {name: kind_ for name, kind_ in kinds.items()
                        if kind_[1] != "node"},
            "missing": sorted({
                kind_[1:] for name, kind_ in kinds.items()
                if kind_[0] != "input" and kind_[1:] not in DECODABLE}),
        }
    return found


def tabulate(found: dict, log=print) -> None:
    """ :func:`survey`, as the markdown table in ``README.md``. """
    log("| algorithm | sampler | directed | weighted "
        "| decoded off nodes | decoders the example lacks |")
    log("|---|---|---|---|---|---|")
    for algorithm, row in found.items():
        graph = row["graph"]
        elsewhere = ", ".join(
            f"`{name}` {stage}/{location}/{type_}"
            for name, (stage, location, type_) in row["located"].items()
            if stage != "input")
        missing = ", ".join(f"`{location}/{type_}`"
                            for location, type_ in row["missing"])
        log(f"| `{algorithm}` | `{row['sampler']}` "
            f"| {graph.get('directed', '*no graph*')} "
            f"| {graph.get('weighted', '—')} "
            f"| {elsewhere or '—'} | {missing or '—'} |")


# --- the script ------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="*", default=ALGORITHMS)
    parser.add_argument("--splits", nargs="*", default=SPLITS)
    parser.add_argument("--generate", action="store_true",
                        help="sample with `clrs` and cache, overwriting")
    parser.add_argument("--check", action="store_true",
                        help="verify what is cached, without `clrs`")
    parser.add_argument("--survey", action="store_true",
                        help="what the samplers draw, for all eight of "
                             "`project.md`; needs `clrs`")
    parser.add_argument("--samples", type=int, default=8,
                        help="trajectories re-decided per split by --check")
    arguments = parser.parse_args(argv)

    if arguments.generate:
        check_spec()
        for algorithm in arguments.algorithms:
            for split in arguments.splits:
                if path_of(algorithm, split).exists():
                    print(f"  {algorithm}/{split}: cached")
                    continue
                save(generate(algorithm, split))
    if arguments.check:
        for algorithm in arguments.algorithms:
            for split in arguments.splits:
                check(load(algorithm, split), arguments.samples)
    if arguments.survey:
        tabulate(survey())
    if not (arguments.generate or arguments.check or arguments.survey):
        parser.error("nothing to do: pass --generate, --check or --survey")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
