# -*- coding: utf-8 -*-

"""
Paths, widths, budgets and the canonical CLRS-30 protocol.

Everything here is a plain value: no torch, no diagrams, nothing that has to
be built.  :data:`CLRS30` is the benchmark's own split specification copied
verbatim from ``clrs._src.samplers``, so that "the benchmark's samples"
means the same thing here as it does in the papers this study is measured
against; a :class:`Widths` fixes the parameter count of a model and a
:class:`Budget` fixes what a training run is allowed to spend.

Part 1 is three algorithms.  The other five of the eight named in
``project.md`` are deliberately absent: they need the directed-edge cell and
the edge-level decoders of Part 2, and listing them here would let a
protocol quietly touch them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

#: Everything this example reads and writes lives inside its own directory,
#: so the whole study is one relocatable folder.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"

for _directory in (DATA_DIR, ARTIFACTS, FIGURES):
    _directory.mkdir(parents=True, exist_ok=True)

#: The CLRS-30 baseline specification, verbatim from
#: ``clrs._src.samplers.CLRS30``: train on 1000 trajectories at ``n = 16``,
#: validate on 32 more at ``n = 16``, test out of distribution on 32 at
#: ``n = 64``.  The seeds are the benchmark's, so these *are* its samples
#: and not a re-draw of its distribution.
CLRS30 = {
    "train": {"num_samples": 1000, "length": 16, "seed": 1},
    "val": {"num_samples": 32, "length": 16, "seed": 2},
    "test": {"num_samples": 32, "length": 64, "seed": 3},
}

#: A larger out-of-distribution split, drawn at the same length as
#: ``test`` but from a seed of its own.  It is the *primary* number of
#: every table -- 32 trajectories is what the literature compares on and
#: is reported beside it for comparability, but a difference between two
#: rows of 32 is not a difference at all until a confidence interval says
#: so.  128 rather than 512 because an edge-level hint of
#: ``floyd_warshall`` at ``n = 64`` is an ``n x n`` matrix per step, and
#: 512 trajectories of them is a gigabyte of cache to hold a number nobody
#: reads to three decimals.
WIDE = {"num_samples": 128, "length": 64, "seed": 30}

#: The sizes a *mixed* training split is drawn at, and how the 1000
#: trajectories of ``CLRS30["train"]`` are divided between them: 200 each,
#: so the data budget is the one the anchors were trained under and only
#: its *shape* changes.
#:
#: At a single size every training trajectory is the same depth, so
#: "iterate further than you ever have" is not a thing the model was ever
#: asked to do -- and Part 2's depth ladder is the measurement of what
#: that costs.  Drawing 8 through 16 puts a range of depths in the
#: training distribution without touching the evaluation protocol, which
#: stays ``n = 16`` in distribution and ``n = 64`` out of it.  The
#: benchmark's own rule is ``n <= 16``, so this is inside it.
#:
#: Even sizes only, and each its own cached split: a :class:`~dataset.
#: Split` is rectangular arrays at one ``n``, and a batch is one compiled
#: diagram, so mixing sizes *within* a batch would multiply the shapes a
#: campaign compiles for nothing.  Batches stay homogeneous and an epoch
#: shuffles across them.
MIXED = (8, 10, 12, 14, 16)

#: The splits :func:`~dataset.load` builds.  The mixed training sizes are
#: named ``train8`` ... ``train16``; ``train`` itself is the single-size
#: split every number of Part 2 was measured under and stays.
SPLITS = ("train", "val", "test", "wide") \
    + tuple(f"train{size}" for size in MIXED)

#: The eight algorithms of ``project.md``.  Part 1 built the harness on
#: the first three -- ``minimum`` is a non-graph sanity check that only the
#: readout relation can solve, ``bfs`` is the canonical parallel wavefront
#: and ``bellman_ford`` is literally a fixed-point iteration.  Part 2 adds
#: the two sequential-frontier boundary probes (``dijkstra``,
#: ``mst_prim``), the one directed task (``dag_shortest_paths``) and the
#: two edge-state showcases H1 runs through (``floyd_warshall``,
#: ``matrix_chain_order``).
ALGORITHMS = ("minimum", "bfs", "bellman_ford", "dijkstra", "mst_prim",
              "dag_shortest_paths", "floyd_warshall", "matrix_chain_order")

#: The seeds every model is trained with, in order.
SEEDS = (0, 1, 2)

#: Gradient-norm clipping, identical for every model.
GRAD_CLIP = 1.0

#: The weight of the per-step hint loss beside the output loss.  Hints
#: supervise; hints are not scored.
HINT_WEIGHT = 1.0

#: How every site reduces its message orbit, a key in
#: :data:`discopy.neural.cells.POOL`.  This is the one architectural knob
#: that a *change of size* can see, so it is a named default rather than a
#: literal buried in :func:`~model.build`: a mean and a sum both rescale
#: when a node's degree grows from ``n = 16`` to ``n = 64``, whereas a max
#: keeps an extremum and an extremum is what ``minimum`` -- and the relaxed
#: edge of ``bellman_ford`` -- is made of.
#:
#: ``mean`` was Part 1's default, on the stated grounds that it is what the
#: CLRS baseline uses.  **That was wrong**:
#: ``clrs._src.processors.PGN.__init__`` -- which ``MPNN`` subclasses
#: without overriding it -- declares ``reduction: _Fn = jnp.max``, so the
#: floor this study is measured against is max-aggregated.  Part 2's
#: **primary** campaign therefore reduces with ``max`` (``--pool max``,
#: filed under ``full-max-*``) and the ``mean`` one is the aggregator
#: ablation it always should have been.
#:
#: This constant stays ``mean`` regardless, because it is what
#: :attr:`Budget.tag` measures a run's name against: changing it would
#: rename every artefact already written and silently re-read a mean run
#: as a max one.  Which campaign is primary is a fact about the study, and
#: it is recorded in ``README.md``, not in a filename.
POOL = "mean"


#: Where the basin at termination is put into the loss; see
#: :attr:`Budget.settle` and :meth:`model.Model.hint_targets`.  ``True`` is
#: accepted for ``"interior"`` so that a budget written before Part 3 --
#: and the artifacts it filed -- still mean what they meant.
SETTLE = (None, "interior", "terminal")


def holding(settle) -> str:
    """
    The member of :data:`SETTLE` a budget asks for, ``None`` for none.

    Example
    -------
    >>> holding(False), holding(True), holding("terminal")
    (None, 'interior', 'terminal')
    """
    if not settle:
        return None
    settle = "interior" if settle is True else settle
    if settle not in SETTLE:
        raise ValueError(f"{settle!r} is not one of {SETTLE}")
    return settle


@dataclass(frozen=True)
class Widths:
    """
    The widths of one model, the knobs that set its parameter count.

    Parameters:
        dim : The width of a node-to-relation message and of an encoded
              input feature.
        state_dim : The width of a node's recurrent state.
        hidden : The width of the hidden layers inside a cell.
        edge_dim : The width of an edge box's recurrent state.  Sending it
                   to zero erases the edge machinery from the same diagram,
                   which is Part 2's H1 ablation.
        graph_dim : The width of the readout relation's recurrent state.
    """
    dim: int = 16
    state_dim: int = 96
    hidden: int = 192
    edge_dim: int = 48
    graph_dim: int = 96

    def asdict(self) -> dict:
        return {"dim": self.dim, "state_dim": self.state_dim,
                "hidden": self.hidden, "edge_dim": self.edge_dim,
                "graph_dim": self.graph_dim}


@dataclass(frozen=True)
class Budget:
    """
    One experiment budget: how much data, how long, and how deep.

    Parameters:
        name : The name of the budget, used for artifact filenames.
        epochs : The passes over the training split.
        batch_size : The trajectories per compiled diagram.  A batch is one
                     diagram -- the disjoint union of its members' graphs --
                     so it is compiled once and reused every epoch, which is
                     what makes a fixed batching worth more here than a
                     reshuffle.
        eval_batch_size : The trajectories per compiled diagram at
                          evaluation time.  Smaller than ``batch_size``
                          because an out-of-distribution sample is four
                          times the nodes and sixteen times the edges of a
                          training one, and compiling a diagram is
                          superlinear in its boxes -- see ``NOTES.md``.
        rounds : The message-passing rounds trained at, or ``None`` for
                 the trajectory rule -- ``HOPS`` rounds per step of the
                 sampled execution, which is the protocol of Part 2 and
                 the only one under which a table of eight algorithms
                 whose trajectories are ``n``, ``n + 1`` and ``3`` steps
                 long means one thing.  A number here is the *fixed-depth*
                 regime Part 1 recorded, kept so that its rows stay
                 reproducible and labelled.
        lr : The learning rate of Adam.
        weight_decay : The decoupled weight decay of AdamW.
        n_train : The trajectories used, ``None`` for the whole split.
        n_wide : The trajectories of the larger out-of-distribution split
                 that are scored; the canonical 32 are always scored whole.
        eval_every : The epochs between two validation passes.
        seeds : The seeds to train.
        sweep : The test-time depths, as multiples of the trained one.
                A *multiple* rather than a round count, because under the
                trajectory rule the trained depth is a property of the
                batch: "three times as deep as the algorithm is long" is
                the only form of the sweep that means the same thing at
                ``n = 16`` and at ``n = 64``.
        widths : The key in :data:`WIDTHS` the model is built at.
        pool : The key in :data:`discopy.neural.cells.POOL` every site
               reduces its message orbit with; see :data:`POOL`.
        edge_state : Whether an edge box carries a recurrent state of its
                     own.  ``False`` is H1's node-only arm: the same
                     diagram with ``ESTATE`` sent to ``Dim(0)``, so the
                     messages still pass through the edges and nothing
                     remembers a pair.
        hint_weight : The weight of the per-step hint loss, ``0`` for the
                      output-only ablation Part 3's solver table needs.
        mixed : Whether training draws :data:`MIXED` sizes rather than
                ``n = 16`` alone, at the same total number of
                trajectories.
        pointer : Which node-pointer head to build; see
                  :func:`model.build`.
        pos : What is done to the ``pos`` input, a key of
              :data:`dataset.POS`.  ``"shuffled"`` destroys the *rank*
              and so the labels' own tie-breaking, which makes the task
              partly unrealizable; ``"uniform"`` keeps the rank and
              destroys only the size-dependent ``1 / n`` spacing.
        settle : Where the basin at termination is put into the loss, a
                 member of :data:`SETTLE`.  ``None`` drops a finished
                 trajectory's remaining checkpoints, which is the
                 protocol every number of Part 2 was measured under.
                 ``"interior"`` holds its final hint on the checkpoints
                 past its own end, which is what ``--settle`` trained;
                 it reaches every checkpoint **but the last**, because
                 :meth:`model.Model.hint_targets` refuses a hint index
                 the batch does not define before it consults ``settle``
                 at all.  ``"terminal"`` holds it there too, which is
                 the one checkpoint a :class:`~discopy.neural.
                 FixedPoint` converges to and therefore the only place a
                 trained basin is a basin H2 can measure; see
                 ``PART3.md``.
        solver : How a run is executed, a key of :data:`model.SOLVERS`.
                 The execution policy is a *test-time* knob here and the
                 training arms differ in their differentiation policy
                 alone: see ``backward``, and ``PART3.md`` for why a
                 residual stopping rule cannot be trained against
                 without changing two things at once.
        probe : Whether the hint loss is decoded from a **detached**
                state, so that it fits the hint decoders and never the
                interaction.  This is what an *output-only* arm of
                Part 3 is, and it is not ``hint_weight = 0``: with no
                hint term at all the hint decoders receive no gradient,
                so the very curves the mandatory per-head split is read
                from would come out of untrained heads, and an arm would
                score nothing on the order-free column without its
                processor having failed at anything.  Detaching keeps
                the axis -- the interaction sees the output alone -- and
                makes the hint heads linear probes of the state, which
                is a caveat their numbers carry rather than a confound
                in the comparison.
        backward : The differentiation policy of a fixed-point solver,
                   ``"full"`` for unrolled backpropagation through every
                   round run or ``"last"`` for the Jacobian-free
                   one-step gradient of the deep-equilibrium models.
                   With ``tol`` disabled -- which every training arm
                   runs with -- ``"full"`` is *bitwise* ``Iterate``, so
                   ``"last"`` is the only fixed-point arm that is a row
                   rather than a rename.
    """
    name: str
    epochs: int
    batch_size: int
    eval_batch_size: int = 4
    rounds: int = None
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_train: int = None
    n_wide: int = 128
    eval_every: int = 10
    seeds: tuple[int, ...] = SEEDS
    sweep: tuple[float, ...] = field(default=(1.0, 1.5, 3.0))
    widths: str = "mpnn"
    pool: str = POOL
    edge_state: bool = True
    hint_weight: float = HINT_WEIGHT
    mixed: bool = False
    settle: str = None
    pointer: str = "bilinear"
    pos: str = "sampler"
    solver: str = "iterate"
    backward: str = "full"
    probe: bool = False

    @property
    def tag(self) -> str:
        """
        What a run's artifacts are filed under: the budget's name, and
        whatever it changed about the architecture.  Two budgets that
        differ in their model can then be run side by side rather than
        overwriting each other's weights, which is what makes an
        architectural ablation a comparison instead of a replacement.

        Example
        -------
        >>> from dataclasses import replace
        >>> FULL.tag, replace(FULL, pool="max", widths="small").tag
        ('full', 'full-small-max')
        >>> replace(FULL, edge_state=False).tag
        'full-nodeonly'
        >>> replace(FULL, pool="max", mixed=True, settle=True).tag
        'full-max-mixed-settle'
        >>> replace(FULL, pool="max", n_train=200).tag
        'full-max-n200'
        >>> replace(FULL, settle="terminal").tag
        'full-settle-term'
        >>> replace(FULL, solver="fixedpoint", backward="last").tag
        'full-fixedpoint-last'
        """
        parts = [self.name]
        if self.widths != "mpnn":
            parts.append(self.widths)
        if self.pool != POOL:
            parts.append(self.pool)
        if not self.edge_state:
            parts.append("nodeonly")
        if self.hint_weight != HINT_WEIGHT:
            parts.append("nohints" if not self.hint_weight else "hint"
                         + str(self.hint_weight).replace(".", "_"))
        if self.pointer != "bilinear":
            parts.append(f"ptr{self.pointer}")
        if self.pos != "sampler":
            parts.append("shufpos" if self.pos == "shuffled" else "unifpos")
        if self.mixed:
            parts.append("mixed")
        if self.settle:
            parts.append("settle")
            if holding(self.settle) == "terminal":
                parts.append("term")
        if self.probe:
            parts.append("probe")
        if self.solver != "iterate":
            parts.append(self.solver)
            if self.backward != "full":
                parts.append(self.backward)
        if self.n_train is not None:
            parts.append(f"n{self.n_train}")
        if self.rounds is not None:
            parts.append(f"r{self.rounds}")
        return "-".join(parts)


#: A few-second miniature, faithful in everything but the amounts, so it
#: exercises exactly the code paths of the large run -- including the
#: trajectory rule, which is why it has no ``rounds``.
QUICK = Budget(name="quick", epochs=4, batch_size=8, eval_batch_size=2,
               n_train=32, n_wide=8, eval_every=2, seeds=(0, ),
               sweep=(1.0, 2.0))

#: The budget behind the recorded baseline: 300 epochs of 32 batches is
#: 9600 optimizer steps, near enough the 10 000 the CLRS-30 baselines take.
FULL = Budget(name="full", epochs=300, batch_size=32, eval_batch_size=4,
              lr=1e-3, eval_every=10)

#: Part 1's regime, kept runnable: a *constant* 16 rounds -- 8 algorithm
#: steps -- for every sample of every algorithm.  Its numbers are not
#: comparable with the ones above and the tables label them as their own
#: row; see ``NOTES.md``.
FIXED = Budget(name="full", epochs=300, batch_size=32, eval_batch_size=4,
               rounds=16, lr=1e-3, eval_every=10)

#: The widths a model can be built at.  ``mpnn`` is the recorded baseline;
#: ``small`` halves every width, which is a quarter of the parameters in
#: every matrix and the budget to run a campaign at when the question is
#: about the *protocol* rather than about capacity -- these are 16-node
#: graphs and four probes, not a task that wants 300k parameters.
#: ``paired`` is the node-only arm of H1, whose edge cell has no recurrent
#: state and therefore fewer parameters: it is widened until the two arms
#: match, which :func:`~model.matched` re-derives and
#: ``test_the_h1_arms_are_parameter_matched`` pins.
WIDTHS = {
    "mpnn": Widths(),
    "small": Widths(dim=8, state_dim=48, hidden=96, edge_dim=24,
                    graph_dim=48),
    "paired": Widths(hidden=208),
}

#: Where the anchors come from, recorded beside them because a number
#: whose table nobody can name is a number nobody can check.
ANCHOR_SOURCE = {
    "paper": "Ibarz et al. (2022), A Generalist Neural Algorithmic "
             "Learner, arXiv:2209.11142",
    "table": "Table 2 -- Single-task OOD average micro-F1 score of "
             "previous SOTA Memnet, MPNN and PGN [5] and our best model "
             "Triplet-GMPNN with all the improvements described in "
             "Section 3",
    "columns": "Alg. Type | Memnet | MPNN | PGN | Triplet-GMPNN (ours)",
    "floor": "the MPNN column, which that table attributes to [5], i.e. "
             "Velickovic et al. (2022), the CLRS-30 benchmark paper",
    "ceiling": "the Triplet-GMPNN column of the *same* table and the same "
               "row; the paper's multi-task generalist has its own per "
               "-algorithm numbers, in Figures 3 and 5, and none of them "
               "is used here",
    "single_task": "one model per algorithm, which is what this study "
                   "trains; the caption reads 'Single-task OOD micro-F1 "
                   "score of previous SOTA Memnet, MPNN and PGN [5] and "
                   "our best model Triplet-GMPNN'",
    "sizes": "trained on n <= 16, evaluated out of distribution at "
             "n = 64, which is the protocol of SPLITS",
    "read": "the ar5iv HTML rendering, 2026-08-12, three times under "
            "different prompts with identical digits, and cross-checked "
            "against the table's own overall average (MPNN 51.02, "
            "Triplet-GMPNN 75.98) and the abstract's 'over 20% "
            "improvement'",
    "caveat": "means under the paper's own training improvements, which "
              "this study does not reproduce",
}

#: The single-task out-of-distribution micro-F1 of :data:`ANCHOR_SOURCE`,
#: as fractions rather than the paper's percentages.
#:
#: The second field is ``sem`` and not ``std`` on the paper's own
#: authority: "error bars represent standard error of the mean across
#: seeds (3 seeds for previous SOTA experiments, 10 seeds for current)",
#: so the floor's interval is over three seeds and the ceiling's over ten,
#: and neither is a standard deviation.  A table that prints ours beside
#: theirs has to say which is which, which is why
#: :func:`~evaluate.summarise` records ``std`` *and* ``sem`` and
#: :func:`~evaluate.tabulate` prints the one the anchors print.
#:
#: ``None`` would mean "not yet transcribed" -- Part 1 left every one of
#: them ``None`` and every reporting path still prints a gap only where a
#: number has been filled in, so that a remembered figure can never be
#: mistaken for a published one.  ``project.md``'s figures *are* from
#: memory and one of them is wrong: it recalls ``dag_shortest_paths`` at
#: 88, and the table says 98.19.
ANCHORS = {
    "minimum": {"floor": {"mean": 0.8534, "sem": 0.0088},
                "ceiling": {"mean": 0.9778, "sem": 0.0055}},
    "bfs": {"floor": {"mean": 0.9989, "sem": 0.0005},
            "ceiling": {"mean": 0.9973, "sem": 0.0004}},
    "bellman_ford": {"floor": {"mean": 0.9201, "sem": 0.0028},
                     "ceiling": {"mean": 0.9739, "sem": 0.0019}},
    "dijkstra": {"floor": {"mean": 0.9150, "sem": 0.0050},
                 "ceiling": {"mean": 0.9605, "sem": 0.0060}},
    "mst_prim": {"floor": {"mean": 0.6908, "sem": 0.0756},
                 "ceiling": {"mean": 0.8639, "sem": 0.0133}},
    "dag_shortest_paths": {"floor": {"mean": 0.9624, "sem": 0.0056},
                           "ceiling": {"mean": 0.9819, "sem": 0.0030}},
    "floyd_warshall": {"floor": {"mean": 0.2674, "sem": 0.0177},
                       "ceiling": {"mean": 0.4852, "sem": 0.0104}},
    "matrix_chain_order": {"floor": {"mean": 0.7984, "sem": 0.0140},
                           "ceiling": {"mean": 0.9168, "sem": 0.0059}},
}


# --- Part 3 ----------------------------------------------------------------
#
# The protocol is argued in ``PART3.md`` and only named here.  Four rules
# bind every arm below and none of them is a default that can drift:
#
# 1. the **decoder is frozen** -- :data:`PART3`'s ``pointer="edge"`` on
#    every compared arm, since a localized constant deficit is an offset
#    and a deficit that moves between arms is a confound;
# 2. every table **splits the heads**, :data:`ORDER_FREE` against
#    :data:`ORDER_DEPENDENT`, so that a solver is never credited with
#    pointer points;
# 3. **no claim against a published anchor**, the `bfs` gate being unmet;
# 4. **three seeds for any verdict**, :data:`SEEDS`.

#: The five rows on which a round approximates a step -- the tasks whose
#: hint curves show the model executing the algorithm for as long as it
#: has iterated before and coming apart after, which is the premise H2
#: needs.  ``minimum`` and ``matrix_chain_order`` are excluded on the
#: measurement that their ``pred_h`` never exceeds 0.14 at any step of an
#: ``n = 64`` trajectory: there is nothing there for a fixed point to
#: converge to, so a solver row on them would measure the absence of
#: something never trained.  ``bfs`` is excluded because it never iterates
#: past its trained depth at all.
EXECUTORS = ("bellman_ford", "dijkstra", "mst_prim", "dag_shortest_paths",
             "floyd_warshall")

#: The probe types whose answer is **one element at a time** -- a sigmoid
#: per node or per pair, a class out of a fixed set.  Their candidate set
#: does not grow with the graph, they are exact out of distribution, and
#: they are the mass a solver may honestly be credited with.
ORDER_FREE = ("mask", "categorical")

#: The probe types whose answer is an **argmax over the node set**: the
#: candidate set *is* the graph, 16 in training and 64 out of it, and the
#: CLRS reference algorithms iterate in index order so their targets are
#: tie-broken by it.  M2 is exactly this class failing while
#: :data:`ORDER_FREE` does not.
ORDER_DEPENDENT = ("pointer", "mask_one")

#: The probe type that is scored by a **mean squared error** and therefore
#: pooled with nothing: it is unbounded and lower is better, so a mean of
#: it with an F1 is a number with no reading.  Averaging it in is what
#: made ``floyd_warshall``'s order-free drop come out *negative*; see
#: ``PART3.md``.
UNPOOLED = ("scalar", )

#: Part 3's base budget: Part 2's protocol with the decoder frozen at the
#: arm that measured best on ``bfs`` (`B`, :class:`model.
#: EdgePointerDecoder2`) and the aggregator the floor uses.  Nothing here
#: is comparable with Part 2's eight-task table, which was measured with
#: the bilinear head; Part 3 carries its own reference row instead.
PART3 = Budget(name="p3", epochs=300, batch_size=32, eval_batch_size=4,
               lr=1e-3, eval_every=10, pool="max", pointer="edge")

#: **R** -- the reference: what every other arm is read against.
H2_REFERENCE = PART3

#: **S** -- the trained basin.  One axis against `R`: whether the loss
#: says anything about what the map does once the algorithm has finished.
#: ``"terminal"`` rather than ``"interior"`` because the checkpoint a
#: fixed point converges to is the last one, and that is the one
#: ``"interior"`` cannot reach.
H2_SETTLE = replace(PART3, settle="terminal")

#: **O** -- the supervision control the solver table needs: an
#: ``Iterate`` whose *interaction* is trained on its output alone.
#: Without it a fixed-point row differs from `R` in the solver *and* in
#: the supervision, and the hint term is most of the gradient here.
#: ``probe`` rather than ``hint_weight=0`` so that the hint heads are
#: still fit and the mandatory per-head split can be read at all.
H2_OUTPUT_ONLY = replace(PART3, probe=True)

#: **F** -- the differentiation policy, one axis against `O`: the
#: Jacobian-free one-step gradient instead of unrolled backpropagation,
#: at the same rounds, the same supervision and the same parameters.
#: The residual stopping rule is *not* trained against -- with it the
#: arm would differ from `O` in the gradient and in the effective depth
#: -- so ``tol`` is a test-time knob and this runs the trajectory's
#: rounds like everything else.
#: ``"grounded"`` rather than ``"fixedpoint"``: the library's solver
#: detaches the whole state for its one-step gradient, and here the
#: inputs ride *in* the state, so it would train frozen encoders and the
#: contrast with `O` would be two changes wide.  See :class:`model.
#: Grounded`, whose forward pass is bitwise the library's.
#: ``backward`` stays at its default because :class:`model.Grounded` *is*
#: the one-step gradient -- it passes ``"last"`` to its own base -- so this
#: arm differs from `O` in exactly one field of this dataclass, which is
#: what a one-axis contrast should look like when someone reads the config
#: rather than the prose.
H2_FIXEDPOINT = replace(PART3, probe=True, solver="grounded")

#: H2's four trained arms, by the letter their tables print.
H2_ARMS = {"R": H2_REFERENCE, "S": H2_SETTLE,
           "O": H2_OUTPUT_ONLY, "F": H2_FIXEDPOINT}

#: The one-axis contrasts H2 is allowed to read off :data:`H2_ARMS`, and
#: nothing else: a pair that differs in two things measures neither.
H2_CONTRASTS = {
    ("R", "S"): "termination supervision -- how much basin is free and "
                "how much is trained",
    ("R", "O"): "supervision regime -- per-step hints against none",
    ("O", "F"): "differentiation policy -- unrolled against Jacobian-free",
}

#: Where the per-row size regime is recorded once the probe has measured
#: it.  A *file* rather than a literal in this module, because the regime
#: is a measurement and not an opinion: the probe writes it, the campaign
#: reads it, and a decision nobody can edit after seeing a campaign's
#: numbers is the whole point.
REGIME_FILE = ARTIFACTS / "regime.json"

#: The margin the mixed arm has to clear to be adopted, as a multiple of
#: the row's own **Part 2 seed standard error** -- the only estimate of
#: run-to-run noise this study owns, and it is already measured.  The
#: probe is one seed an arm, so a difference smaller than the noise of a
#: single seed is not a difference.
REGIME_MARGIN = 1.0

#: Whether a row trains on :data:`MIXED` sizes, **declared per row before
#: the campaign and frozen after**.  ``minimum`` holds at a fifth of the
#: data at one size and collapses to 0.17 under mixing at the full
#: budget, so the blanket protocol is dead and each row owes its own
#: decision.
#:
#: The rule the probe applies is pre-registered here, before any probe
#: number exists, and it is deliberately conservative: **"fixed" unless
#: "mixed" wins by more than the row's own seed noise, and never when it
#: costs the order-free heads.**  Part 2's protocol is the incumbent, it
#: is the one that did not destroy ``minimum``, and a one-seed probe
#: cannot support more than that.
#:
#: ``None`` is a row the probe has not decided.  :func:`train.regime_of`
#: refuses to train it rather than defaulting it, because a default here
#: is a protocol chosen by whoever ran the script first.
REGIME = {algorithm: None for algorithm in ALGORITHMS}
if REGIME_FILE.exists():
    REGIME.update({
        name: found["regime"] for name, found
        in json.loads(REGIME_FILE.read_text())["rows"].items()})
