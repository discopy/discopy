# Notes from Parts 1 and 2

Part 2's notes are at the [end](#notes-from-part-2); Part 1's follow, kept
as they were written, including the mistake it made.

# Notes from Part 1

What was noticed while building the harness and deliberately *not* smuggled
into a docstring: the places where the brief in [`project.md`](project.md)
turned out to be wrong, the protocol choices it left open, and the costs it
asked to be measured.

## Where `project.md` was wrong

**`bellman_ford`'s CLRS graphs are undirected.** The brief says "for
`bellman_ford` on directed CLRS graphs, symmetrize in Part 1 and note it;
Part 2 fixes it properly". There is nothing to fix: `clrs`'s own
`BellmanFordSampler._sample_data` calls `_random_er_graph(...,
directed=False, ...)`, and `dataset.check` asserts on every cached split
that `adj` is symmetric. So is `bfs`'s. Direction enters CLRS-30 at
`dag_shortest_paths` (`acyclic=True`) and at nothing Part 1 touches, which
means the `Sym.PERM` two-port edge box is the *honest* signature here rather
than a simplification — `check_equivariant` is owed and is measured.

**`from_incidence` needs no `Batch`, and `Batch(pad=True)` would be wrong
here.** The brief worries about interning per `(algorithm, n, edge-set)`.
The resolution is the SAT example's: several samples are laid side by side
in *one* incidence list, which already is their monoidal product, so a batch
is one diagram built in one pass. Padding never enters, because CLRS batches
are same-`n` by construction.

**Only one of the eight algorithms is directed, and it is not the one the
brief plans around.** Part 2's first named deliverable is "the directed-edge
cell in `model.py` (the ~30-line custom Cell)". `python dataset.py --survey`
reads the samplers: `dijkstra` and `mst_prim` both resolve to
`BellmanFordSampler` and `floyd_warshall` to `FloydWarshallSampler`, all
three drawing `directed=False`; `matrix_chain_order` draws no graph at all.
`dag_shortest_paths` is the only directed task in the project, and it is
also the only one wanting a `node/categorical` and a `graph`-located probe.
The cell is worth writing for it, but it is one algorithm out of eight and
not the shape of Part 2. What *is* the shape is in the same table's last
column — `edge/pointer`, `edge/scalar` and `edge/mask` decoders, which
`floyd_warshall` and `matrix_chain_order` both need and which H1 runs
through. The table is in `README.md` and the command that prints it is in
`dataset.py`, so it can be re-checked rather than remembered.

## The defect the first campaign had

**One decoder per feature *type* was wrong; it is one per *probe*.** The
first version of `model.py` built one decoder per `(location, type)` and
shared it between every probe of that kind, hint and output alike. For
`bfs` and `bellman_ford` it is nearly harmless — `pi_h` and `pi` are the
same question asked at two times, and they still validated in the high
nineties. For `minimum` it is fatal: `min`, `min_h` and `i` are *three*
`node/mask_one` probes, respectively the answer, the running answer and the
loop counter, and one logit per node cannot be three distributions at once.
The symptom was an in-distribution score of **0.9375 on the easiest task in
the suite**, with the validation score swinging between 0.34 and 0.94
between neighbouring epochs of a converged run, and an output loss that
stalled at 0.09 while the hint loss sat at 1.15. Same recipe, one head per
probe:

| `minimum`, mean pooling | ID `n = 16` | OOD `n = 64` | OOD, 128 more |
|---|---|---|---|
| one decoder per feature type | 0.9375 | 0.4375 | 0.3203 |
| one decoder per probe | **1.0000** | **0.9375** | **0.9297** |

It is worth being precise about what was and was not structural here.
Mean pooling explains a gap between `n = 16` and `n = 64`; it cannot explain
a miss *at* `n = 16`, where an argmin over sixteen scalars is trivial. Both
failures were real and they had different causes, so the aggregator was
worth changing too — but on the evidence above, most of what looked like a
size-generalization collapse was a head that could not represent its own
target at any size.

`clrs._src.nets.Net._construct_encoders_decoders` keys `dec` by `name`, so
this is also the benchmark's own convention, and departing from it was the
kind of simplification that looks like economy and is a loss of expressive
power. `decoded` now returns probe names and
`test_a_probe_is_decoded_by_a_head_of_its_own` pins it.

**And the loss is now reported per probe, not only per stage.** The head
was failing in plain sight for three hundred epochs, and the log showed
`out 0.09, hint 1.15` — two numbers, neither of which can say *which* probe
is not learning. `Model.loss` returns a `probe/<name>` term for every
decoded probe, `train.py` prints them beside the stage totals and keeps them
in the history, and `test_the_loss_is_reported_per_probe` asserts the stage
terms are the sums of the probe terms that feed them. It is four lines, and
Part 2 needs it twice over: `floyd_warshall` decodes seven probes to
`bfs`'s three, three of them `n × n` edge matrices, so there is more room to
hide — and per-probe hint divergence is a diagnostic that part asks for
anyway.

## Protocol choices the brief left open

**Two clocks.** A node reaches a node through a box — `node → edge → node`
and `node → readout → node` are both two rounds — so one step of the
imitated algorithm is `model.HOPS = 2` rounds, and `Model.checkpoints` reads
a checkpoint every second round. Aligning round `r` with hint step `r`
instead would ask the model to propagate a hop per round on a factor graph
that cannot. This is the sudoku study's hop law (`A@2R ≈ B@R`) reappearing
where the depth is semantically meaningful.

**The output is supervised from the trajectory's end onwards, not at the
last round alone.** CLRS decodes its outputs at the final step of the
unroll. Here every checkpoint at or after a sample's termination step is
supervised on the output, which asks the model to reach the answer *and stay
there*. That is what makes the test-time depth sweep a measurement rather
than a lottery, and it is the protocol H2 will need in Part 3. The
alternative is one line in `Model.loss`.

*With one caveat that only `minimum` hits.* `settled` clamps to the last
checkpoint, so a trajectory longer than the run gets its output supervised
**once**, at the end, and the "stay there" pressure disappears. `bfs` and
`bellman_ford` terminate in 3 to 8 steps against 8 checkpoints, so most of
their samples are supervised several times over; `minimum`'s trajectory is
`n` steps — 16 at `n = 16` and 64 at `n = 64` — so it never is. The depth
sweep then measures exactly what one would expect of a model never asked to
hold still: `minimum` scores 1.0 at the depth it was trained at and 0.78 at
three times that. It is a property of the supervision, not of the
aggregator, and both poolings show it equally.

**Hints are open-loop and are never fed back**, per the brief; `hint_weight`
in `config.py` is the knob that turns the hint loss off entirely, which is
the ablation Part 2 will want.

**`minimum`'s hints are sequential and its model is not.** The `i` probe
advances one index per step, which a permutation-equivariant synchronous
cell can only imitate through `pos`. The hint loss is kept anyway — hints
supervise, hints are not scored — and the output, which *is* scored, needs
only the readout relation.

*This paragraph used to end "expect the hint term of `minimum` to stay
high; that is a property of the alignment, not a bug", and that was
wrong.* The first campaign's hint term sat at 1.15 and the prediction
looked confirmed. It was the three `mask_one` heads colliding: with a head
per probe the same term is 0.002. `pos` is enough after all — a node knows
its own index and its recurrent state counts the rounds, so "which node is
index `t`" is learnable. The mistake is kept on the page because it is why
the real defect went unexamined for a whole campaign: a structural story
that explains a number is not evidence that the number is structural.

## What Part 1 was asked to measure

**Batched calls per round.** `model.calls_per_round(model, batch)` counts
the distinct `(name, port signature)` groups of a compiled batch, which is
what a round costs in kernel launches. It is recorded in every training
artefact. Degree heterogeneity is mild: at `n = 16`, `p = 0.5` a batch of 32
`bfs` samples has ~13 distinct node degrees, one edge group and one readout
group.

**Wall clock.** Recorded per epoch in the artefact, beside the device and
the torch version.

**A smaller model is not a faster run.** `WIDTHS["small"]` halves every
width -- a quarter of the parameters in every matrix -- and costs exactly
the same, on one H100, single-threaded:

| algorithm | widths | parameters | s/epoch |
|---|---|---|---|
| `minimum` | `mpnn` | 312 131 | 1.64 |
| `minimum` | `small` | 78 755 | 1.65 |
| `bfs` | `mpnn` | 418 417 | 8.00 |
| `bfs` | `small` | 105 529 | 7.92 |
| `bellman_ford` | `mpnn` | 418 514 | 8.08 |
| `bellman_ford` | `small` | 105 578 | 8.10 |

A round is many small kernels, one per `(name, degree)` group, so a run is
bound by the *launches* and not by the arithmetic inside them: shrinking
the matrices shrinks the only part that was not costing anything. The
ratio that does predict the cost is `calls_per_round` -- `minimum` has two
groups and takes 1.6 s, `bfs` has about fifteen and takes 8.0 s. So the
levers that would make a campaign quicker are fewer rounds, fewer diagrams
per epoch or fewer distinct degrees, all three of which change what is
being measured, and the recorded baseline therefore keeps `mpnn`. The
preset stays because the question is now answered with a number instead of
an intuition.

**The compile cache, counted rather than assumed.** `MapNN.compile` keeps
an LRU of compiled interactions, and since a batch *is* a diagram here, the
whole of the compilation cost above is paid in the first epoch and nothing
after it — *if* the cache holds every diagram at once. `MapNN.cache_stats`
now counts hits and misses so that "if" is a number. On `bfs` at the `FULL`
budget:

| | hits | misses | held | capacity |
|---|---|---|---|---|
| epoch 1, cold | 288 | 32 | 32 | 66 |
| epoch 2, warm | 320 | **0** | 32 | 66 |
| ...and a validation pass | 16 | 8 | 40 | 66 |
| a whole `report()` | 168 | **48** | **66** | 66 |
| the training epoch after it | 288 | **32** | 66 | 66 |

So a training run on its own was never thrashing: 32 training diagrams into
a cache of 66, and a warm epoch is 320 lookups and zero compilations. The
answer to "is it thrashing" is *no, at the size it is run at* — which is not
the same answer as "the size is right". `report()` draws three more splits,
and 32 train + 8 val + 8 test + 32 wide is 80 diagrams into 66 slots:
`held` pins to `capacity`, the test split is evicted by the wide one and
recompiled by the depth sweep behind it, and the last row is a training
epoch that follows a report recompiling **every diagram it just ran** — 32
misses, byte for byte the cold epoch. The old size was
`4 + 2 × (1000 / batch_size)`, an arithmetic guess that was right for one of
the two paths and silently wrong for the other.

`model.fit_cache(model, *batches)` replaces the guess: it sizes the cache
from the batches that exist, is called by both `train.py` and `evaluate.py`,
and never shrinks. `test_a_warm_epoch_compiles_nothing` pins it, and the
warm-epoch counters are recorded in every training artefact and every
report, because this is exactly the kind of leak that costs only wall clock
and so is invisible in a loss curve. It matters more later than now: Part 2's
`floyd_warshall` diagram is the complete graph, about 2016 edge boxes per
sample at `n = 64` against the ~1000 of an Erdős–Rényi one, so a
recompilation that is a nuisance here is the wall-clock story there.

**The residual, as a curve and not only as a number.** `evaluate.py` records
`Interaction.residual` after *every* round, on the in-distribution and the
out-of-distribution split, run past the trained depth -- whether a state
stays put after the rounds it was supervised at is exactly the question a
residual measured *at* that depth cannot answer. This is Part 3's H2
instrument, built in Part 1 on purpose: it is worth more with a baseline
behind it, and the three algorithms are entitled to disagree. `bellman_ford`
*is* a fixed-point iteration, so a model aligned with it should show a
falling curve; `minimum` is a sequential scan with no fixed point to reach,
so a flat or rising one is a fact about the task and not a failure of the
model. Having the two under one instrument is the contrast, and neither
reading is scored.

The first reading already says two things `minimum` alone could not.
**The curve oscillates with period two**, and the two clocks are why: on the
in-distribution split of the mean-pooled model the first rounds read 6.83,
5.62, 3.67, 2.69, 6.53, 3.26, 6.42, 2.99 — a node round and a readout round
are different distances from a fixed point, so the hop law that `HOPS = 2`
asserts is visible in an instrument that was never told about it. Anything
Part 3 reads off a residual has to be read at hop boundaries or not at all.
**And nothing settles**: at the trained depth the residual is 5.5 (mean) and
4.1 (max), and running to three times that depth walks it down to 0.74 and
1.17 without reaching zero, while the score falls from 1.0 to 0.78. So the
run is *drifting*, not converging, and the drift costs accuracy — which is
the honest baseline H2 should be measured against rather than the hope that
an `Iterate` model is secretly a `FixedPoint` one.

**Compilation is superlinear in the boxes of one map, and nothing in the
library was changed about it.** Drawing and compiling the diagrams of a
whole split costs, on this machine, single-threaded: 106 s for the 32
training diagrams at `n = 16`, 63 s for the 8 out-of-distribution ones at
`n = 64`. That is a **one-off per process** — a diagram is compiled once and
reused for the whole run — so it is about two minutes against a
fifty-minute training run.

Half of it is `CMap._box_port_indices` being a `property` where it could be
a `cached_property`; making it one takes those two figures to 35 s and 17 s.
That change was tried, measured and reverted: a study is the wrong place to
edit the core library, and a 2 % overhead is not a reason to. See
`../../NOTES.md` for the numbers and for what would change the verdict.

The example lives with the cost by keeping evaluation batches small
(`Budget.eval_batch_size = 4`) rather than compiling a whole split as one
map — which is *cheaper*, since the cost is superlinear inside a map and
linear across maps.

## The one thing the library was asked for

**`"max"` in `discopy.neural.cells.POOL`.** The aggregator is the only part
of a message passer that a *change of size* can see: a mean and a sum both
rescale when a node's degree grows from `n = 16` to `n = 64`, so a model
that learned an extremum at one size reads a different number at the other,
whereas a max stays inside the range of its members however many there are.
`minimum` is an extremum, the relaxation step of `bellman_ford`
is an extremum, and the CLRS paper reports max aggregation as decisive for
exactly these; a size-generalization study cannot be run without the
choice being available.

It is one line in a dictionary, it changes no default and no
recorded number, it is exactly permutation-equivariant rather than
equivariant up to the reordering of a float sum, and it is tested in
`test/neural/test_general.py` rather than only in the study. That is the
bar the reverted performance fix did not clear, and the difference is not
size: it is that one of them is a capability the library was missing and
the other was a convenience.

## Left alone, on purpose

**`adj` is not encoded.** An edge box exists exactly where `adj` is true off
the diagonal, so the diagram already says everything `adj` says; encoding it
would write a learned constant onto every edge. `A` *is* encoded, on the
carried weight loop, and for `bfs` that is the same constant — harmless, and
it keeps one code path for the two algorithms.

**The readout relation is a `Site`, not a `Relation`.** A `Relation` has a
single orbit and no traced role, so a stateless readout could not carry the
graph-level state the brief asks graph-level features to be read off. The
cost is one more recurrent cell in the parameter count; the benefit is that
`minimum` has somewhere to accumulate.

**Model selection on 32 validation trajectories is coarse, and the coarseness
is visible.** CLRS-30's validation split is 32 trajectories; for a
``mask_one`` output like `minimum`'s that is one decision per trajectory, so
the validation score can only take 33 values and swings by a tenth between
neighbouring epochs of a converged run — after the decoder fix `minimum`
still oscillates between 0.9375 and 1.0000 from epoch 100 to 300, on a
model whose output loss is 0.001. Keeping the best-scoring weights is
therefore partly a lottery, and "1.0000" here means "31 or 32 of 32", not
"solved". It is the benchmark's protocol and it is kept, but a study that
wants to compare two rows should compare them over seeds, which is what
`evaluate.summarise` reports.

**The larger out-of-distribution split is reported beside the canonical one,
never instead of it.** 32 trajectories is what the literature compares on;
`WIDE` adds 512 more from a seed of its own, of which `Budget.n_wide` are
scored, purely so that a difference between seeds can be told from noise.

**The anchors are `None` until they are transcribed.** `config.ANCHORS`
holds `None` for every floor and ceiling, and every reporting path prints a
gap only where a number has been filled in. The figures in `project.md` are
marked there as being from memory; they stay out of the code until someone
reads them off Ibarz et al. (2022), table 1.

*Part 2 read them off, and it is table **2**, not table 1.* They are in
`config.ANCHORS` with `config.ANCHOR_SOURCE` beside them, and the
from-memory figure for `dag_shortest_paths` was wrong by ten points -- 88
recalled against 98.19 published. Every other one was within a point, which
is exactly why the rule is worth having: a set of figures that is right
seven times out of eight is indistinguishable from one that is right eight
times, until it is checked.

# Notes from Part 2

Five algorithms, three new locations to decode, one direction, and one
protocol change that invalidates Part 1's own numbers. What follows is
what the brief did not say, what it said wrongly, and what each decision
cost.

## The change the library was *not* asked for

The budget for `discopy/neural` was "small changes", and Part 1 spent
40 lines of it: `"max"` in `cells.POOL` and the `cache_stats` counters.
Part 2 spends **nothing**, and the candidate that came closest is worth
recording, because the rule is that the next one is argued before it is
written.

**The candidate.** H1 asks for the same diagram with `ESTATE → Dim(0)`.
That erases the state *ports*, which the library does exactly right --
and then `Site.__init__` raises `"a site needs a state to carry"`, because
a `Site` is a recurrent cell and a recurrent cell without a state is not
one. The change would have been a `Site` that tolerates an empty
`states`, emitting from its pooled encoding instead of from a recurrence:
about eight lines, behaviour-preserving on every existing model, golden
-gated.

**Why it was not made.** It is not a capability the library was missing;
it is a capability *this study* wants, and the difference is the same one
Part 1 drew between the `"max"` pooling (a reduction a size-generalization
study cannot run without, tested in `test/neural/test_general.py`) and the
reverted `cached_property` (a 2 % wall clock). A `Site` with no state is
not a site: it is a *different cell*, and the honest place for a different
cell is the file where the task's artefacts live. So Part 2 writes
`model.Link` -- thirty lines subclassing `cells.Cell` -- and gets two
things the library change would not have given:

* the **directed** edge in the same class. `dag_shortest_paths` needs a
  cell whose two ports are distinguishable, which no stock cell is
  (`Gate` is `Sym.NONE` but arity-fixed and rewrites its carried roles);
* **one axis** in the H1 comparison. Both arms are the same class with
  the same encoder and the same emission, differing in whether the belief
  comes from a recurrence or from the round's own pooling. Had the arms
  been `Site` and `Relay`, the ablation would have varied an
  implementation as well as a memory.

The cost is that `Link` is not covered by the library's golden gate. It is
covered by `check_equivariant` in float64 (residual 0.0 exactly, in the
undirected case), by the smoke test's shape and gradient assertions, and
by the same doctests every cell in `cells.py` carries.

## What the brief got wrong, again

**The directed-edge cell is not the shape of Part 2, and Part 1 already
said so.** `project.md` puts "the directed-edge cell in `model.py` (the
~30-line custom Cell)" first among Part 2's deliverables. It is one
algorithm out of eight, it took the same thirty lines the brief predicted,
and it was the *last* thing that mattered. What Part 2 actually turns on
is the edge-level decoders and the diagram they read from.

**`matrix_chain_order`'s intervals are node pairs -- and it has no graph
at all.** The brief expects it to be "the same shape" as `floyd_warshall`,
and it is, but only because both are drawn on the complete graph: its
sampler is `SortingSampler`, which draws a sequence and no adjacency
matrix whatsoever. Its edge boxes therefore carry *no input* -- there is
no weight to encode -- and an interval `(i, j)` learns what it is from the
two node states that reach it. That it works at all is a property of the
readout relation and `pos`.

## The trajectory rule is the benchmark's own rule

Worth stating plainly, because it changes what Part 1's numbers were:
`clrs._src.nets.Net.__call__` sets ``nb_mp_steps = max(1, hints[0].data.
shape[0] - 1)`` and masks each sample with ``_is_not_done_broadcast
(lengths, i)``. The published baselines therefore run **one message-passing
step per algorithm step, for as many steps as the batch's trajectory** --
which is the trajectory rule, and not a constant. So a fixed 16 rounds at
`n = 64` was not a stricter protocol than the anchors', nor a looser one:
it was a *different* one, and comparing against Ibarz et al.'s table under
it would have been meaningless. This is the strongest argument for the
change and it is an argument from the reference implementation rather than
from taste.

Two differences from that reference remain, both deliberate:

* a step is :data:`~model.HOPS` **rounds** here, because a node reaches a
  node through a box on a factor graph, where CLRS's processor is a dense
  message passer and reaches it in one;
* the run is `max(lengths)` steps, one more than CLRS's
  `max(lengths) - 1`. The extra checkpoint carries no hint -- there is
  none to carry -- and exists so that the last sample's *output* is
  supervised at a checkpoint of its own rather than at the clamp. It is
  what makes "reach the answer and stay there" trainable for a sample
  whose trajectory is the longest of its batch.

## The protocol change, and what it cost

**`rounds = HOPS × max(lengths)`, per batch.** The reasoning is in
`README.md`; here is the bill.

| algorithm | steps `n = 16` | rounds | steps `n = 64` | rounds | edge boxes / batch | calls / round |
|---|---|---|---|---|---|---|
| `minimum` | 16 | 32 | 64 | 128 | 0 | 2 |
| `bfs` | 7 | 14 | **4** | **8** | ~1000 | 13 |
| `bellman_ford` | 8 | 16 | 9 | 18 | ~960 | 12 |
| `dijkstra`, `mst_prim` | 17 | 34 | 65 | 130 | ~960 | 12 |
| `floyd_warshall` | 16 | 32 | 64 | 128 | 3840 | **3** |
| `matrix_chain_order` | 15 | 30 | 63 | 126 | 3840 | **3** |
| `dag_shortest_paths` | 49 | 98 | 187 | 374 | ~1950 | 13 |

(The steps are the longest trajectory of the split, which is what a batch
runs at; the edge boxes and calls are for a batch of 32 at `n = 16`.)

The `bfs` row is the one nobody predicts. Its out-of-distribution
trajectories are **shorter** than its in-distribution ones -- 4 steps
against 7 -- because an Erdős–Rényi graph at `p = 0.5` has a diameter of
two or three whatever its size, so a bigger `bfs` sample is an *easier*
one for a wavefront. Under a fixed 16 rounds that was invisible; under the
trajectory rule it is the protocol, and it is worth knowing before reading
`bfs`'s out-of-distribution number as evidence about size generalization.
The tasks whose depth genuinely grows with `n` are the other seven.

Two things fall out of that table that no amount of reasoning would have
given.

**The complete graph is the cheap diagram.** `floyd_warshall` has four
times the edge boxes of `bfs` and costs *less* per epoch (11.8 s against
14.1 s on 128 training trajectories, one H100, single-threaded), because a
round is bound by its kernel launches and a complete graph has exactly one
degree: 3 batched calls against `bfs`'s 13. Part 1 measured that
`calls_per_round` predicts the cost and the parameter count does not; this
is the same law read in the other direction. What the complete graph does
cost is *drawing*: 31 s against 4 s, and superlinear in the boxes. It is
paid **once per size** rather than once per batch, because the wiring of a
complete graph does not depend on the sample -- `model.dense_graph` is an
`lru_cache` and `test_the_showcase_diagram_is_the_complete_graph` pins
that two batches of a size are the same object.

**`dag_shortest_paths` is the expensive one, by a factor of five, and it
is inherent.** Its hints advance one elementary depth-first operation per
step -- push a node, colour it, pop it -- so a 16-node sample takes up to
49 steps where `bellman_ford` takes 8, and up to 187 at `n = 64`. Under
the trajectory rule that is 98 rounds at train time and 374 at test. The
alternative was to cap it, which is a protocol dodge: the cap would be
exactly the confound the trajectory rule exists to remove.

**`deep=True` retains a checkpoint per round and half of them are dead
weight.** `Iterate(deep=True)` keeps the state after *every* round; the
loss reads every `HOPS`-th. At `HOPS = 2` that is half the retained
states in the backward graph for nothing. This is a deliberate choice, not
an oversight: at Part 2 train sizes the peak is 2.3 GiB
(`dag_shortest_paths`, the worst of the eight) against an 80 GiB card, and
the alternative -- a solver that checkpoints every `k`-th round -- is a
library change to save memory nobody is short of. It is worth revisiting
in exactly one place: Part 3's `Recursion`-versus-`Iterate` memory
comparison, which is run at *train* size and at these new longer depths,
and where the dead half is a real number in the plot rather than a
rounding error. That strengthens the study rather than weakening it: the
one-graph arm is now genuinely deep.

## The floor is max-aggregated, and Part 1 said it was not

The fourth time a structural-sounding symptom reduced to a protocol
defect, and the most expensive one.

`bfs` under mean pooling scores **0.9883 in distribution and 0.4719 ±
0.0171 out of it**, against a floor of 0.9989. The hint curves say the
wavefront is fine -- `reach_h` is 0.43, 0.99, 1.00 across the three
out-of-distribution steps -- and that the *pointer* is what fails:
`pi_h` reaches 0.485. So the model knows which nodes are reachable and
not which neighbour reached them, at `n = 64` and not at `n = 16`.

The obvious reading is "size generalization", and the obvious reading is
where this study has been wrong three times before, so: what is different
about a node at `n = 64`? Its degree. An Erdős–Rényi graph at `p = 0.5`
gives a node about 8 neighbours at `n = 16` and about 32 at `n = 64`, and
a **mean** over 32 messages dilutes the one that carries the parent by
four times what a mean over 8 does. Picking one neighbour out of a
mean-pooled state is exactly the operation that cannot survive that.

Which raises the question Part 1 thought it had answered.
`config.POOL`'s docstring said ``mean`` was the default "because it is
what the message-passing baseline of the CLRS paper uses".
`clrs._src.processors.PGN.__init__` declares ``reduction: _Fn = jnp.max``
and ``class MPNN(PGN)`` does not override it. **The floor is
max-aggregated.** The study has been comparing a mean-pooled model
against a max-pooled anchor and calling the difference architecture.

So the primary campaign is now ``--pool max``, filed under ``full-max-*``,
and the mean campaign is the aggregator ablation it always should have
been -- which is a real deliverable rather than a write-off, since one
axis with both arms measured is what the discipline asks for. The
constant `config.POOL` stays ``mean`` regardless, because it is what
`Budget.tag` names a run against: flipping it would rename every artefact
already written and silently re-read a mean run as a max one. Which
campaign is primary is a fact about the study and lives in `README.md`.

The lesson is not "read the source". It is that **a default with a reason
attached is not a default that was checked**: Part 1's docstring gave a
justification, which is exactly what made it look settled.

## What the trajectory rule did to the first row

`minimum`, mean pooling, one seed, the same recipe in both regimes:

| `minimum` | ID `n = 16` | OOD `n = 64` (32) | OOD (128) | rounds at `n = 16` / `n = 64` |
|---|---|---|---|---|
| Part 1, fixed 16 rounds | 1.0000 | 0.9375 | 0.9297 | 16 / 16 |
| Part 2, trajectory rule | 1.0000 | 0.8438 | 0.7969 ± 0.0700 | 32 / **128** |

The number went **down**, and that is the rule working rather than failing.
`minimum`'s trajectory is `n` steps, so matching it means the
out-of-distribution run is four times deeper than the in-distribution one --
128 rounds against 32 -- where the fixed-depth regime ran 16 at both sizes
and never asked the model to hold an answer for longer than it had been
trained to. Depth generalization is now *on* the critical path instead of
being quietly excluded from it, which is what a study about execution
policy has to want. It is also CLRS's own protocol: its baselines unroll
for the trajectory's length.

The per-probe hint curves say exactly where it goes. In distribution every
probe is at 1.00 for all sixteen steps; out of distribution `i` -- the loop
counter, which advances one index per step -- scores **0.000** at every one
of the sixty-four, and `pred_h` climbs from 0.03 only to 0.16. `min_h`,
the running minimum, is the one that survives: 0.41 at the first step,
0.59 at the middle one and 0.81 at the last -- which is where the output
score of 0.797 comes from. So the model has not learned to count to 64
having been taught to count to 16, and the answer is carried by the one
probe that is a *reduction* rather than a position. That is a specific
failure with a name, visible because the curve is per probe; a pooled hint
score would have read "mediocre".

## Decisions a table would otherwise hide

**The larger out-of-distribution split is 128 trajectories, not 512, and
it is the primary number.** 32 is what the literature compares on and is
reported beside it for comparability; but for a `mask_one` output 32
trajectories is 33 possible scores, and a difference of a tenth between
two rows is not a difference. Every report now carries
`ood_wide_interval`: the mean over trajectories of the *per-trajectory*
score, with a 95 % normal-approximation interval. 512 was dropped to 128
because an edge-level hint of `floyd_warshall` at `n = 64` is an `n × n`
matrix per step and 512 trajectories of them is a gigabyte of cache to
hold a number nobody reads to three decimals.

Note that the interval's mean and the pooled score are *not* the same
number for a `mask` probe, because an F1 is not linear. Both are
reported; neither is a correction of the other.

**H1's arms are parameter-matched by search, not by eye.** An edge cell
without a recurrent state loses a GRU and gains an emission that reads a
hidden vector instead of a state, so the arms do not match by default.
`model.matched` searches the one width that is free to move and
`config.WIDTHS["paired"]` is the recorded answer: `hidden = 208` against
the edge-state arm's 192, which is 0.27 % apart on `floyd_warshall` and
0.26 % on `matrix_chain_order`. `test_the_h1_arms_are_parameter_matched`
holds it under 10 %.

**`matrix_chain_order` is scored on the whole `n × n` matrix, including
the half that means nothing.** Its `s[i, j]` is defined for `i < j`; the
lower triangle and the diagonal are zeros the reference never writes.
CLRS's own evaluation scores the full array, so this study does too and
the number is comparable with the published one -- but it is inflated in
absolute terms, and a model that learned "predict node 0 below the
diagonal" is rewarded for it. Whoever reads that row should read
`hints/msk` beside it.

**The warm-epoch counters are in every artefact, and they are zero.**
Part 1 built `fit_cache` and `cache_stats` because an evicting compile cache
costs only wall clock and so is invisible in a loss curve. Part 2 is where
that mattered -- a `floyd_warshall` diagram is 4384 boxes and 31 s to draw --
and the number says it does not: `minimum`'s second epoch is 1088 lookups
and **0** compilations, and every training record carries the pair under
`compile_cache.training`, with the scoring pass's beside it. The one-line
summary is printed at the end of every run, so a regression would be read
rather than deduced.

**`adj` is encoded on the dense diagrams and nowhere else.** Part 1's
rule was "an edge box exists exactly where `adj` is true, so the diagram
already says everything `adj` says". On the complete-graph diagram it
says nothing of the kind, so `adj` becomes an edge input --
`model.encoded` is the one place that decides it, and `edge_features` in
`dataset.py` the one place that lists it.

## The seed budget, decided before the table

Three seeds for **every row of the eight-task table and both arms of H1**:
24 primary runs plus 6 node-only ones, `config.SEEDS = (0, 1, 2)` used as
it was written rather than overridden with `--seeds 0` for speed, which is
what the first day of the campaign had been doing. The alternative on
offer was to seed the rows that turned out to matter, and that is
selective rerunning wearing a lab coat: the rows that "turn out to matter"
are the ones whose first seed was surprising, and a second seed is then
being spent to confirm or to erase a surprise. Deciding the budget while
the table is empty is the only version of the decision that is not that.

It costs about 27 GPU-hours on two H100s, and the tail is
`dag_shortest_paths`: forty algorithm steps is eighty rounds, and it is
the one task where the trajectory rule is expensive rather than merely
honest. Its three seeds run concurrently for that reason -- three seeds
of the slowest task in parallel is the same GPU-hours and a third of the
wall clock.

The aggregator ablation is **single-seed and stays single-seed**. It is
not a row of the table; it is the evidence for one protocol decision, and
it is labelled as such wherever it appears.

## Two spreads, two columns, two questions

A row of this study has two independent sources of spread and they are not
interchangeable:

* **over seeds** -- three trainings differing in initialization and batch
  order. This is what the anchors report, and the paper says so outright:
  "error bars represent standard error of the mean across seeds (3 seeds
  for previous SOTA experiments, 10 seeds for current)". So the anchor's
  field is named `sem` in `config.ANCHORS`, not `std`, and the column this
  study prints beside it is the same statistic. It answers *would another
  initialization have given this*.
* **over trajectories** -- 128 out-of-distribution samples scored one at a
  time, with a 95 % normal-approximation interval. It answers *can this
  split resolve a difference this small*, and it does not shrink when a
  seed is added.

`evaluate.summarise` records `std`, `sem` and `half_width`; `tabulate`
prints the last two in separate columns with a legend under the table, and
H1's table carries the delta with the standard error a difference of two
independent means has, `sqrt(sem_a**2 + sem_b**2)`. A single `±` would
have been a number that means one of two things depending on who reads it.

## The anchor column, checked in the direction that hurts

`project.md`'s `dag_shortest_paths` figure was wrong by ten points and the
transcription caught it -- which is an argument for the transcription and
*not* an argument that the transcription is right. The failure mode it
does not rule out is worse: a column wrong by table-column looks
authoritative. So the three things that had to be true were checked
against the source rather than against memory.

It is the **single-task** table -- "Single-task OOD micro-F1 score of
previous SOTA Memnet, MPNN and PGN [5] and our best model Triplet-GMPNN",
one model per algorithm, which is what this study trains. The paper's
multi-task generalist has its own per-algorithm numbers, in Figures 3 and
5, and none of them is used here. Floor and ceiling are the **same table
and the same row** -- the MPNN and Triplet-GMPNN columns of Table 2 -- so
the pair is internally comparable and not two protocols side by side. And
the sizes are the ones this study runs: "we train the model on samples
with sizes n <= 16" and evaluate "on OOD samples of size n = 64".

`config.ANCHOR_SOURCE` carries all of it -- table, caption, columns,
sizes, the meaning of the error bar, and that the digits were read three
times under different prompts and cross-checked against the table's own
overall average -- and every report copies it beside the numbers.

## The signature of an edge is a function of the sampler

`Link` is one cell class and it is *not* "the directed cell". Its
signature is an argument: `edge(algorithm in DIRECTED)` in `model.build`,
which is the single call site, and `dataset.DIRECTED` is a one-element
tuple read off `--survey`. Six edge cells are `Sym.PERM`, pool their two
legs, and owe the equation an orbit owes; one is `Sym.NONE` and owes
nothing; `minimum` has no edge cell to ask about.

This is worth an entry because the substitution the formalism exists to
prevent is exactly the convenient one: a cell whose ports are always
distinguishable would have been simpler to write, would have trained
identically on the undirected seven, and would have cost the study its
measurement basis for H4 -- `check_equivariant` has nothing to check when
the group has no generators, so six residuals would have quietly become
six empty dictionaries. `evaluate.equivariance` measures every cell of
every model, edge cells included, and
`test_an_edge_is_an_orbit_wherever_the_sampler_is_undirected` asserts, per
algorithm, both the signature and that a residual was measured for it.

## What the library gate caught that the example's own tests could not

The example's suite is 96 tests and it passed while one of them was
running nothing at all. Under the whole `test/neural/` run --
which the example alone never exercises -- the sudoku example's `config`
is in `sys.modules` under that name by the time the CLRS example's
doctests are collected, and `doctest.DocTestFinder` decides whether a
class belongs to the module it was handed by looking its `__module__` up
*there*. Every class of the CLRS `config` was therefore skipped, the
module's own docstring has no examples, and `testmod` returned
"0 attempted, 0 failed" -- which is a pass unless something asserts
otherwise. `assert found.attempted` is what turned it into a failure, and
the fix is four lines that put the example's modules back under their own
names for the duration.

The general rule it is an instance of: an example's tests guard the
example, and only the whole-suite run guards the claim that the example's
tests ran. `test/neural/ test/cmap.py` is 227 passed, 1 skipped, 5
deselected (the `neural_e2e` marker) after the 40-line library diff, and
`test/neural/test_equivalence.py`'s golden gate -- 43 tests, all four
recorded sudoku models bitwise in float32 and float64 -- is inside that
number.

## H2's first artefact: where the map settles against where the algorithm does

`bellman_ford`, the aggregator-ablation run (`mean`, one seed) because it
is the one that finished first; the primary `max` row will redraw the same
figure. Read off `artifacts/full-bellman_ford-report.json`, which now
carries `settles` beside `residual_curve` so that the two are one figure
and not two.

The algorithm stops moving at **round 10** (median; 8-10 at `n = 16`,
10-12 at `n = 64`), and the model's residual falls off a cliff over rounds
**10 to 14** -- 2.18, 1.47, 1.05, 0.72 -- then flattens at about 0.6.
The learned map settles where the algorithm does, to within a couple of
rounds, on both splits and without having been asked to: nothing in the
loss mentions a fixed point, only per-step hints.

Then it drifts. Past round 20 the residual *rises* again, 0.57 to 0.90 by
round 36, and the depth sweep says what that costs: 0.376 at the trained
depth, 0.327 at 1.5x, **0.073 at 3x**. So the honest statement of the
baseline is two sentences, not one -- the map settles on the algorithm's
own schedule, and it does not *stay*, which is exactly the gap a
`FixedPoint` solver and Part 3's H2 are for. A run that had only the
scalar residual at the trained depth would have reported 0.72 and neither
half of it.

The per-probe settling is the reason the overall number is a maximum: on
`bellman_ford`'s out-of-distribution split `msk` stops at round 4 and `d`
at round 10, and a model that matched the mask's schedule would look
aligned while getting the distances wrong. `minimum` is the extreme case
-- `pred_h` never changes at all, `min_h` settles at round 68, and the
loop counter `i` runs to round 126 -- which is the same finding the hint
curves gave from the other side.

## The second change the library *was* asked for, argued first

**What.** `discopy/neural/core.py`, one decorator: `CMap.port_widths`
becomes a `cached_property` instead of a `property`. Nine of the eleven
lines are the docstring saying why.

**Why.** `port_widths` rebuilds `CMap.ports` -- a `Port` namedtuple per
port of per box -- and `CMap.forward` reads it on **every call**. One call
is many rounds when a solver iterates, so training never noticed. But
`Interaction.residual` is `advance(state, 1)`, i.e. exactly one round per
call, so a residual curve calls `forward` once per round and rebuilds
every port of the diagram each time. On `floyd_warshall`, whose diagram
has a box per *pair*, that is the whole cost: measured at `n = 16`, one
residual call was 36.9 ms of which 25.4 ms was `port_widths`, and the
scoring pass sat at 100 % of one core with the GPU at 0 % for two hours
before `py-spy` was pointed at it. Cached: **3.6 ms, a 10.3x speedup**,
and the factor grows with the number of boxes, so it is larger at
`n = 64` and larger again on the complete graph.

**Why it is not a number.** A cached tuple of integers and a rebuilt one
are the same tuple: the boxes are fixed by `CMap.__init__` and nothing in
`discopy.neural` mutates them afterwards. The claim is checked rather
than asserted -- `test/neural/test_equivalence.py`, the golden gate, is
43 tests over all four recorded sudoku models in float32 and float64, and
it passes unchanged. `module_list`, five lines below on the same class,
is already a `cached_property` for exactly this reason, so this is the
existing convention applied to the attribute that was missed rather than
a new one.

**Why it is worth the diff.** It is not this study's convenience: H2 in
Part 3 *is* the residual, measured per round, over the deepest runs of
the campaign, and a `FixedPoint` solver's stopping criterion calls it
every iteration of every batch of training. The library diff goes from
38 insertions to 49, which is the price of Part 3 being runnable at all.

The alternative considered and rejected: caching `CMap.ports` itself in
`discopy/cmap.py`. It would help more callers, and that is precisely the
argument against making it here -- `ports` is read by drawing, by
rewriting and by the hypergraph path, none of which this study exercises,
and a stale cache there would be a correctness bug in code no test of
this branch runs. The narrow fix is the one whose blast radius the
golden gate actually covers.

## The eight rows split by depth, and by nothing else

The table's shortfalls are not spread over the eight tasks in the way an
architectural lesion would spread them. They line up exactly with whether
the sampled trajectory grows with `n`:

| | steps at `n = 16` → `n = 64` | OOD | at its trained depth |
|---|---|---|---|
| `bfs` | 7 → 4 | 0.8556 | 0.8501 |
| `bellman_ford` | 7 → 8 | 0.5737 | 0.5703 |
| `dag_shortest_paths` | 46 → 175 | 0.5631 | (running) |
| `matrix_chain_order` | 15 → 63 | 0.3459 | 0.6607 |
| `minimum` | 16 → 64 | 0.7005 | 0.9479 |
| `dijkstra` | 17 → 65 | 0.0610 | 0.6258 |
| `mst_prim` | 17 → 65 | 0.0320 | 0.2150 |
| `floyd_warshall` | 16 → 64 | 0.0728 | 0.2379 |

The two rows whose trajectory does not grow are the two that hold up, and
their score at the trained depth is their score. Every row whose
trajectory grows loses between a third and nine tenths of its answer
between the depth it trained at and the depth its own execution asks for.
`floyd_warshall` at its trained depth is 0.2379 against a floor of 0.2674,
and `matrix_chain_order` 0.6607 against 0.7984 -- both within reach --
while at their own depth they are 0.07 and 0.35.

Three things follow, and the third is the one that matters.

**It is not the aggregator.** That was the previous protocol defect and it
is fixed: `max` moved `bfs` from 0.4719 to 0.8556. The ladder is measured
on the fixed campaign.

**It is not the edge state.** H1's two arms are both in it, and the
node-only arm is *better* on `floyd_warshall`, which is the opposite of
what an edge-memory lesion would predict.

**It is the map not being stable under iteration**, which is the same
statement as the residual curve rising past the trained depth and as
`bellman_ford` scoring 0.059 at three times its depth. The study now has
that claim in three independent measurements -- a residual per round, a
score per depth, and a hint curve that decays with the step index -- and
they agree. It is H2, it is Part 3's, and `FixedPoint` is the solver that
speaks to it.

The comparison this makes possible is worth stating in advance: CLRS's own
baselines re-encode the algorithm's hints as *inputs* at every step, so
their state is refreshed from outside the recurrence sixty-five times.
This study's model does not -- the diagram carries everything -- and it is
that difference, not the aggregation or the pair memory, that the eight
rows are measuring. Part 3 should say so and then measure it, rather than
adopting hint re-encoding and quietly closing the gap.

## The column that read as a size

`OOD (128)` was the header of the primary column. Every out-of-distribution
number in this study is at `n = 64` — `config.WIDE` is
`{"num_samples": 128, "length": 64, "seed": 30}`, so the 128 counts
*trajectories* — but a reader who has not opened `config.py` sees a size,
and a reader who sees a size sees a table comparing our `n = 128` against
anchors published at `n = 64`. That is a category error, and it is the kind
that is fatal to a comparison rather than merely untidy: an anchor row that
looks like it crosses two sizes is worth nothing.

It did not happen, and the header still caused it: the first careful reader
of the table reached exactly that conclusion and rewrote the headline
finding around it. So the defect is real and it is in the reporting, not in
the protocol. Every parenthesis now says `traj.`, `tabulate` prints a legend
line stating that a parenthesis counts trajectories and not nodes, and the
docstring says why the redundancy is deliberate.

The general lesson is the one this file keeps re-learning in new costumes: a
number is only as good as the label a stranger reads it under. Four
"structural" symptoms have now reduced to protocol defects, and this is the
first one to be a defect in the *description* of the protocol rather than in
the protocol.

## Three mechanisms, and the two rows that showed there was more than one

The previous section of this file said the eight rows "split by depth and by
nothing else". That was wrong, and the two rows that refute it are the two
that were supposed to be controls:

| row | ladder: trained → half → own depth | floor | gap |
|---|---|---|---|
| `bfs` | 0.8501 / 0.2607 / 0.8592 | 0.9989 | −0.14 |
| `dag_shortest_paths` | 0.5591 / 0.5985 / 0.5721 | 0.9624 | −0.40 |

Neither moves with depth — `bfs` because its graphs get *shallower* as they
grow (7 steps at `n = 16`, 4 at `n = 64`), `dag_shortest_paths` because its
ladder is flat inside its own seed spread — and both are far below floor. A
mechanism that is absent from a row cannot explain that row's gap. So:

* **M1, depth.** The five rows whose trajectory grows with `n` decay
  monotonically with rounds run. Measured three ways that agree: score per
  depth, residual per round, hint score per step.
* **M2, the `argmax` heads.** Out of distribution the `mask` probes are
  essentially exact and the `pointer`/`mask_one` probes are where every row
  loses. A `mask` is a sigmoid per node — one candidate, whatever `n` is. A
  pointer is a softmax over the nodes — 16 candidates in training, 64 out of
  it. `bfs` is the clean case: `reach_h` reaches **1.000** at `n = 64`, so
  the model computes BFS correctly, and `pi_h` sits at 0.845, which is its
  score. The whole `bfs` gap is one head.
* **M3, heads that never execute.** See below.

Note what M2 is not, since the obvious suspect is already spent: the
aggregator was the *previous* protocol defect, `max` is degree-stable, it is
what the floor does, and it is already the primary campaign. A degree-stable
pooling does not fix a softmax over four times as many candidates. The
remaining size effect is in the decoder, and it is a different object.

## Executor or shortcut, and why the answer is per probe

`dijkstra` scores 0.6258 with 17 of its 65 extractions run. Two readings:
an executor whose iteration diverges, or an `n = 16` shortcut that degrades
gracefully when under-run. They are not the same finding — H2 presupposes
that a round approximates a step, and if the rounds never approximated the
steps then `FixedPoint` is looking for a fixed point that was never trained
for.

`evaluate.tracking` decides it, on data already on disk. The trick is to
read the **best** out-of-distribution step rather than the last: both
readings end low, and only the best separates "reached the algorithm and
lost it" from "never reached it". The figure needed the same change in
picture form — plotted against the absolute step index the two splits are
15 steps and 63 steps on two different axes, so against the *fraction* of
the trajectory instead.

The answer is **both, per probe**:

* Executors that drift: `floyd_warshall`'s `Pi_h` 0.806 → 0.083,
  `dag_shortest_paths` four probes near 0.8 fanning to 0.1, `dijkstra`'s
  `pi_h` peaking at 0.604 a quarter of the way in. H2 is aimed correctly
  here, and the 0.6258 datum needs no shortcut to explain it: for those 17
  steps the model really is computing.
* Heads that never execute: `minimum`'s `pred_h` never exceeds 0.141 at any
  step against 1.000 in distribution — while `minimum` scores 0.70 on its
  output; `matrix_chain_order`'s `pred_h` peaks at 0.099;
  `floyd_warshall`'s `k` peaks at 0.094 against a perfect 1.000. All three
  are the same shape: a head whose target is a **global index into the node
  set** — a loop counter, a chain predecessor — rather than a neighbour.
  Sixteen positions to memorize at training size, sixty-four at test size,
  forty-eight never seen.

So Part 3 is well-posed on five rows and not on three heads, and that is a
result rather than a caveat. It is the first figure of Part 2 because it
decides how every other number in it should be read.

One measurement caveat that the figure carries and the ratio does not:
`hint_curve` scores each step over the trajectories *still running*, so a
curve's last point is computed on the deepest samples only. `bfs`'s
in-distribution `pi_h` of 0.845 is that survivorship and not an undertrained
head — the same weights score 0.9928 on the output of the same split. An
endpoint is not a level; the `reached` ratio compares like with like.

## What three seeds can support

H1's delta was reported as "three and a half standard errors", which invites
a 1.96 threshold that does not apply at three seeds. Two corrections, and
the second is the one worth keeping:

Welch's degrees of freedom for `floyd_warshall`'s comparison are **2.9**,
where the two-sided 95 % threshold is near `|t| = 3.2`; `t = −3.74` clears
it by little. And the distribution-free reading is sharper: the two arms are
*perfectly* separated — every edge-state seed below every node-only seed —
and the exact permutation test over all 20 ways of dealing six seeds into
two arms of three still returns `p = 0.10`, because **0.10 is the floor**.
No three-versus-three comparison can do better, however large the effect.

`evaluate.significance` computes both and `h1_table` prints them, so the
ceiling on what the design can claim is printed beside the claim rather than
left for a reader to derive. The word is *suggestive*, and it would be
suggestive at this sample size even if the effect were enormous.

The confound is worth stating in the same breath, because it runs in the
direction observed: an edge state is `n²` more things to drift, and both
arms are inside M1. Compounding drift *predicts* the sign of H1's result.
So the negative is not merely underpowered, it is confounded, and H1 has to
be rerun on a stable map rather than reinterpreted on this one.

## The two root-cause changes, and why neither touches the library

The rule for this study is to fix what generates the invalid regime rather
than to route around it. M1's generator is visible once stated: **at a
single training size every trajectory has nearly the same depth**, so
"iterate further than you ever have" was never in the training distribution
and the depth ladder is the measurement of what that costs.

**T2, mixed training sizes.** `config.MIXED = (8, 10, 12, 14, 16)`, 200
trajectories each, so the *budget* stays the benchmark's 1000 and only its
shape changes — which also keeps the data-budget parity with the anchors
that the brief asked to be stated. Each size is its own cached split
(`train8` … `train16`) and `Batches.over` keeps every batch homogeneous in
`n`: a batch is one compiled diagram, so a ragged batch would be a diagram
per combination of sizes instead of one per size, and what mixing is *for*
— several trajectory depths in the distribution — a homogeneous batch does
just as well. The compile cache still hits 100 % on a warm epoch.

It is a controlled intervention rather than a blanket one, which is the
property that makes the before/after readable: `dijkstra` now trains at 9,
11, 13, 15 and 17 steps instead of 17 alone, while `bfs` sees 7 to 9 either
way. It should move the M1 rows and leave the two flat rows alone.

**T3, termination in the loss.** A converged algorithm emits a constant
hint. The output loss has always been supervised "from the end of the
trajectory onwards" — reach the answer and stay there — and the hints were
not: a finished sample was *dropped* from the hint loss, so what the map
does with a converged state was never optimised. `Model.hint_targets` under
`settle` clamps each sample's hint index at its own last step instead, so
the checkpoints past its end are supervised on its final hint repeated.
That is a basin at termination, put in the loss rather than hoped for — and
Part 3 then *measures* a basin instead of measuring the absence of one it
never trained.

Both are `dataset.py` / `config.py` / `model.py` / `train.py`, all of them
in the example. **Zero library code**: `discopy.neural` is unchanged, and
the golden gate says so. The one library change this part took remains the
`cached_property`, and the check a reviewer asked for it is now written —
`test_a_map_keeps_the_shape_it_was_built_with` asserts the premise the cache
rests on, that a map's boxes, ports, wiring and recomputed widths are
identical after it has been advanced and asked for residuals, since a map
mutated after construction is the one way that cache goes wrong and it would
go wrong silently.

## The guard that could not fail

`test_the_examples_own_doctests` was fixed last round after it was caught
passing having run nothing: with another example's `config` in
`sys.modules`, `doctest.DocTestFinder` skips every class and `testmod`
returns "0 attempted, 0 failed". The fix restored the example's modules and
`assert found.attempted` was added to catch a recurrence.

Except it could not catch one on its own. The collision arrives only when
something else has already claimed the name, so removing the fix and running
the file alone still passed — 96 green — and only a whole-suite run failed.
A regression guard whose failure depends on what else pytest collected, and
in which order, is not a guard.

It now **plants** the collision: a decoy `ModuleType` under each shared name
before the restore runs. Removing the fix now fails the test standing alone.
Verified by doing exactly that, in both directions, before writing this.

The general form is worth keeping, because this file has now recorded two
instances of it: a test written in response to a silent failure has to be
shown to fail. Writing the assertion is not the same as arming it.

## What arm C actually measured: the labels, not the model

The `pos` ablation was read as "the model leans on `pos`". That was wrong,
and the correction is the most useful thing this part has produced.

CLRS's reference algorithms iterate in **index order**, so their labels are
tie-broken by it. `_bfs` is the clean case: `for i in range(size)` ascending,
`if parent[j] == j: parent[j] = i`, i.e. the parent of `j` is the
**lowest-indexed** already-reached neighbour. The pointer targets are
therefore functions of node order and not of graph structure alone. `pos =
arange(n)/n` is the only input that carries that order. Permute it and the
tie-breaking information is erased from the inputs while remaining in the
targets: among `k` tied candidates no model of any capacity beats `1/k`.
The task stops being realizable.

Measured on the cached splits, counting the candidates the reference's rule
was choosing among at each assignment:

| `bfs` | assignments | tied (`k>1`) | mean `k` | max `k` | order-blind ceiling on `pi` |
|---|---|---|---|---|---|
| `n = 16` | 467 | 31.7 % | 1.40 | 5 | **0.8444** |
| `n = 64` | 8064 | **69.7 %** | 3.38 | 20 | **0.5012** |

Against which the two shuffled arms are pinned:

| arm | ID `n=16` | ceiling | OOD `n=64` | ceiling |
|---|---|---|---|---|
| C, bilinear head | 0.8711 | 0.8444 | 0.4607 | 0.5012 |
| D, edge-aware head | 0.8535 | 0.8444 | 0.4905 | 0.5012 |

Both sit on the bound at both sizes, and the *ID* deficit is the signature
that separates the two hypotheses: a model leaning on a feature to its cost
fits in distribution and fails out of it, and a model facing partly
unrealizable labels fails in distribution. These fail in distribution, by
about the tie mass. So the arms measured the dataset, not the model, and
nothing about M2 follows from either.

The reference authors hit the same wall and their fix says so. Ibarz et al.
§3.2.2, on the position scalar: *"we replaced them with random values,
uniformly sampled in [0,1], **sorted to match the initial order** implied by
the linearly spaced values."* Randomised **values**, preserved **rank** —
because rank is part of the task's definition. `Split.repositioned` now
keeps the two apart by construction: `"shuffled"` destroys rank and is an
ablation of the labels, `"uniform"` destroys only the `1/n` spacing and is
an ablation of the input.

## Order-dependent semantics, not pointers-versus-masks

The right statement of M2 is not "the pointer decoder is deficient". It is
that **every head that fails is a head whose target is a function of node
order** — a tie-broken parent, a loop counter, an argmin index — and the
processor is permutation-equivariant with one order channel that is a
size-dependent scalar. That predicts the dead heads without a second story:
`minimum`'s `pred_h` and `i`, `floyd_warshall`'s `k`,
`matrix_chain_order`'s `pred_h` are all order-dependent and all sit at
`reached <= 0.35`, while the masks — order-free — are essentially exact out
of distribution.

It also says what `reach_h = 1.000` does and does not exonerate. The
processor computes *reachability* correctly at `n = 64`; that is an
order-free predicate. It says nothing about parent selection, which is an
argmin-by-`pos` over neighbours: representable under a max pooling, but not
certified by the reach probe.

## The floor's provenance, checked rather than assumed

The fairness argument ("the floor lacks a randomised position scalar too")
rested on an unverified claim about where Table 2's numbers came from. It
is checkable and now checked: Ibarz et al. state the Memnet/MPNN/PGN
columns were *"taken directly from [Veličković et al. 2022]"* rather than
re-run, so the floor is the original 2022 recipe with none of the
generalist paper's stabilizers. The seeds and the error-bar convention
(3 previous-SOTA, 10 current, s.e.m.) are confirmed from the same source
and were already what `config.ANCHORS` recorded.

## `minimum` under mixed sizes: not the budget

The mixed-size negative was recorded with "plausibly 200 versus 1000
samples per size" as the guess. The control says otherwise:

| `minimum`, seed 0 | ID `n=16` | OOD `n=64` |
|---|---|---|
| 1000 trajectories at `n = 16` | 1.0000 | 0.8047 |
| **200 trajectories at `n = 16` only** | 0.9688 | **0.8281** |
| 1000 trajectories, mixed sizes + settle | 0.9688 | **0.1719** |

At one fifth of the data and a single size the row *holds* — 0.8281, no
worse than the full budget. So the collapse is not a data-budget artefact:
**size mixing itself is destructive on this task**, and T2 cannot be
applied as a blanket protocol change. `minimum`'s heads are the
order-dependent ones (`pred_h`, `i`, both `reached <= 0.35`), and mixing
sizes changes what a given step index means across the batch, which is the
hypothesis worth testing next rather than asserting now.

The tag now records `n_train`, because without it this control silently
loaded the full-budget checkpoint instead of training — a reuse-by-tag
collision that would have produced a clean-looking null.

## The combiner, not the channel

The edge-aware pointer head was adopted on an audit finding — that
`clrs._src.decoders._decode_node_fts` scores a `POINTER` from three terms
including `decoders[2](edge_fts)` and ours scored from two node states —
and it bought 3.4 points on `bfs`. The write-up then said the decoder had
been "discarding the channel the baselines use". That attribution was
wrong, and the third corner proves it.

`EdgePointerDecoder2` changed **two** things against `PointerDecoder`: it
added the edge term, and it replaced the bilinear dot product with a `max`
of broadcast source/target terms through a linear head.
`EdgeBilinearPointerDecoder` is the corner that separates them — the edge
term, the original combiner:

| head | edge term | combiner | OOD `n=64` | parameters |
|---|---|---|---|---|
| `PointerDecoder` | no | bilinear | 0.8958 | 409 969 |
| `EdgeBilinearPointerDecoder` | **yes** | bilinear | 0.8796 | 410 067 |
| `EdgePointerDecoder2` | yes | `max` + linear | **0.9296** | 380 531 |

The edge term on its own is slightly *negative*. The whole gain is the
combiner, and it comes with 29 000 fewer parameters. So the mechanism is an
**order statistic in the head**, not an input channel the model was denied
— which is a different thing to fix and points somewhere else for the
remaining seven points.

Worth stating plainly because it is the fourth time in this study that a
plausible mechanism survived one experiment and died to the control that
separated it from its neighbour. The rule that keeps paying: when an
intervention changes two things, the two-change result is a measurement of
neither.

# Notes from Part 3, before it runs

Everything here was measured before a Part 3 model was trained, which is the
only time these findings are cheap. Four of them changed the design; two
killed something that had already been proposed, one of them mine.

## The cell the 2x2 cannot have

The grid Part 3 was asked for is `{Iterate, FixedPoint} x {settle,
no-settle}`. `FixedPoint x no-settle` does not exist, for two independent
reasons that both reduce to one fact about the trajectory rule.

A run is `batch.steps` checkpoints long and the `k`-th is supervised on hint
`k + 1`, so the last checkpoint asks for hint `batch.steps` -- which
`hint_targets` refuses before it consults `settle` at all. Measured over the
whole training split, the alive fraction at the final checkpoint is **0.0 %
on all eight algorithms**; it is a consequence of `batch.steps =
max(lengths)` and not of a length distribution.

And `FixedPoint(backward="last")` differentiates exactly that checkpoint and
no other. So it receives no hint gradient, `settle` is a no-op for it, and it
is output-only whether or not `hint_weight` says so. The README had already
argued this from the shape of the solver; it is now a number.

The third way out is closed too: with `tol=None` and `inject=False`,
`FixedPoint(backward="full")` is **bitwise** `Iterate`. So `backward` is the
whole of a fixed-point row and there is no variant of it that can sit in a
settle column.

What replaces the grid is in `PART3.md`: four arms, three one-axis
contrasts, and the execution-policy comparison moved to test time where it
costs nothing and confounds nothing.

## `settle` did not reach the checkpoint it was built for

The same guard is a defect in T3. `Budget.settle` was introduced to put a
basin at termination into the loss, and it holds a finished trajectory's last
hint at every checkpoint **except the last one** -- the one state a
`FixedPoint` converges to, which is the only place H2 reads. On `bfs` at
`n = 16` it lifts the penultimate checkpoint from 8 supervised rows of 32 to
all 32, and leaves the final checkpoint at 0.

`config.SETTLE` therefore has three members rather than two, and the repair is
additive: `"interior"` is what the mixed campaign trained under and keeps its
tag, `"terminal"` is what arm `S` trains with. Redefining `settle` in place
would have silently re-labelled an already-trained campaign, which is the
`Budget.tag` failure again in a different coat.

## A fixed point that cannot train an encoder

`FixedPoint(backward="last")` differentiates one round from `state.detach()`.
For an interaction that re-injects its inputs that is correct and the inputs
stay in the graph. These cells are **resumable**: the inputs ride on traced
loops inside the state precisely so that a run can be stopped and resumed, so
detaching the state detaches the encoders with it. Measured on `bfs`: all six
encoder parameters come back with `grad is None`, and nothing else does.

Had that shipped, arm `F` would have trained frozen random encoders and its
comparison with arm `O` would have been two changes wide -- the failure this
study keeps being caught by, and this time it would have been invisible: a
frozen encoder does not raise, it just costs a few points that read as the
differentiation policy.

The repair is free because the roles involved are *carried*: a site re-emits
`FEAT` and `WEIGHT` unchanged, so the limit holds bitwise what the encoders
wrote and writing them back into the detached limit changes no number in the
forward pass. `model.Grounded` does that, `discopy.neural` is untouched, and
two tests pin the two halves.

It is worth saying which side the library is on here. `FixedPoint` is right
for the interaction it was written for; this example's `inject=False` plus a
resumable cell is a combination in which "detach the state" and "detach the
inputs" are the same operation, and nothing in the solver could know that.
The candidate library change -- a solver that is told which port families are
inputs -- is declined for the same reason the last one was: it puts a fact
about a study inside the machinery.

## An output-only arm still has to fit its heads

`hint_weight = 0` was the obvious way to write arms `O` and `F`. It would
have made Part 3's own second rule unreadable: with no hint term the hint
decoders get no gradient, so the hint curves the order-free column is read
from would come out of untrained heads, and an arm would score zero on
order-free mass without its processor having failed at anything.

So an output-only arm detaches the state the hints are decoded from
(`Budget.probe`) rather than dropping the term. The interaction's gradient is
bitwise unchanged -- the axis is intact -- and the hint heads become linear
probes of the state, which is a caveat and also a better question than the
one it replaces: *does a processor trained on outputs alone still contain the
order-free computation?*

## H4's independent variable is a constant

`check_equivariant` on every cell of every trained model of Part 2, float64:

| pooling | `node` | `readout` | `edge` |
|---|---|---|---|
| `max`, the primary campaign | **0.0** | **0.0** | **0.0** |
| `mean`, the ablation | 8.9e-16 | 1.8e-15 | 0.0 |

Exactly zero, not nearly. `POOL["max"]` is order-invariant in floating point,
so at the aggregator the primary campaign uses the equivariance law is
**strict**, not lax, and there is no residual to vary. Under `mean` it is
machine epsilon over the width of a reduction, i.e. a fact about orbit sizes.

H4 asked whether measured symmetry covaries with generalization. In this
formalism the symmetry is exact by construction, so it covaries with nothing,
and a correlate with no variance is not a weak correlate. The verdict is a
negative one under a controlled protocol, which the brief counts as a result
-- and, honestly, one that could have been reached by reading `laws.py` in
Part 1 rather than by measuring eight tasks in Part 2. `evaluate.
equivariance`'s docstring claimed the residual was "the quantity H4 will
correlate against"; it now carries the measurement instead.

## Every output probe of all eight is order-dependent

The amendment to H4 -- make the dependent variable the order-free-head drop,
or partial out order-dependent mass -- is right, and neither half can be
applied at the output level:

| | `minimum` | `bfs` | `bellman_ford` | `dijkstra` | `mst_prim` | `dag_sp` | `floyd_warshall` | `matrix_chain` |
|---|---|---|---|---|---|---|---|---|
| output | `min` | `pi` | `pi` | `pi` | `pi` | `pi` | `Pi` | `s` |
| type | mask_one | pointer | pointer | pointer | pointer | pointer | pointer | pointer |

Every algorithm has exactly one output probe and on all eight it is an
`argmax` over the node set. So order-dependent output mass is 100 % on every
row: the order-free output drop is an empty column, and there is no varying
mass to partial out. It also says something about the benchmark that is worth
stating plainly -- CLRS-30's own micro-F1, and therefore both published
anchors, lives entirely inside the head class M2 says does not transfer.

Both halves are repaired the same way, by scoring the split over the hints as
well as the output, where order-free mass exists and varies. `minimum` has no
order-free probe of any kind, so H4's `n` is **7, not 8**.

One more exclusion, forced by a scale rather than by semantics: a `scalar` is
scored by a mean squared error, unbounded and lower-is-better, so it is
pooled with nothing. Averaging it in is what made `floyd_warshall`'s
order-free drop come out at -0.046, which is not an improvement.

## The type-mass replacement died to its own control

With the residual dead, H4 needed an independent variable that varies, and
the obvious one was M2 promoted to a law: order-dependent mass predicts the
drop. Operationalized as a share of probe types it is dead on arrival --
`r = -0.65`, exact permutation `p = 0.046`, significant **with the wrong
sign**. A share of probe types is a ratio of counts and the tasks with few
probes are the easy ones: `minimum` is 100 % order-dependent mass and drops
0.195, `dijkstra` is 50 % and drops 0.908. It is an inverse proxy for
difficulty wearing a mechanism's name, which is the failure the amendment
existed to prevent, one level down.

Kept in `evaluate.h4_table`'s output labelled as a rejected candidate, so
that nobody derives it again and reports the `p`. What is left is the **tie
rate** -- measured for `bfs` (31.7 % at `n = 16`, 69.7 % at `n = 64`,
ceilings 0.8444 and 0.5012) and needing one function per algorithm, since the
tie-breaking rule is per algorithm. H4 does not run until that exists.

## The worst failure class in this project, named

The `minimum` budget control caught a `Budget.tag` collision: without
`n_train` in the tag, a 200-trajectory run silently loaded the 1000-
trajectory checkpoint and reported it. It would have produced a **clean-
looking null** -- no exception, no warning, a number in the same range as its
neighbours, and a conclusion ("the data budget is not the cause") that was
correct by accident.

That is the worst failure class here, worse than a wrong number, because a
wrong number is eventually contradicted and a stale one agrees with
everything. Two rules follow and both are now in code rather than in habit:
every knob that changes what a run *is* goes in `Budget.tag`
(`n_train`, `settle`, `probe`, `solver`, `backward`), and a protocol
decision that has not been made raises instead of defaulting
(`train.regime_of` on `config.REGIME`).

# Notes from the floor-gap audit (Phases 0–1)

## The parity audit is a file, and what it found is ranked there

`artifacts/parity.md`: every recipe row ours-vs-floor with file+line on
both sides, the floor's provenance settled from the papers' own text, and
the differences ranked by expected leverage.  The two that survive as
suspects: the floor **feeds hints back as inputs at every step** (train:
ground truth w.p. 0.5, else its own hard-decoded predictions; eval: its
own predictions), and the floor **freezes each sample's output prediction
at that sample's own termination step** (`nets.py`, the `is_not_done`
blend) — it is never asked to hold an answer, which with the depth ladder
in mind is an exemption from M1 by construction.  Everything else is
parity or minor, including the losses' forms (verified), the optimizer,
the capacity (414 468 reference parameters against our 418 514 on
`bellman_ford`, counted by instantiating both), and the scoring
(`artifacts/parity-eval-xcheck.json`: pointer and `mask_one` bitwise,
`mask` equal up to the reference's own float32 accumulation).

## The dm-clrs environment had to be rebuilt, and its pins matter

The brief said an isolated dm-clrs venv existed; no interpreter on this
machine could import `clrs`.  Rebuilt at `/scratch/tommaso.salvatori/
dm-clrs`: Python 3.11, `dm-clrs` 2.0.3, jax 0.10.2 + CUDA plugin, and
**`protobuf 6.x` exactly** — 7.x breaks the tfds *read* path
(`FieldDescriptor.label` removed) while 5.x is older than the gencode
`clrs` 2.0.3 ships, so 6.33 is the one band in which both the dataset
reader and the package import.  The generation path works under 7.x,
which is how the pin hid until the second run.

## The cached splits are the published samples, now as a checked fact

`dataset.py --generate`'s claim ("these *are* its samples") rested on
sampler determinism across `clrs` versions.  Checked, twice: per-sample
content hashes of our cache against (i) a fresh tfds generation under
2.0.3 and (ii) **the published `CLRS30_v1.0.0.tar.gz` from the GCS
bucket itself** — multiset-equal on all six `bellman_ford`/`bfs` splits
(element-wise comparison differs only because tfds shuffles sample
order, which cost one wrong first reading).  `artifacts/
floor-dataset-provenance.json`.  The local floor (`floor.py`) trains on
the extracted archive directly, so its provenance ends at the bucket.

## `floor.py` is the reference harness, not a port

Phase 1's local floor runs DeepMind's `BaselineModel` inside the dm-clrs
venv with the 2022 recipe named field by field (`floor.RECIPE`), because
the current repo's defaults are the Ibarz recipe and the two differ in
exactly the stabilizers the floor must not have.  The one knob the 2022
paper does not pin is the validation cadence (v1.0.0's default evaluates
every 10 steps; we evaluate every 100 and record it).  Two shared-GPU
lessons are in the file rather than in a shell history: the tfds reader
will reserve the GPU unless told not to
(`tf.config.set_visible_devices([], "GPU")`), and a BFC pool grab races
the neighbours' allocation bursts, so the runs use
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` — the model is a few MB and the
pool was pure liability.
