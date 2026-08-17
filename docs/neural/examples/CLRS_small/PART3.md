# Part 3, before it runs

This file is the protocol of Part 3, written before a single Part 3 model is
trained. It exists because the two things it decides — H2's grid and H4's
dependent variable — are the two places where a design decided halfway
through is indistinguishable from a result.

Everything below is either a rule, or a measurement that forced a rule.
Where a measurement killed something, the something is named and kept, so
that nobody derives it a second time and reports it.

## The gate, re-scoped

Part 2 ended gated: `bfs` sits at 0.9296 against a floor of 0.9989 and the
remaining seven points are unexplained, so the retrained T2/T3 campaign is
trained and unscored.

The gate was protecting against an *unlocalized* defect. When `bfs` sat 14
points under floor with no account of why, any Part 3 number could have been
that defect wearing a solver's costume. That is no longer the situation:

* the gap is localized to one head class — `reach_h` reaches **1.000** at
  `n = 64`, so the processor computes the order-free part of BFS exactly;
* the tie mass is measured — **69.7 %** of `bfs` parent assignments at
  `n = 64` are decided by index order, order-blind ceiling **0.5012**;
* the recoverable part is identified — the `max`-of-broadcasts combiner, not
  the edge channel, and at 29 000 *fewer* parameters;
* the floor's provenance is verified — Ibarz et al. Table 2's MPNN column is
  Veličković et al. 2022 verbatim, not a re-run.

Part 3 varies solvers, differentiation policy and measured symmetry. None of
those touch the decoder. A localized, constant, decoder-layer deficit shared
by every arm is an **offset**, not a confound.

**The gate re-scopes to what it was protecting: order-free heads at ceiling,
gap localized and frozen. That is met.** The unmet part becomes a standing
constraint, which is rule 3 below.

## The four rules

1. **The decoder is frozen.** `pointer="edge"` (`model.EdgePointerDecoder2`,
   arm `B`) on every compared arm, `config.PART3`. No mid-study change: the
   offset is only harmless while it is constant. One consequence, stated
   rather than discovered later — Part 2's eight-task table was measured with
   the bilinear head, so **no Part 3 number is comparable with it**. Part 3
   carries its own reference row, arm `R`.
   The switch is on the *node*-pointer head; `floyd_warshall`'s `Pi_h` and
   `matrix_chain_order`'s `s_h` are edge pointers and keep
   `model.EdgePointerDecoder`, unchanged across arms like everything else.
2. **Every table splits the heads.** `evaluate.head_table`, order-free
   against order-dependent, `python evaluate.py --heads`. A solver never gets
   credit or blame for pointer points.
3. **No claim against a published anchor.** Part 3 was designed anchor-free;
   the unmet `bfs` gate makes that binding rather than stylistic.
4. **Three seeds for any verdict.** `config.SEEDS`.

## H2: the 2×2, and the cell that cannot exist

### What was asked

`{Iterate, FixedPoint} × {settle, no-settle}` on the five executor rows,
matched parameters, three seeds. The reasoning is right and it is why this
section exists: with `settle` training a basin at termination, a small
`FixedPoint` residual is partly a trained-in property, so the no-settle
column is what keeps *"the learned map converges where the algorithm does"* a
finding rather than a construction.

### What the grid does when you measure it

Four measurements, all on the code as it stands.

**(i) A fixed point differentiates one checkpoint.** `FixedPoint`'s
`backward="last"` is the Jacobian-free one-step gradient: it iterates under
`no_grad` and differentiates a single round from the detached limit.

| solver | checkpoints | differentiated |
|---|---|---|
| `Iterate` | 5 | `[True, True, True, True, True]` |
| `FixedPoint(backward="last")` | 5 | `[False, False, False, False, True]` |
| `FixedPoint(backward="full")` | 5 | `[True, True, True, True, True]` |

**(ii) The one checkpoint it differentiates is the one the hint loss never
reaches.** Under the trajectory rule a run is `batch.steps` checkpoints long
and the `k`-th is supervised on hint `k + 1`, so the last asks for hint
`batch.steps` — which `Model.hint_targets` refuses before it consults
`settle` at all. On `bfs` at `n = 16`, rows supervised per checkpoint:

| checkpoint | 0 | 1 | 2 | 3 | **4 (last)** |
|---|---|---|---|---|---|
| `settle=None` | 31 | 31 | 29 | 8 | **0** |
| `settle="interior"` | 32 | 32 | 32 | 32 | **0** |

Measured over the whole training split, the alive fraction at the final
checkpoint is **0.0 % on all eight algorithms** — it is a consequence of
`batch.steps = max(lengths)`, not of a length distribution.

So a `FixedPoint(last)` arm receives **zero hint gradient**, and `settle` is
a no-op for it. It is output-only whether or not `hint_weight` says so, which
is what the README already concluded from the shape of the solver and is now
a measurement rather than an argument.

**(iii) The other backward mode is not a different solver.** With `tol=None`
and `inject=False`, `FixedPoint(backward="full")` is **bitwise** `Iterate` —
`test_a_full_fixed_point_is_an_iterate` pins it. So there is no third way to
put a fixed point in the settle column.

**(iv) And `settle` is a literal no-op under output-only training** — it
enters through `hint_targets` alone, which the loss multiplies by
`hint_weight = 0`.

**Conclusion: the `FixedPoint × no-settle` cell is empty, twice over.** Under
a fixed-point solver the model is only ever supervised at termination, so the
termination intervention is unconditionally on. Filling that cell anyway
would produce a contrast between "trained with hints" and "trained with no
hints at all" labelled *termination supervision*, which is the two-change
failure this study has now been caught by four times.

Two more measurements came out of building the arms, and both changed one.

**(v) A fixed-point arm cannot train an encoder.** `backward="last"`
differentiates one round from `state.detach()`. That keeps the inputs in the
graph for an interaction that *re-injects* them; these cells are resumable
instead, so the inputs ride on traced loops **inside** the state and
detaching it detaches them. Measured on `bfs`: under
`FixedPoint(backward="last")` **all six encoder parameters have
`grad is None`**, and nothing else does. An arm built on it would train
frozen random encoders, so `O vs F` would again be two changes wide.

The repair costs nothing, because those roles are *carried*: a site re-emits
`FEAT` and `WEIGHT` unchanged, so the limit already holds bitwise what the
encoders wrote there. `model.Grounded` writes them back into the detached
limit before the differentiated round — **the forward pass is bitwise the
library's** and the missing gradient path is restored. That is also the
standard deep-equilibrium gradient, in which `x` stays attached in
`f(z*, x)`; the library's solver is right for a re-injecting interaction and
this example is not one. `discopy.neural` is unchanged.

**(vi) `hint_weight = 0` would make the mandatory per-head split
unreadable.** With no hint term the hint decoders receive no gradient, so an
output-only arm's hint curves — which is what `evaluate.head_split` reads the
order-free column from — would come out of untrained heads. The arm would
score nothing on the order-free column without its processor having failed at
anything, and rule 2 would report an artefact of the instrument.

So an output-only arm is `probe=True` instead: the hint loss is decoded from
a **detached** state, so it fits the hint decoders and never the interaction.
The axis is intact — the interaction sees the output alone — and the hint
heads become what they then are, linear probes of the state, which is a
caveat their numbers carry and a genuinely useful reading: *does a processor
trained on outputs alone still contain the order-free computation.*

One asymmetry, pinned by a test: under a fixed-point arm run at the
trajectory's depth, `probe` changes **nothing at all** — every round but the
last is outside the graph already and the last is the checkpoint no hint
index reaches. So `O` has to declare the probe and `F` gets it for free,
which is exactly what makes the two comparable. (It is a property of the
trajectory rule, not of the solver: at Part 1's shorter fixed depth the
terminal checkpoint is inside the trajectory, is supervised, and the two do
differ.)

### What replaces it

Four trained arms, `config.H2_ARMS`, each pair differing in exactly one
thing. Frozen across all four: `pool="max"`, `pointer="edge"`, widths
`mpnn`, 300 epochs × 32 batches, lr 1e-3, AdamW, clip 1.0, the trajectory
rule, 1000 trajectories, and the per-row size regime of `config.REGIME`.

| arm | solver | differentiated | hints reach | settle | what it is |
|---|---|---|---|---|---|
| **R** | `Iterate` | every round | interaction + heads | none | the reference row |
| **S** | `Iterate` | every round | interaction + heads | `terminal` | the trained basin |
| **O** | `Iterate` | every round | heads only (`probe`) | none | the supervision control |
| **F** | `Grounded` | the last round | heads only | none | the differentiation policy |

The contrasts, and nothing else may be read off the grid
(`config.H2_CONTRASTS`):

* **R vs S** — termination supervision. *How much basin is free and how much
  is trained.* This is the question the 2×2 was asked for, and it survives
  intact: it is the settle axis, at matched solver.
* **R vs O** — supervision regime, hints against none. Needed because the
  hint term is most of the gradient here, so without it a fixed-point row
  differs from `R` in two things.
* **O vs F** — differentiation policy, unrolled against Jacobian-free, at
  matched supervision, matched rounds and matched parameters. This is the
  RQ's axis (iii) and it is the only trainable fixed-point contrast that is
  one axis wide.

**The execution-policy axis moves to test time, where it is clean.** A
solver runs from and to the same flat state, so every arm is evaluated under
both `Iterate` and `FixedPoint` on the *same weights* — an execution-policy
comparison with zero training confound, at no extra training cost. That is
the `{Iterate, FixedPoint}` half of the requested 2×2, realized where it is
askable.

Two rules for that evaluation:

* **the depth sweep runs with `tol` disabled.** A fixed point that stops on
  its residual is depth-robust by construction — running it "3× deeper"
  changes nothing once it has settled — so a sweep with the stopping rule on
  measures the rule and not the map. Depth robustness is a property of the
  learned map and is measured with the rule off;
* **rounds-actually-used is reported separately**, with the rule on. That is
  the compute number, and it is H3's currency rather than H2's.

Correspondingly, **`tol` is never active during training** (`model.SOLVERS`
sets it to `None`): a residual stop shortens the supervised sequence, so a
fixed-point arm trained with it would differ from `O` in the gradient *and*
in the effective depth.

### The `settle` repair, and why it is a new name rather than an edit

`Budget.settle` was introduced to put a basin at termination into the loss.
Measurement (ii) says it does not reach termination: the guard fires first,
so a hold that stops at the interior trains a basin everywhere except at the
state a fixed point converges to — the one place H2 reads.

`config.SETTLE` therefore has three members and `settle` is no longer a flag:

* `None` — drop, Part 2's protocol;
* `"interior"` — the hold as it was implemented, and **what the salvaged
  mixed campaign was trained under**, so its tag keeps meaning what it meant;
* `"terminal"` — the hold that reaches the last checkpoint. Arm `S`.

The repair is additive and the old behaviour is reachable by name, because
re-defining `settle` in place would silently re-label an already-trained
campaign.

### The five rows

`config.EXECUTORS`: `bellman_ford`, `dijkstra`, `mst_prim`,
`dag_shortest_paths`, `floyd_warshall`. These are the rows whose hint curves
show the model executing the algorithm for as long as it has iterated before
and coming apart after, so a round approximates a step and there is a fixed
point to look for. `minimum` and `matrix_chain_order` are out on their
`pred_h` never exceeding 0.14 at any step of an `n = 64` trajectory, and
`bfs` is out because it never iterates past its trained depth at all.

`dag_shortest_paths` is in as the **negative control**: its ladder is flat, so
M1 is absent on it, and a solver that fixes depth instability should not move
it. A design where the control is also the most expensive row is not ideal,
which brings us to:

### What it costs

Per 300-epoch run, measured from Part 2's own `seconds_per_epoch`:

| row | h / run | × 4 arms × 3 seeds |
|---|---|---|
| `bellman_ford` | 1.28 | 15.4 |
| `dijkstra` | 2.14 | 25.7 |
| `mst_prim` | 2.08 | 25.0 |
| `floyd_warshall` | 3.40 | 40.8 |
| `dag_shortest_paths` | 9.75 | **117.0** |
| | | **224 GPU-hours** |

**Those figures are wrong and the table is kept as the correction.** They
were built from Part 2's recorded `seconds_per_epoch`, which were measured
with three seeds sharing a GPU — contended-throughput numbers used as if
they were per-run costs. Measured solo on the campaign's own environment,
one optimizer step at a time:

| row | recorded | solo | × 4 arms × 3 seeds |
|---|---|---|---|
| `bellman_ford` | 1.28 h | **0.46 h** | 5.6 |
| `dijkstra` | 2.14 h | **1.07 h** | 12.8 |
| `mst_prim` | 2.08 h | **0.98 h** | 11.8 |
| `floyd_warshall` | 3.40 h | **0.35 h** | 4.2 |
| `dag_shortest_paths` | 9.75 h | **2.76 h** | 33.1 |
| | | | **67.4 GPU-h** |

Three things the profiling settled, all measured rather than argued:

* **The cost law is `≈ 1 ms × groups × rounds`**, independent of batch size
  and of box count. `floyd_warshall` has 4384 boxes against `dijkstra`'s 1501
  and the same ~33 rounds, and it is **three times faster** — because its
  diagram is the complete graph, so every node has one degree and a round
  evaluates **3** groups where `dijkstra`'s sampled graph gives **12**.
* **The GPU saturates, not the CPU.** Concurrency on one H100 tops out at
  about 1.8× single-job throughput, and the curve is the same with 8 cores
  and with 48 — so four slots per device is the knee.
* **`CMap.compile` does not help here.** The library documents it as a
  several-fold speedup and it is one on `sudoku`; on these maps it is a
  **0.78×** regression on both `bfs` and `dag`, and `reduce-overhead` is
  0.68×. The round step is a Python loop over a dozen heterogeneous groups,
  which is not what it fuses well.

The lever that follows from the cost law and has not been taken: under `max`
pooling — the primary campaign's aggregator — padding every node's message
orbit to a common degree is *numerically exact*, and would collapse 13 groups
to 3. It is a library change to `_routing`'s grouping, exact only for `max`,
and on the measured law worth about 4× on every future campaign.

`dag_shortest_paths` is 49 % of the campaign, so it runs **last**: the other
four rows are complete and evaluable on their own before it starts.

## H4: what the amendment has to be

### What was asked, and why it was right

H4 as written correlates the `check_equivariant` residual of the trained
cells against the per-task ID→OOD drop. The amendment: make the dependent
variable the **order-free-head** drop only, or partial out order-dependent
mass — because M2 shows the drop is dominated by order-dependent head mass, a
label-semantics phenomenon in the decoder, so the raw correlation would be
real, significant and meaningless.

That reasoning is correct. Applying it turned up two facts that make the
amendment necessary but not sufficient.

### The independent variable is identically zero

Measured on every trained model of Part 2 — eight tasks, three seeds, every
cell, float64:

| pooling | `node` | `readout` | `edge` |
|---|---|---|---|
| `max` (the primary campaign) | **0.0** | **0.0** | **0.0** |
| `mean` (the ablation) | 8.9e-16 | 1.8e-15 | 0.0 |

Exactly zero, not nearly. `POOL["max"]` is order-invariant in floating point,
so at the primary campaign's aggregator the equivariance law is **strict**,
not lax, and there is no residual to vary. Under `mean` it is machine epsilon
over the width of a reduction — a fact about orbit sizes, not about learned
weights, so correlating it against the drop would be correlating *how many
neighbours a task's graph has* with generalization, dressed as a symmetry law.

**A correlate with no variance is not a weak correlate.** H4 as written is
closed by measurement, and its verdict is: in this formalism the symmetry is
exact by construction, so it cannot covary with anything. The brief counts a
hypothesis cleanly refuted under a controlled protocol as a result, and this
is one — it is also, honestly, a hypothesis that could have been closed by
reading `laws.py` in Part 1. `evaluate.equivariance`'s docstring said the
residual was "the quantity H4 will correlate against"; it now says this.

### The amended dependent variable is empty at the output level

Every one of the eight algorithms has **exactly one output probe**, and on
all eight it is a `pointer` or a `mask_one`:

| | `minimum` | `bfs` | `bellman_ford` | `dijkstra` | `mst_prim` | `dag_sp` | `floyd_warshall` | `matrix_chain` |
|---|---|---|---|---|---|---|---|---|
| output | `min` | `pi` | `pi` | `pi` | `pi` | `pi` | `Pi` | `s` |
| type | mask_one | pointer | pointer | pointer | pointer | pointer | pointer | pointer |

So order-dependent **output** mass is 100 % on every row
(`evaluate.head_mass`). The benchmark's own micro-F1 — and therefore both
published anchors — is entirely inside the failing head class. Two
consequences for the amendment as stated: an *order-free output drop* is an
empty column on all eight rows, and *partialling out order-dependent mass*
partials out a constant.

The repair is the same for both: **score the split over the hints as well as
the output.** `evaluate.head_split` does, and the order-free drop then has
real spread — −0.039 to 0.469, sd 0.191, over the seven tasks that have an
order-free probe at all. `minimum` has none, so **n = 7, not 8**, and every
H4 table says so.

One more exclusion, forced by a scale rather than by semantics: `scalar`
probes are scored by a mean squared error, which is unbounded and
lower-is-better, so they are pooled with nothing (`config.UNPOOLED`).
Averaging them in is what made `floyd_warshall`'s order-free drop come out at
**−0.046**, which is not an improvement; it is two scales in one mean.

### The replacement independent variable, and the one I already killed

With the residual dead, H4 needs an independent variable that varies. The
obvious one is the promotion of M2 from an observation to a law:
order-dependent mass predicts the drop.

**Operationalized as a share of probe types, it is dead on arrival.**
Measured, `evaluate.h4_table`:

* `mass_vs_drop`: r = **−0.65**, exact permutation p = 0.046 over n = 8.

Significant, and with the **wrong sign**: more order-dependent mass goes with
a *smaller* drop. The reason is that a share of probe types is a ratio of
counts, and the tasks with few probes are the easy ones — `minimum` is 100 %
order-dependent mass and drops 0.195, `dijkstra` is 50 % and drops 0.908. It
is an inverse proxy for task difficulty wearing a mechanism's name: exactly
the failure the amendment was written to prevent, one level down. It stays in
the output labelled as a rejected candidate so that nobody re-derives it.

**What is left is the tie rate**, which is a property of the labels and not
of how many probes a spec declares: the share of a task's order-dependent
targets that the reference algorithm decided by index order, with the induced
order-blind ceiling. It is already measured for `bfs` — 31.7 % of assignments
tied at `n = 16` and 69.7 % at `n = 64`, ceilings 0.8444 and 0.5012 — and
generalizing it needs one function per algorithm, because the tie-breaking
rule is per algorithm and `dataset.reference` already transcribes them.

**H4 does not run until that exists.** Its spec, pre-registered:

* **IV** — tie rate at `n = 64`, per task, from the cached splits;
* **DV** — the ID→OOD drop, in two columns: order-free and order-dependent,
  both over hints ∪ output, `scalar` excluded;
* **the control that decides it** — the IV must predict the
  order-**dependent** drop and *not* the order-free one. If it predicts both,
  it is a difficulty proxy and the mechanism reading is wrong, exactly as the
  type-mass candidate was;
* **the test** — Pearson `r` with an exact permutation `p` over all
  relabellings (`evaluate.correlate`). With 7 tasks the floor is 2/7! =
  0.0004, so unlike H1's three-versus-three seed comparison this design *can*
  resolve something; the small-sample caveat is about seven points being
  seven tasks, and it is stated in the table rather than in a footnote;
* **the honest prior** — `bellman_ford`'s order-free drop is −0.039 and
  `floyd_warshall`'s is 0.029, i.e. two of the seven have essentially no
  order-free drop to explain. A DV that is at ceiling on a third of its
  points is a DV to report cautiously.

## Still open, and not blocking

* **There was nothing to salvage.** The plan was to read `config.REGIME` off
  the killed T2/T3 campaign, within-protocol. That campaign left **one
  checkpoint** — `minimum` — and seven empty training logs, so the regime is
  measured rather than salvaged: `regime.py` trains each row twice, fixed
  against mixed, one seed, under `PART3`'s frozen decoder, and freezes the
  answer into `artifacts/regime.json`.

  The rule is **pre-registered in `config.REGIME` before any probe number
  existed**, and it is deliberately conservative because the probe is one
  seed an arm: *fixed unless mixed beats it on the wide out-of-distribution
  score by more than the row's own Part 2 seed s.e.m., and never when it
  costs the order-free heads.* Part 2's protocol is the incumbent and it is
  the one that did not destroy `minimum`.

  The decision lives in a **file**, not a literal: a regime nobody can edit
  after seeing a campaign's numbers is the point, and `regime.py --write`
  never rewrites a row already recorded. `train.regime_of` still refuses an
  undeclared row rather than defaulting it — a default here is a protocol
  chosen by whoever ran the script first, which is the same failure class as
  a stale checkpoint loaded by tag.
* **The closed-loop `bfs` arm** — one seed, `bfs` only, hints fed back. Our
  brief chose open-loop citing the post-Hint-ReLIC literature; the floor is
  the original 2022 recipe, which trains with hints fed back. It is the last
  structural difference between this pipeline and the floor's, it runs in the
  background, and Part 3 does not wait on it. If it closes most of the
  remaining seven points then the gap was a declared protocol difference and
  the README says so with a number; if not, the remainder stays open.
* **The tie-rate functions**, above. H4 is blocked on them and nothing else
  is.

## The rule that keeps paying

When an intervention changes two things, the two-change result is a
measurement of neither. It has now outperformed both the author's and the
implementer's intuitions four times — the edge channel, the `pos` arms, the
"one cause" reading and the `minimum` budget — and twice more in this file,
on the `FixedPoint × settle` cell and on the type-mass IV. It belongs in the
first sentence of the methods section.
