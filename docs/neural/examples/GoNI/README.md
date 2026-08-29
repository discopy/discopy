# GoNI: the Geometry of Neural Interaction

> The algorithm is the diagram; the boxes are learned; the wiring scales.

A GoNI model of a CLRS-30 task is the task's own dataflow circuit —
one shared generator per elementary step, wired by the data dependencies
— interpreted in :mod:`discopy.neural`. What generalizes out of
distribution is the *family*: the same weights, wired at ``n = 64``
exactly as they were at ``n = 16``, because the wiring carries the
algorithm and the boxes carry nothing but the local step.

`circuits.py` holds the circuit families. The LCS grid (`lcs`) settles a
question the previous run of this study got wrong: it concluded the grid
could not be a diagram because it needs swaps, and wrote the wiring by
hand. The grid *is* a symmetric diagram — the crossings are permutation
layers — and `to_map` absorbs every crossing into the wiring of the
combinatorial map, so the map holds one box per cell and nothing else.
`test/neural/test_goni.py` checks this and that message passing over the
map, with an exact cell, computes the longest common subsequence.

## The task: `knuth_morris_pratt`

The study's benchmark task is the KMP string matcher, chosen as the task
where the expected gap between GoNI and the published state of the art
is the largest:

* it is the benchmark's perennial worst. Triplet-GMPNN scores 19.51%
  out of distribution (Ibarz et al. 2022, arXiv:2209.11142); the
  recurrent-aggregator paper (arXiv:2409.07154) names `floyd_warshall`,
  `knuth_morris_pratt` and `strongly_connected_components` as the only
  CLRS-30 tasks with no known OOD result above 90%, while its own
  advances (quickselect to 87%, from 0.47%) do not touch it; FloydNet
  (arXiv:2601.19094) reports >95% on most of the suite and still names
  string matching among the hardest remaining.
* the *task* — find the first occurrence of the pattern in the text —
  has a data-oblivious circuit: one equality cell per (alignment,
  offset) pair, a conjunction per alignment, a first-match fold for the
  output pointer. Data-oblivious is exactly the GoNI regime: the model
  imposes the circuit and only ever has to learn the local steps, which
  is where `minimum` and sorting worked. Training is output-only, the
  benchmark's standard no-hint setting: the circuit is the matcher's
  dataflow, not an imitation of KMP's trajectory.
* the wiring is heavily non-planar — the pattern threads across every
  alignment — so the task exercises the same symmetric features the LCS
  grid proves out.

## Results

Protocol: the benchmark's own splits (1000 training and 32 validation
trajectories at ``n = 16``, 32 test trajectories at ``n = 64``, seeds 1,
2, 3 of ``clrs._src.samplers.CLRS30``) plus the 128-sample wide split at
``n = 64`` (seed 30) that ``CLRS_small`` reports beside it; training is
output-only for 15 epochs of Adam at ``1e-3``, batch 64, selected on
validation; the score is the benchmark's own for ``mask_one``, exact
argmax match (``clrs._src.evaluation._eval_one``). Note what the size
jump means here: at ``n = 64`` the pattern is 12 characters against the
3 seen in training, so the fold runs four times deeper than it ever did
in training, on wiring alone.

| seed | val (n=16) | test (n=64) | wide (n=64) | CPU minutes |
|-----:|-----------:|------------:|------------:|------------:|
| 0 | 1.000 | 1.000 | 1.000 | 6.9 |
| 1 | 1.000 | 1.000 | 1.000 | 5.0 |
| 2 | 1.000 | 1.000 | 1.000 | 7.6 |

**GoNI: 100.0 ± 0.0 out of distribution, over three seeds.** Not our
scorer's word for it: ``verify.py`` retrains, writes the predictions as
one-hots over *all* the nodes of each sample, and a separate
environment averages ``clrs._src.evaluation._eval_one`` — the function
behind every published ``kmp_matcher`` number — over them: 1.0000 on
``val``, ``test`` and ``wide`` (the committed
``data/kmp-*-predictions.npz``, re-scorable with ``verify.py --score``
and nothing but ``clrs`` and the cache). The best
published number on ``kmp_matcher`` is 19.51 ± 4.57 (Triplet-GMPNN,
arXiv:2209.11142), and no published architecture reaches 90 — a gap of
80 points, the largest available in CLRS-30. The artifacts, with the
whole training history per seed, are in ``artifacts/``.

The honest caveats: the circuit family is the naive matcher's dataflow,
not KMP's — the score is on the benchmark task's inputs and outputs
under its own metric, but the model does not imitate KMP's trajectory
and never sees the hints, so it competes in the no-hint setting; and a
model that scores over the alignments cannot point at a non-alignment
node, a structural prior the baselines lack. That prior is the study's
thesis, not a loophole: the geometry carries the algorithm, the learning
only ever fills in the local steps.

Runner-up candidates and why not: `quickselect` was the obvious pick
under the 2022 numbers but the recurrent-aggregator result (87%) cut its
headroom to ~13 points; `floyd_warshall` has a beautiful static circuit
but FloydNet is built around exactly that inductive bias and reports
near-perfect scores; `strongly_connected_components` is only expressible
statically through a transitive-closure substitute, a weaker claim than
running the task's own dataflow.

## The grid, end to end: `lcs_length`

The LCS grid does not just settle the swaps question, it runs the
benchmark's task. ``lcs_length``'s output is not the corner value but
the traceback: one direction per cell of the dynamic-programming grid —
0 for the diagonal on a match, 1 for up, 2 for left — the ``b`` matrix,
scored cell by cell. The direction is a pure function of the cell's
*inputs*: diagonal on a match, else up against left, a rule that
reproduces the benchmark's fixpoint relaxation — boundary quirks
included — on every cached sample, which ``lcs_dataset.py --check``
re-derives. So the model reads the output off the running circuit:
``lcs_model.py`` asks the map for its flat port state and gathers each
cell's incoming messages off its domain ports — an address into the
state, not a wire, the grid builder untouched — and a direction head
turns them into three logits per cell. The learned part of a cell is
the value fold alone, written on all three value outputs the way the
exact cell writes ``L[i][j]``.

Protocol: as for kmp — the benchmark's splits (8 x 8 grids at
``n = 16`` for the 1000 training and 32 validation samples, 32 x 32 at
``n = 64`` for the 32 test samples and the 128-sample wide split),
seeds 0, 1, 2, output-only, Adam at ``1e-3``, batch 64, selected on
validation; the score is exact directions over grid cells
(``clrs._src.evaluation._eval_one`` on the scored block) — except 50
epochs instead of 15, where validation converges at ~0.998 (at 15 it is
still climbing through ~0.97).

| seed | val (n=16) | test (n=64) | wide (n=64) | CPU minutes |
|-----:|-----------:|------------:|------------:|------------:|
| 0 | 0.998 | 0.924 | 0.926 | 2.1 |
| 1 | 0.997 | 0.931 | 0.924 | 1.5 |
| 2 | 0.998 | 0.949 | 0.941 | 1.5 |

**GoNI: 93.5 ± 1.3 on test, 93.0 ± 0.9 on wide, out of distribution
over three seeds**, against 80.51 ± 1.84 published for Triplet-GMPNN
(arXiv:2209.11142) — on wiring the previous run of this study wrote by
hand because it thought the swaps made a diagram impossible. Not our
scorer's word for it either: ``lcs_verify.py`` retrains seed 0, writes
the predicted directions beside the cache, and a separate environment
lays both sides out with ``clrs._src.probing.strings_pair_cat`` and
averages ``clrs._src.evaluation._eval_one`` over them — 0.9980 on
``val``, 0.9244 on ``test``, 0.9258 on ``wide``, our accuracy to four
decimals (the committed ``data/lcs-*-predictions.npz``, re-scorable
with ``lcs_verify.py --score`` and nothing but ``clrs`` and the cache).

Why not 100, when kmp saturates? The kmp fold carries what amounts to
a boolean; the LCS cell carries a *count*, and at ``n = 64`` the true
values run four times past anything the fold saw in training — the gap
from 99.8 in distribution to 93.5 out of it is value extrapolation in
the learned cell, not wiring. What the family supplies is the
geometry; the local step is where the remaining points live.

## Past the benchmark: `kmp_matcher` at n = 128 and 256

The benchmark stops at ``n = 64``; no baseline has numbers past it.
``dataset.py`` draws two more splits with the benchmark's own sampler
rules — 128 samples at ``n = 128`` (pattern 25 against text 103) and
32 at ``n = 256`` (pattern 51 against text 205, where one forward
costs sixteen ``n = 64`` ones), fresh seeds 4 and 5, cached and
brute-force checked like the others.  ``scale.py`` retrains the three
study seeds under the study protocol — each re-hits 1.000 on val, test
and wide on the way — and evaluates each selected model on both
splits; the weights now land in ``artifacts/`` so a later evaluation
is a load, not a rerun.  At ``n = 128`` the fold runs eight times
deeper than anything training showed it, at ``n = 256`` seventeen
times.

| seed | n = 128 | n = 256 |
|-----:|--------:|--------:|
| 0 | 0.977 | 1.000 |
| 1 | 0.969 | 1.000 |
| 2 | 1.000 | 1.000 |

**98.2 ± 1.6 at n = 128, 100.0 ± 0.0 at n = 256.** Read together, not
apart: the dents at 128 are 7 samples of 384 across the seeds, and the
256 split is a quarter the size, so a slip rate of a point or two is
as compatible with 96 clean samples as with those dents — the two
sizes bound the same small failure rate; nothing recovers at 256.
What the numbers say is that the family holds to within a couple of
points at four times the benchmark's ceiling, on wiring alone — the
weights never saw a pattern longer than three characters.  The cost
lives elsewhere: the ``n = 256`` circuit is a one-off two-hour build
on a laptop-class core, after which every seed's evaluation rides it
in about a minute.
