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
