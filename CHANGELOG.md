# Changelog

All notable changes to DisCoPy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since [`1.2.2`](https://github.com/discopy/discopy/releases/tag/1.2.2).

### Added

- Combinatorial map representation, `discopy.cmap`, encoding diagrams in
  compact categories as a permutation on the ports of each box
  ([#338](https://github.com/discopy/discopy/pull/338)).
- Syntax and drawing for 2-categories
  ([#354](https://github.com/discopy/discopy/pull/354),
  [#355](https://github.com/discopy/discopy/pull/355)).
- `Transformation` in `discopy.cat`, the natural transformations between
  functors ([#351](https://github.com/discopy/discopy/pull/351)).
- `cat.Equation` with an argument `up_to` for computing quotients
  ([#415](https://github.com/discopy/discopy/pull/415)).
- Ribbon diagram support with configurable wire spacing
  ([#358](https://github.com/discopy/discopy/pull/358)).
- Opt-in colour legend for drawings
  ([#357](https://github.com/discopy/discopy/pull/357)).
- Rich display hooks (`_repr_svg_`/`_repr_html_`) for `Diagram` and `Drawing`
  in Jupyter/IPython
  ([#445](https://github.com/discopy/discopy/pull/445)).
- Composition benchmark suite for diagram operations, reproducing the
  scaling experiments of arXiv:2105.09257
  ([#346](https://github.com/discopy/discopy/pull/346)).
- CMap cases for the composition benchmark suite, mirroring its Hypergraph
  workloads. Benchmark reports now include per-suite HTML, Markdown and CSV
  tables with scaling plots.
- Conversion benchmarks between Diagram, Hypergraph and CMap representations.
- The benchmark job runs only on `main` and on pull requests labelled
  `benchmark` ([#385](https://github.com/discopy/discopy/pull/385),
  [#459](https://github.com/discopy/discopy/pull/459)).
- Diagram spacing is now automatically computed from exact font-dependent
  text width, for both box names and wire labels, instead of overflowing
  or colliding with neighbouring wires
  ([#364](https://github.com/discopy/discopy/pull/364),
  [#365](https://github.com/discopy/discopy/pull/365)).
- Explicit permutations in symmetric layers: `symmetric.P` supports the
  permutation operations and functorial semantics, while `symmetric.Layer`
  alternates permutations with generators without canonicalising diagram
  state ([#362](https://github.com/discopy/discopy/pull/362)).
- The category of parametric maps, `discopy.para`, wrapping morphisms
  `dom @ param -> cod` of any symmetric underlying category, with
  reparametrisation as a method and a subclass lifting each level of the
  hierarchy below symmetric: traced, Markov, closed, feedback, compact and
  hypergraph ([#558](https://github.com/discopy/discopy/issues/558),
  refactoring [#325](https://github.com/discopy/discopy/pull/325)).
- `discopy.neural.MapNN`, the central abstraction of `discopy.neural`: an
  ordinary `torch.nn.Module` holding the width of every atomic role, one
  shared learnable module per generator name and a solver. It compiles a
  DisCoPy diagram — or a `Batch` of diagrams whose shapes differ — into a
  global interaction and hands it to the solver, so that a dataset of
  `(diagram, inputs, target)` samples trains with an ordinary PyTorch loop.
- `discopy.neural.solver`, the execution strategies as one hierarchy:
  `Iterate` (finite iteration), `FixedPoint` (Picard iteration towards
  `s = T(s)`, with either the Jacobian-free one-step gradient or unrolled
  backpropagation), `Recursion` (segmented execution with a `Refresh` of a
  trace between cycles) and `ACT` (a `HaltHead` on top). `FixedPoint` and
  `Interaction.residual` are new; the others reproduce the previous
  schedules bitwise.

### Changed

- `monoidal.Layer` holds a list of boxes and non-empty types with at least
  one box and no two consecutive types, instead of an odd-length list
  alternating type and box. Whiskering extends the list only when the type
  is non-empty and the outermost element is a box, otherwise it merges into
  the boundary type, and tensoring two layers merges a trailing type with a
  leading one. The constructor type checks and normalises to restore the
  invariant unless it is called with `normalise=False`, which the internal
  call sites do, so tensoring `n` layers is linear rather than quadratic.
  `Layer` is a `ColouredMonoid`, i.e. it defines `tensor` and inherits `@`
  and its right-whiskering mirror from it, embedding types and boxes as
  layers, and `Layer.cast` is removed since `Layer(box)` already builds the
  singleton layer. `symmetric.Layer` follows with "permutation" in place of
  "type". `Diagram.interchange` checks its preconditions up front, so an
  out-of-range index raises `IndexError` and a diagram with more than one box
  in a layer raises `NotImplementedError` even when `i == j`
  ([#438](https://github.com/discopy/discopy/pull/438)).
- `Arrow` is refactored onto a `FreeCategory` base class
  ([#350](https://github.com/discopy/discopy/pull/350)).
- The `tensor` module is refactored to go through `CMap` for `einsum`
  ([#402](https://github.com/discopy/discopy/pull/402)).
- Add a `functor_factory` attribute to each `Diagram` class and remove
  `hypergraph_factory`: `Hypergraph` is now a `NamedGeneric["category"]`
  instead of a `NamedGeneric["functor"]`
  ([#379](https://github.com/discopy/discopy/pull/379)).
- Documentation notebooks are migrated from Jupyter (`.ipynb`) to marimo
  markdown, with docs (`nbsphinx` → embedded marimo HTML) and CI
  (`nbmake` → `marimo export`) updated to match
  ([#404](https://github.com/discopy/discopy/pull/404)).
- The `Functor` keyword arguments `ob`/`ar` are renamed to
  `ob_map`/`ar_map` throughout the codebase, docs and benchmarks
  ([#369](https://github.com/discopy/discopy/pull/369),
  [#411](https://github.com/discopy/discopy/pull/411),
  [#417](https://github.com/discopy/discopy/pull/417)).
- `Ty.name` is a cached property computed from its `inside`
  ([#421](https://github.com/discopy/discopy/pull/421)).
- SVG drawings are made deterministic by ordering spiders and boxes
  reproducibly
  ([#457](https://github.com/discopy/discopy/pull/457),
  [#469](https://github.com/discopy/discopy/pull/469)).
- Documentation images are converted from PNG to SVG and checked in as
  drawing-test baselines: there are no separate test images anymore,
  every image in the docs doubles as a drawing test
  ([#419](https://github.com/discopy/discopy/pull/419),
  [#435](https://github.com/discopy/discopy/pull/435),
  [#463](https://github.com/discopy/discopy/pull/463),
  [#470](https://github.com/discopy/discopy/pull/470)).
- The `test/` directory is reorganised to mirror `discopy/`
  ([#403](https://github.com/discopy/discopy/pull/403)).
- Symmetric categories generate their swaps with `swap_factory` rather than
  `braid_factory`, which is now a `classproperty` reading it
  ([#440](https://github.com/discopy/discopy/pull/440)).
- `abc.SymmetricCategory` extends `abc.BraidedCategory` directly, so
  symmetric and Markov categories are not required to implement `twist` and
  `trace`; balanced categories stay traced, and the two branches meet again
  in `abc.CompactCategory` where the twist is the identity. The free diagram
  classes keep their freely interpreted traces by subclassing
  `traced.Diagram` ([#349](https://github.com/discopy/discopy/issues/349)).
- `biclosed` defaults `left` to `True` in `Diagram.curry`, `Diagram.ev`,
  `Diagram.uncurry`, `CMap.curry` and `CMap.uncurry`, so that `abc`,
  `biclosed`, `closed` and `rigid` all agree on one convention: the default
  exponential is `Over`, i.e. `<<`. Previously `closed` inherited
  `curry` defaulting to the right from `biclosed` while overriding `ev` to
  the left, so the default currying was never evaluated by the default
  `ev`. Code relying on the old right-handed default should pass
  `left=False` explicitly
  ([#560](https://github.com/discopy/discopy/issues/560)).
- The committed benchmark baseline is stored gzipped as
  `benchmark/baseline.json.gz`, which `benchmark/report.py` reads
  transparently.
- The benchmark regression gate divides each case by the run-wide median
  change rather than comparing raw medians, so that the CPU model a
  GitHub-hosted runner happens to give out does not read as a regression. Its
  default threshold is 25%.
- Benchmark cases now use `pytest-benchmark`'s automatic calibration.
- Every `monoidal.Wire` subclass named `Ob` is renamed to `Wire`: `rigid`,
  `braided`, `biclosed`, `pivotal`, `frobenius`, `feedback` and
  `quantum.circuit`, completing the rename that introduced `monoidal.Wire`;
  `cat.Ob` keeps its name. Accessing the old name still works, returning the
  new class with a `DeprecationWarning` through a module-level `__getattr__`
  (`utils.deprecated_ob`), on those seven modules and on `compact` and
  `grammar.pregroup` which re-exported it; trees serialised with an `Ob`
  factory string load the same way
  ([#566](https://github.com/discopy/discopy/pull/566)).
- `discopy.neural` is reorganised around `MapNN`: `model.py` (`MapNN`),
  `map.py` (the interpretation, the compiled `Interaction`, and the
  specifications `ParamMap` / `InteractionMap`), `solver.py`, `batch.py`,
  `cells.py`, `signature.py`, `laws.py` and the unchanged category in
  `core.py`. `skeleton.py`, `functor.py`, `parametric.py`, `dynamics.py`
  and `engine.py` are gone, and with them `Skeleton`, `Interpretation`,
  `Wiring`, `Router`, `Schedule` and the four engines: a diagram plus a
  `MapNN` is the whole pipeline, and the port families a solver reads are
  now derived from the wiring rather than from a declared signature per
  box. `InteractionMap` no longer offers `>>`, since two interactions
  glued along a shared object compose by wiring and iteration rather than
  by substitution; its tensor is kept. Training loops, adaptive-computation
  -time policy and the evaluation protocols move out of the library into
  `docs/neural/examples/sudoku`, which reproduces every recorded result in
  six files. The arithmetic is unchanged, checked bitwise against
  `docs/neural/golden/` for all four recorded models in float32 and
  float64; pre-refactor checkpoints load through
  `examples/sudoku/model.py`'s `rename` / `load_checkpoint`.

### Added

- `MapNN.cache_stats`, and the `hits`/`misses` counters behind it.
  `MapNN.compile` keeps an LRU of compiled interactions and compiling is
  the expensive half of running a diagram once and free every time after,
  so a cache one diagram too small turns a one-off setup cost into a
  per-epoch one — a difference that shows in the wall clock and nowhere
  else. It is now countable, and `examples/CLRS_small` sizes its cache from
  the batches that exist (`model.fit_cache`) rather than from an arithmetic
  guess.
- `"max"` in `discopy.neural.cells.POOL`, beside `"mean"` and `"sum"`: the
  order-invariant reduction a change of degree leaves alone. A mean and a
  sum both rescale when a site's orbit grows, so a model that learned an
  extremum at one size reads a different number at another; a max does not,
  which is what a size-generalization study needs of its aggregator. It is
  also exactly permutation-equivariant rather than equivariant up to the
  reordering of a floating-point sum.
- `docs/neural/examples/CLRS_small`, Part 1 of a port of the sudoku
  methodology to the CLRS-30 algorithmic-reasoning benchmark: the
  benchmark's own trajectories cached as arrays, an incidence diagram per
  batch of graphs with a node, an undirected edge and a graph-level readout
  generator, encoders and decoders for the benchmark's feature types, and
  the training and evaluation protocol for `minimum`, `bfs` and
  `bellman_ford`. `test/neural/test_clrs_smoke.py` runs the whole thing on
  four trajectories in seconds.
- Part 2 of `docs/neural/examples/CLRS_small`: the five remaining
  algorithms of the project brief (`dijkstra`, `mst_prim`,
  `dag_shortest_paths`, `floyd_warshall`, `matrix_chain_order`) under one
  protocol, and the three things they needed. **Edge-level decoders** —
  scalar, mask and pointer over pairs — with the two dynamic programs
  drawn on the *complete* graph, since they answer about every pair and a
  model with one box per sampled edge has nowhere to keep the answer for
  the others; the wiring then depends on the size alone, so a whole split
  compiles once. A **`model.Link` cell**, thirty lines subclassing
  `cells.Cell`, whose two ports are distinguishable when the sampler is
  directed (`Sym.NONE`, so no equation is owed) and whose recurrent state
  may be erased, which is H1's node-only arm. And the **trajectory rule**:
  a run is `HOPS` rounds per step of the sampled execution rather than a
  constant 16, so that the depth means the same thing on an algorithm
  whose trajectory is three steps and on one whose trajectory is sixty
  -four. Nothing in `discopy.neural` changed; the candidate change that
  came closest — a `Site` that tolerates an erased state — is argued and
  declined in the example's `NOTES.md`.
- The checkpoint-to-step correspondence of `examples/CLRS_small` is one
  named function, `model.alignment`, read by both the solver and the loss
  rather than repeated in either, and pinned constructively: `bfs` on a
  path graph, where the `k`-hop ball is computable by inspection.
- A per-round residual curve in `examples/CLRS_small/evaluate.py`, beside
  the scalar residual: `Interaction.residual` after every round, run past
  the trained depth, on the in-distribution and out-of-distribution splits
  of all three algorithms.
- Per-probe hint curves and a confidence interval in
  `examples/CLRS_small/evaluate.py`: one score per hint probe per step of
  the trajectory — where the imitation comes apart — and the
  128-trajectory out-of-distribution split reported as a mean over
  trajectories with a 95% interval, which is what a table of two rows
  needs before it can call a difference a difference.
- A per-probe loss term in `examples/CLRS_small`, beside the per-stage
  ones, logged every validation epoch and kept in the training history: a
  total hides a head, which is how a decoder shared between three probes
  survived a whole campaign.
- `examples/CLRS_small/evaluate.py`'s `settling`, and the overlay it
  exists for: the round at which the *algorithm* stops moving, read off
  the hints of a split — the last index at which any hint probe changes,
  padding excluded, on the round axis of `residual_curve` — reported per
  probe and drawn in the same panel as the residual curve rather than
  beside it. A falling residual is a fact about a learned map; "the
  learned map settles where the algorithm does" is a distance between two
  things on one axis, which is the sentence Part 3's H2 lives or dies on.
- The two spreads of a row, separately: `summarise` records the standard
  deviation over seeds, its standard error, and the 95% interval over the
  128 trajectories of a run, and the eight-task and H1 tables give the
  last two their own columns with a legend. `config.ANCHORS` names its
  own field `sem` rather than `std`, on Ibarz et al.'s statement that
  their error bars are standard errors across seeds, so the column beside
  them is the same statistic; H1's table carries the delta of its two
  arms with the standard error of a difference of two independent means.
- `examples/CLRS_small`'s `evaluate.tracking` and its figure, which
  separate an **executor from a shortcut**: a hint score read at its best
  out-of-distribution step rather than its last, over the
  `argmax`-over-the-nodes probes, plotted against the *fraction* of the
  trajectory so that the two splits share an axis. Both readings end low
  and only the best tells them apart, and the distinction decides whether
  a `FixedPoint` has a fixed point to find at all. It is per probe, not
  per algorithm: `floyd_warshall`'s `Pi_h` tracks the algorithm and
  drifts, its `k` never tracks it at any step.
- Mixed training sizes and termination supervision in
  `examples/CLRS_small`, the two root-cause fixes for the depth column:
  `config.MIXED` draws training trajectories at `n` in 8..16 at the same
  total budget with `Batches.over` keeping every batch homogeneous in
  `n`, and `Budget.settle` supervises the checkpoints past a sample's own
  trajectory on its final hint repeated rather than dropping them, which
  puts a basin at termination into the loss instead of hoping Part 3
  finds one. Both are the example's; `discopy.neural` is unchanged.
- `evaluate.significance`, printed under H1's table: Welch's `t` with its
  degrees of freedom and an **exact permutation test**. With three seeds
  an arm the permutation floor is `p = 0.10` however cleanly the arms
  separate, so the design's own ceiling is printed beside the claim.
- `examples/CLRS_small/dataset.py --survey`, which reads off `clrs` what
  each of the eight algorithms of the project brief actually draws — the
  sampler class, its `_random_er_graph` flags, and the probes located
  anywhere but on nodes — so that the scope of the parts still to be
  written is a measurement rather than a recollection.
- `docs/neural/examples/CLRS_small/PART3.md` and the protocol it names:
  Part 3's four arms (`config.H2_ARMS`), the head partition every one of
  its tables owes (`evaluate.head_table`, `--heads`), and H4 as amended
  (`evaluate.h4_table`, `--h4`). Written before a Part 3 model is
  trained, because the two things it decides are the two places where a
  design settled halfway through is indistinguishable from a result. Four
  measurements changed it and are on the record in `NOTES.md`: the
  requested `{Iterate, FixedPoint} x {settle, no-settle}` grid has an
  **empty cell** — the terminal checkpoint is the only one a
  `FixedPoint(backward="last")` differentiates and the only one no hint
  index reaches, so `settle` is a no-op there, and with `tol=None` the
  other backward mode is bitwise `Iterate`; `settle` as Part 2
  implemented it never reached that checkpoint either, so
  `config.SETTLE` gains `"terminal"` beside the `"interior"` the
  campaign was trained under; a fixed point in this example cannot train
  an encoder, since the inputs ride in the state that its gradient
  detaches, which `model.Grounded` repairs at bitwise-identical forward
  values; and an output-only arm fits its hint heads on a detached state
  (`Budget.probe`) rather than dropping the term, or the mandatory
  per-head split would be read off untrained heads. `discopy.neural` is
  unchanged throughout.
- `test/neural/test_sudoku_act_e2e.py`, an end-to-end check that trains model
  C through the real ACT loop on the Palm et al. (2018) benchmark and then
  reads it with the noise study's own `latent_stats` and `run_segment`: the
  first test in which the model is good enough for "noise costs board
  accuracy" to be falsifiable. It is marked `neural_e2e` and deselected by
  default — three minutes, a GPU and a cached benchmark — so run it with
  `pytest -m neural_e2e` when `discopy.neural` changes.
- `docs/neural/examples/sudoku/optuna_act.py`, the search behind the `act`
  recipe, carried over to the refactored API: model C with a detached
  soft-minimum halt head on sudoku-extreme, ranked on the best validation
  board rate across evaluations, with the budget and the evaluation cadence
  counted in *puzzles consumed* rather than optimizer steps — the only unit
  under which trials of different halting depths see the same data. It adds
  two things the pre-refactor script did not have: `--seed-from`, which
  copies the completed trials of an earlier study with their whole
  intermediate curves, so `MedianPruner` has a median to compare against
  from the first trial rather than after five fresh ones; and
  `--workers-per-gpu`, since oversubscribing a launch-bound model fills the
  gaps one worker leaves between its kernels.
  `--schedule-epochs` separates the learning-rate schedule from the budget,
  which is what lets a trial be *truncated* rather than *rescheduled*:
  stopping at six epochs on the ten-epoch schedule leaves the learning rate
  at every step exactly what a full-length trial had, so a short trial is a
  prefix of a long one and the two are comparable check for check, where
  compressing the cosine into six epochs would be a different recipe needing
  a study of its own. Six is where the records say the information ends —
  six of the seven completed trials peak by check 27, and truncating all of
  them at check 30 would have cost 0.0004 board on average against a 40%
  shorter run.
  `--seed-max-step` cuts an imported curve at the check the new trials will
  stop at and re-reads its value as the maximum over what is left, so a
  ten-epoch record is imported as the six-epoch trial it contains — which
  also lets a run *interrupted past* that check count as a finished short
  trial instead of being discarded, since under truncation it is one.
  `--storage` also accepts a path ending in `.journal`, which builds an
  optuna `JournalStorage` over a `JournalFileBackend` with a symlink lock:
  sqlite takes a whole-file `fcntl` lock, which a shared filesystem
  implements per client rather than per cluster, so pooling workers from
  more than one allocation onto one study needs the append-only journal
  rather than a `.db`.
- `docs/neural/examples/sudoku/eval_best.py`, the test-split report for a
  search winner: accuracy against test-time compute, then the same under
  answer noise with the halt head selecting among rollouts. The search
  ranks trials on 2,000 puzzles of `valid`, choosing over the evaluations
  of a trial and then over the study, so its number is a doubly-selected
  maximum on a small sample; this is what turns it into a claim about the
  authors' 422,786-puzzle test split, which the search never touches.
  `eval_best.Recorder` writes every accuracy measured to one tidy table —
  a row per `(stage, compute, sigma, protocol, metric)`, as CSV and as
  JSON, rewritten as each row lands. A sweep's own JSON keeps its nesting
  and its `npz` keeps the per-example arrays, but neither is what a plot
  reads; the flat table is, and `Recorder.merge` puts stages that ran as
  separate jobs into a single one. Writing through after every row also
  means a run killed at hour two still reports what it measured.

### Fixed

- Pre-refactor checkpoints of the searches load again. The old model held
  its generators twice — once as attributes, once in a `ModuleList` the
  solver iterated — so its `state_dict` recorded the same tensors under two
  names, and a `MapNN` has nowhere to put the second copy. All eight
  recorded search checkpoints carry eighteen such keys, bitwise equal to
  the ones they alias, and every strict load of them raised on unexpected
  keys; `model.translate` now drops the aliases (`model.ALIASES`). The
  claim that `rename`/`load_checkpoint` reads pre-refactor checkpoints held
  only for the baselines, which have no `ModuleList` to duplicate.
- `pandas` is a `dev` dependency. `docs/neural`'s training harness used to
  import it unconditionally, so without it `pytest.importorskip` silently
  skipped the whole `discopy.neural` equivalence gate, the sudoku smoke test
  and three tests of `test_general.py` — in CI as well as locally. The
  example no longer uses `pandas` at all, so nothing is skipped either way.
- The noise study records how a sweep was produced — the device and the
  torch version — beside what was asked of it, since eager and compiled
  agree only up to the rounding freedom `CMap.compile` documents. It also
  reads the answer and latent widths off the compiled interaction rather
  than off a `widths` attribute no model had, which made `latent_stats`
  raise `AttributeError` on its main path. Both now live in
  `docs/neural/examples/sudoku/evaluate.py` and are covered by
  `test/neural/test_noise_eval.py`.
- `frobenius.Diagram.unfuse`'s doctest no longer sets `Spider.color = "red"`
  to draw its example, which was leaking into every later doctest in the
  same pytest process
  ([#522](https://github.com/discopy/discopy/issues/522)).
- Tensor networks are contracted with `opt_einsum` when the number of
  indices exceeds `numpy.einsum`'s 52-index limit
  ([#448](https://github.com/discopy/discopy/pull/448)).
- `grammar.categorial.cat2ty` reads a fully parenthesized category such as
  `(S\NP)` as a category rather than an atom, strips CCGbank features
  wherever they occur rather than on atoms only, and associates slashes to
  the left as CCG does
  ([#528](https://github.com/discopy/discopy/issues/528)).
- `biclosed.Application` lists its free variables in the same order as the
  wires of its `dom`, so that `Abstraction` strips the right end of it and
  `eval` preserves both `dom` and `cod`
  ([#550](https://github.com/discopy/discopy/issues/550)).
- Hypergraph hash
  ([#387](https://github.com/discopy/discopy/pull/387)).
- Bubble drawing
  ([#431](https://github.com/discopy/discopy/pull/431)).
- Controlled gate drawing: the control wire is anchored on the indexed
  input of the controlled box rather than its first one, so gates with a
  classical wire or a distance other than one are drawn on the right wires
  ([#439](https://github.com/discopy/discopy/pull/439)).
- Drawing a discard on more than one wire: `draw_discard` was shadowing the
  layer index with its inner loop counter
  ([#513](https://github.com/discopy/discopy/issues/513)).
- `closed.Context.dom` called `category.ob.tensor` unbound, which raised
  `TypeError` for an empty context instead of returning `Ty()`
  ([#549](https://github.com/discopy/discopy/issues/549)).
- Both branches of `closed.Abstraction.eval` curry on the right: the
  context branch curried out the wrong end of its domain, so an abstraction
  applied to an argument sharing a free variable did not compose, and a
  left abstraction evaluates through its right counterpart
  ([#562](https://github.com/discopy/discopy/issues/562)).

### Performance

- `neural.CMap.port_widths` is a `cached_property`, like `module_list`
  beside it: it is a function of the boxes, which a map fixes in its
  constructor, and `CMap.forward` reads it on every call. One call is one
  round when a solver asks for a residual, so `Interaction.residual`
  rebuilt every `Port` of the diagram per round — 69% of a residual
  call on a diagram with a box per pair, and the whole cost of a
  per-round residual curve over one. Measured on `examples/CLRS_small`'s
  `floyd_warshall` at `n = 16`: 36.9 ms to 3.6 ms, and the factor grows
  with the number of boxes. The values are unchanged and
  `test/neural/test_equivalence.py`'s golden gate says so.
- `Ty` construction is sped up with `assert_isinstance` and lazy naming
  ([#420](https://github.com/discopy/discopy/pull/420)).
- `Hypergraph` equality, permutations and other micro-optimizations bring
  equality checks down to `O(n)`
  ([#353](https://github.com/discopy/discopy/pull/353)).

### Project

- `AGENTS.md`/`CLAUDE.md`/`RULES.md`/`STYLE.md` introduced and iterated on,
  and `CONTRIBUTING.md`/`README.md` updated to match, to describe the
  collaboration and coding protocol for AI agents working on the repo
  ([#378](https://github.com/discopy/discopy/pull/378),
  [#422](https://github.com/discopy/discopy/pull/422),
  [#428](https://github.com/discopy/discopy/pull/428),
  [#471](https://github.com/discopy/discopy/pull/471),
  [#477](https://github.com/discopy/discopy/pull/477),
  [#481](https://github.com/discopy/discopy/pull/481)).

## [1.2.2] - 2025-12-19

See the [GitHub release](https://github.com/discopy/discopy/releases/tag/1.2.2).

## Older releases

See the [GitHub releases page](https://github.com/discopy/discopy/releases)
for the changelog of `1.2.1` and earlier.
