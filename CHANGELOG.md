# Changelog

All notable changes to DisCoPy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since [`1.2.2`](https://github.com/discopy/discopy/releases/tag/1.2.2).

### Added

- A `workflows` job in `build.yml`, so that the code running our pull
  requests is checked like the code it checks: `actionlint` over the
  workflows, `pflake8` over `.github`, and `pytest .github/tests/*.py`
  over the three scripts and the composite action, whose steps take a
  strict subset of a workflow step's keys that `actionlint` does not
  check. Three of the last five changes to `.github`
  were fixing bugs in `.github`
  ([#611](https://github.com/discopy/discopy/issues/611),
  [#615](https://github.com/discopy/discopy/issues/615),
  [#640](https://github.com/discopy/discopy/issues/640)), every one found
  in production. On its first runs `actionlint` found the `style-review.yml`
  bug below, and shellcheck the `A && B || C` in `benchmark.yml`'s summary
  step, now an `if` ([#645](https://github.com/discopy/discopy/pull/645)).
- `.github/actions/setup`, one composite action for installing uv, Python,
  the project and, for the jobs that draw, Graphviz. The three `build.yml`
  jobs called for it four times between them and the Graphviz incantation
  was byte-identical twice. `benchmark.yml` keeps its own steps: it checks
  out two arbitrary commits and one of them predates this action
  ([#645](https://github.com/discopy/discopy/pull/645)).
- `.github/dependabot.yml`, grouping the monthly GitHub Actions updates
  into one pull request, now that every action is pinned by commit
  ([#645](https://github.com/discopy/discopy/pull/645)).
- The style review can be asked for, and turned off, from the pull request
  itself: `@discopy review this` in a comment reviews it now, and the
  `no-style-review` label stops the automatic reviews on it, while the
  comment goes on working — it is "stop reviewing this on its own", not
  "never review this". The comment is read from people with write access
  only, and labelling already is, so nobody who can merely comment can
  silence the reviewer or spend the gateway budget. It replaces the
  `style-review` label, which did the same on demand except that it never
  handed over to the correctness reviewer. A pull request already open and
  not about to change had no trigger at all otherwise, since only a push
  reaches one ([#638](https://github.com/discopy/discopy/issues/638)).
- `Diagram.to_compact` and `CMap.to_compact`, bending curry bubbles into
  coevaluation and feedback. Since a biclosed category has no trace, the
  `biclosed` method lands in `CMap`, which is compact whatever hosts it,
  while the `closed` one stays in diagrams. Unlike `rigid.to_rigid` and
  `interaction.Int`, this keeps the exponential atomic and bends the wire
  with `biclosed.Coeval`, the transpose of `Eval`, which a biclosed
  category only has when its exponential is read at a reflexive object
  ([#532](https://github.com/discopy/discopy/pull/532)).
- A style review workflow: on a revision of a same-repo pull request, one
  model request reads every changed Python file whole — with the
  package-local files they import as context — checks the diff against the
  file's own conventions and `STYLE.md`, and
  discopy-bot posts the findings as one review — style only, correctness
  stays with the correctness reviewer, whom discopy-bot calls once the
  style review has nothing to say. Inference runs on an open-weights
  model behind an OpenAI-compatible gateway, configured by the
  `STYLE_REVIEW_BASE_URL` and `STYLE_REVIEW_MODEL` repository variables and
  the `STYLE_REVIEW_API_KEY` secret
  ([#608](https://github.com/discopy/discopy/pull/608)).
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
  workloads. Benchmark reports now include a per-suite Markdown table with
  a scaling plot.
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
- `para.Symmetric` carries an optional coparameter space: a map is
  `inside : dom @ param -> cod @ copar` with `copar` empty by default, so
  parametric maps read as before, coparametric maps are the empty-`param`
  case and the diagonal `param == copar` is the free category with feedback
  — the type of one time step of a `Stream`. The constructor reads
  `(dom, cod, inside, param, copar)` with both hidden spaces optional.
  Composition and tensor accumulate the hidden objects on both sides,
  `trace` and `feedback` route the coparameters out of the way and
  `recopar` post-composes them, covariantly where `reparam` is
  contravariant ([#572](https://github.com/discopy/discopy/issues/572)).
- The pivotal structure of `Rep(H)`: `HopfAlgebra.drinfeld_element`,
  `pivotal_element` and `ribbon_element`, cached single tensors named after
  the literature (Reshetikhin–Turaev; Kassel; Radford), with pivotal cups
  and caps twisting the dual leg so all four orientations are intertwiners.
  `taft(n)`, the smallest algebras with a pivot of order `n` (Sweedler's
  algebra is `n = 2`), realise the Kauffman–Radford ribbon criterion
  ([#484](https://github.com/discopy/discopy/pull/484)).
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

- The benchmark measures a pull request against its merge base rather
  than the tip of its base branch. The head does not contain what landed
  on `main` since it forked, so measuring against the tip charged the pull
  request for everyone else's commits. `benchmark.yml` resolves it with one
  `compare` call and records it as `previous` in the artifact metadata,
  next to the `base` the comment still validates itself against
  ([#645](https://github.com/discopy/discopy/pull/645)).
- `benchmark-comment.yml` is 33 lines of YAML calling
  `.github/scripts/benchmark_comment.py` rather than 140 lines of
  JavaScript embedded in YAML. Nothing needed `actions/github-script`: the
  event payload is a JSON file named by `GITHUB_EVENT_PATH` and the REST
  API is `urllib`, which `.github/style-review/post.py` already talks to.
  In Python it is lintable, testable and in the one language this
  repository is written in; its validation is `unreadable`, `unattested`
  and `mismatch`, three pure functions the tests state the refusals of.
  The job also stopped taking the artifact's word for three things, since
  the pull request can write it: the pull request number is checked to be
  an integer before it reaches a URL rather than after, the merge base the
  comment links is checked against one the job computes itself from two
  commits it already trusts, and a run that lists no pull request of its
  own -- one from a fork -- must name the single open pull request for its
  head rather than any that shares its branch. A download that fails is no
  longer silence: the job asks whether the artifact was staged at all, and
  only then posts nothing
  ([#645](https://github.com/discopy/discopy/pull/645)).
- `build.yml` and `benchmark.yml` cancel a pull request's superseded runs
  but let every commit on `main` finish, `cancel-in-progress` reading
  `github.event_name == 'pull_request'`. Cancelling on `main` left commits
  nothing ever built — `112b6036` is one — and threw away the pair of
  measurements a benchmark run exists to produce
  ([#645](https://github.com/discopy/discopy/pull/645)).
- Every action is pinned by commit, not by moving tag, as
  `benchmark-comment.yml` already pinned two of them; `build.yml` declares
  `permissions: contents: read` like the other four workflows; and every
  checkout sets `persist-credentials: false`
  ([#645](https://github.com/discopy/discopy/pull/645)).
- `build.yml` drops the `SRC_DIR` and `TEST_DIR` variables, which nothing
  read, and the `tooling/uv-migration` push trigger, whose branch is gone
  ([#645](https://github.com/discopy/discopy/pull/645)).

- `CMap` is aligned on `Hypergraph`. It is parameterised by a category as
  `NamedGeneric["category"]` instead of carrying `require_*` flags, and it is
  always compact whatever category hosts it, so every compact operation is
  available when manipulating maps. The host category is asked for structure
  only on the `to_diagram` downgrade path, i.e. in `make_monogamous`, which
  needs cups and caps, and in `make_causal`, which reorders acyclic maps
  without traces and only asks for traces when cycles or scalar loops remain,
  cutting every backward wire and loop at once. Each box is placed where its
  first domain wire already is, so the decoder no longer swaps that wire to
  the front.
  The predicates follow the `Hypergraph` names and are local conditions on
  the edges, `__init__` takes a keyword `check`, and `curry`, `uncurry` and
  `ev` come from the cups and caps of `abc.RigidCategory` when the host
  category is rigid and stay explicit boxes otherwise, all three defaulting
  `left` to `True` like the rest of the hierarchy. `CMap.eval` delegates to
  the `eval` of the host category, e.g. contracting a tensor map in a
  single `einsum`, instead of `tensor` grafting it onto its `CMap` alias
  ([#532](https://github.com/discopy/discopy/pull/532),
  [#560](https://github.com/discopy/discopy/issues/560)).
- `uncurry` is defined once in `abc.BiclosedCategory`, in terms of a new
  method `base_and_exponent` for the two objects that `ev` evaluates.
  `abc.RigidCategory` and `cmap.CMap` override that method instead of
  duplicating the composition with `ev`: a pregroup has no exponential
  object, so its exponent is the `n` objects at the end resp. the start of
  the codomain, dualised, and a map reads it off its wiring when the host
  category is rigid ([#532](https://github.com/discopy/discopy/pull/532)).
- `balanced` and `pivotal` export a `CMap` alias like the other levels of
  the hierarchy ([#532](https://github.com/discopy/discopy/pull/532)).
- `Hypergraph.to_diagram` raises `messages.NOT_RIGID/FROBENIUS/TRACED/...`
  where it checks that the category has the wiring structure
  ([#532](https://github.com/discopy/discopy/pull/532)).
- `Swap` is now the two-wire transposition subclass of `Permutation`, and
  constructing `Permutation(x @ y, [1, 0])` returns a `Swap`. A swap is
  plumbing like any other permutation: it coalesces with its neighbours in
  a `symmetric.Layer`, so a whiskered swap is stored and drawn as one wider
  permutation, and `foliation` composes consecutive layers of pure plumbing
  into one, unless they compose to the identity. The pictures stay the same:
  a permutation no longer re-labels a wire it keeps in place, nor pushes its
  input labels off the canvas, so the redrawn baselines only differ by their
  serialisation, except `symmetric/foliation.svg` (input labels come back on
  canvas), `int/symmetric-feedback.svg` (one row taller) and
  `symmetric/yang-baxter.svg` (gains its foliated middle)
  ([#444](https://github.com/discopy/discopy/issues/444)).
- The quantum `SWAP` is a gate rather than the symmetry of the category, so
  that a physical swap is distinguishable from a logical one. It is a
  `QuantumGate` drawn as a crossing, while `Circuit.swap` still gives the
  plumbing `quantum.circuit.Swap`: the two evaluate to the same array but
  only the gate survives compilation, `to_tk` emitting `OpType.SWAP` for
  the gate while compiling a logical swap away by applying later gates to
  the permuted qubits.
  `discopy.quantum` exports both, `discopy.quantum.gates` only the gate.

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
  `hypergraph_factory` and `map_factory`: `Hypergraph` and `CMap` are
  parameterised directly as `NamedGeneric["category"]`
  ([#379](https://github.com/discopy/discopy/pull/379),
  [#532](https://github.com/discopy/discopy/pull/532)).
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
- `abc.ColouredMonoid.unit` takes a colour and may return an object of `C0`
  rather than an element of `C1`, since the unit of a coloured monoid is the
  identity on a colour and need not belong to the monoid. `monoidal.Layer`
  overrides it to give the empty type: a layer has at least one box, so
  `Layer()` raises and `Layer.unit()` used to raise with it, while
  `Layer.unit(colour)` is now the empty type that `tensor` accepts on either
  side ([#568](https://github.com/discopy/discopy/issues/568)).
- `monoidal.Layer.id` raises instead of building a layer of empty plumbing,
  which denoted the identity diagram while not being the empty sequence of
  layers: inside a `Diagram` it survived `normal_form`, compared unequal to
  `Diagram.id` and made `foliation` and `draw` raise. `Layer.whisker` leaves a
  type as a type and `tensor` merges it into the boundary, so whiskering never
  builds one. Passing `normalise=False` still does, which is left as an
  explicit opt-out of the invariant
  ([#599](https://github.com/discopy/discopy/issues/599)).
- `biclosed` defaults `left` to `True` in `Diagram.curry`, `Diagram.ev`,
  `Diagram.uncurry`, `CMap.curry` and `CMap.uncurry`, so that `abc`,
  `biclosed`, `closed` and `rigid` all agree on one convention: the default
  exponential is `Over`, i.e. `<<`. Previously `closed` inherited
  `curry` defaulting to the right from `biclosed` while overriding `ev` to
  the left, so the default currying was never evaluated by the default
  `ev`. Code relying on the old right-handed default should pass
  `left=False` explicitly
  ([#560](https://github.com/discopy/discopy/issues/560)).
- Benchmarks compare two commits measured on the same runner rather than a
  committed baseline, so no baseline is stored in the repository and no
  normalisation is needed to account for the CPU model a GitHub-hosted runner
  happens to give out. A pull request compares its head against its base, a
  push to `main` against the branch before the push. The comparison goes to
  the job summary and, on a pull request, to a comment listing the regressions
  and speedups over 25%; a regression raises a warning annotation and never
  fails the job, since a shared runner can push an unrelated case over the
  threshold on noise alone.
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
- `docs/neural/examples/GoNI`, the Geometry of Neural Interaction study:
  a CLRS-30 task's own dataflow circuit as a diagram, with one shared
  learned cell per generator, so that out-of-distribution size
  generalization is a property of the circuit family rather than of the
  training. `circuits.lcs` draws the LCS grid as a symmetric diagram
  whose crossings are permutation layers absorbed by `to_map` — settling
  that the swaps which stopped a previous run of the study cost nothing
  — and `circuits.match` draws the string matcher of `kmp_matcher`, the
  benchmark's hardest task, trained output-only on the benchmark's own
  splits. `test/neural/test_goni.py` checks both circuits compute their
  algorithms exactly and that the matcher trains.
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

- `style-review.yml`'s hand-over to the correctness reviewer, and its
  token generation, ran on every style review rather than the intended
  ones. Both conditions were written as `if: >` folding a wrapped
  `${{ ... }}` into a string with a trailing newline: with characters
  around it the expression is no longer the whole value, so GitHub read a
  non-empty string and took it as true. `@cubic-dev-ai review` was
  therefore posted whatever the style review found, where it is meant to
  wait for a clean one. [#634](https://github.com/discopy/discopy/pull/634)
  rewrote both conditions and the shape survived, so the fix is applied to
  its versions: written bare, as the file's other five conditions are
  ([#645](https://github.com/discopy/discopy/pull/645)).

- The style review no longer depends on a transition that may never
  happen. `ready_for_review` fires on the draft-to-ready edge alone, so a
  pull request whose `TODO.md` was deleted before it was ever opened went
  unreviewed, silently — no run, no notice, nothing in the Actions tab —
  and a pull request the review did find something on was never reviewed
  again, since fixing a nitpick is a plain push, leaving the correctness
  reviewer, called only on a clean review, never called at all.
  `style-review.yml` now triggers on `opened` and `synchronize` as well: a
  pull request that is not draft and carries no `TODO` file is in the
  review phase by construction, since `no-todo-on-main.yml` forces draft
  while a `TODO` is there, so every revision of it is reviewed. Every
  automatic trigger waits while a `TODO` file is in the tree, which also
  keeps the review from racing that guard — on a `main`-based pull request
  the deleting push lands while the guard still holds it draft, so the
  review comes from the `ready_for_review` that follows rather than twice,
  while a pull request based on anything else, which the guard watching
  `main` alone never drafts and never marks ready, is reviewed on the push
  itself. The hand-over to the correctness reviewer happens once per pull
  request rather than on every clean run, since it re-reviews each push on
  its own. A draft is never reviewed, whatever the trigger, and asking for
  one by comment is what ignores the wait
  ([#615](https://github.com/discopy/discopy/issues/615),
  [#636](https://github.com/discopy/discopy/issues/636)).
- Pivotal diagram-to-map conversion now encodes cups and caps as `CMap`
  wiring rather than keeping them as boxes
  ([#532](https://github.com/discopy/discopy/pull/532)).
- `CMap.cups` and `CMap.caps` now require the handedness of the host category,
  i.e. `cups(x, x.r)` and `caps(x.r, x)`, so that these factories reject badly
  oriented cups and caps, rather than fixing the handedness at downgrade time.
  ([#532](https://github.com/discopy/discopy/pull/532)).
- `Hypergraph.explicit_trace` and `CMap.explicit_trace` no longer mistake the
  inherited `trace_factory` of a user-defined subclass for a class method,
  which used to raise `AttributeError: type object 'Trace' has no attribute
  '__func__'` ([#532](https://github.com/discopy/discopy/pull/532)).
- `CMap.topological_order` raises `AxiomError` on a map with a directed
  cycle, where it used to crash with `TypeError` on the `None` returned by
  `box_ranks` ([#532](https://github.com/discopy/discopy/pull/532)).
- `Hypergraph.to_diagram` no longer asks for swaps when one of their two
  sides is empty, where the identity does
  ([#532](https://github.com/discopy/discopy/pull/532)).
- A boxless `monoidal.Layer` can no longer be placed inside a `Diagram`:
  `Diagram.__init__` raises `ValueError` for a layer with no box, restoring
  the invariant that every layer holds at least one box and that the identity
  diagram is the empty sequence of layers. Such a layer is the internal unit
  of `Layer.tensor`, built by `Layer.id` and merged away by `Layer.normalise`;
  put inside a diagram by hand it survived `normal_form` and made `foliation`
  and `draw` raise. The check is gated on `_scan`, so the internal fast paths
  that build layers by construction are unaffected
  ([#599](https://github.com/discopy/discopy/issues/599)).
- `review.py`'s style-review request: `ask` used to let a gateway
  `HTTPError` propagate without reading its body, so a 400 gave no clue
  whether it meant a dead model slug or an oversized prompt; it now prints
  the response body before re-raising. `assemble` used to budget the raw
  file texts against `BUDGET`, but `numbered`'s line-number prefixes, the
  per-file headers, `prompt.md` and `STYLE.md` were all added on top,
  uncounted, so the assembled prompt could exceed `BUDGET` on a PR
  touching a large module even when its diff was small; every part is now
  budgeted as assembled. `ask` also used to unconditionally send
  `"reasoning": {"enabled": False, "exclude": True}`, which not only 400s
  on models that mandate reasoning (e.g. `stealth/ox-alpha`, with
  "Reasoning is mandatory for this endpoint and cannot be disabled") but
  measurably hurt review quality by forcing it off; `ask` no longer sends
  the `reasoning` field at all, leaving it to each model's own default,
  with `max_tokens` raised from 8,192 to 32,768 so reasoning tokens don't
  starve the answer, and it now logs `finish_reason`/`usage` on every
  response and the raw answer on a JSON-parse failure, so a truncated or
  malformed answer is diagnosable instead of a bare traceback
  ([#611](https://github.com/discopy/discopy/issues/611)).
- `style-review.yml` diffed `-- '*.py'` only, so a pull request touching
  only a `docs/notebooks/*.md` marimo notebook always diffed empty: the
  review step was skipped silently and the correctness reviewer was called
  with no style pass at all. The diff now covers every authored file —
  Python, notebooks, docs, workflows, config — excluding generated
  artefacts (`docs/_static/**`, `discopy/*.gif`, `test/drawing/tikz/**`,
  `test/fixtures/**`, `uv.lock`). `review.py` fences each changed file by
  its own type (`python`, `markdown`, `yaml`, …) instead of assuming
  everything is Python, and picks a fence at least one backtick longer
  than any run already inside the file, so a notebook's own cell fences
  or an inline code span can never close it early. Each changed file is
  now sent once, not twice: rather than the full new file followed by a
  separate global diff, `review.py` asks git for the full-context
  (`-U100000`) diff of each file and turns it into one listing — every
  added or context line numbered by its position in the new file, with a
  leading `+` for one added; a removed line carries a `-` instead and no
  number, since it has none in the new file — reusing git's own diff
  algorithm instead of reimplementing it
  ([#633](https://github.com/discopy/discopy/pull/633)).
- `no-todo-on-main.yml`'s guard reads the pull request's live `draft`
  field rather than `github.event.pull_request.draft`, a snapshot taken
  when the event fires and stale by however long the event then waited
  for delivery. On [#633](https://github.com/discopy/discopy/pull/633) a
  `synchronize` delivered thirteen minutes late read `false` although the
  guard's own previous run had drafted the pull request fifty seconds
  earlier; the "make ready" branch is gated on that state, so neither
  branch fired and the pull request stayed draft with no `TODO.md` and
  nothing to correct it. The guard also leaves the decision to the newer
  run when the branch has already moved past the event it is handling,
  rather than drafting a head that no longer exists behind its back
  ([#640](https://github.com/discopy/discopy/issues/640)).
- `build.yml` timeouts and a bounded, retried Graphviz install
  ([#591](https://github.com/discopy/discopy/issues/591)).
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
- Non-linear terms in `discopy.closed`: an `Application` with no free variables
  builds instead of raising, and its free variables keep first-occurrence order
  rather than going through a set whose iteration order depends on hashing
  ([#542](https://github.com/discopy/discopy/issues/542),
  [#543](https://github.com/discopy/discopy/issues/543)).
- `closed.Abstraction` discards a variable that does not occur in the body
  instead of raising, and nested abstractions curry the abstracted wire rather
  than the first one, so `eval` preserves `dom` and `cod`
  ([#541](https://github.com/discopy/discopy/issues/541),
  [#544](https://github.com/discopy/discopy/issues/544)).
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
- `python.additive.Function.trace` fed a looping output tag straight back
  in as an input tag, reading the wrong traced summand (or raising
  `IndexError`) whenever `dom` and `cod` have different lengths
  ([#554](https://github.com/discopy/discopy/issues/554)).
- Both branches of `closed.Abstraction.eval` curry on the right: the
  context branch curried out the wrong end of its domain, so an abstraction
  applied to an argument sharing a free variable did not compose, and a
  left abstraction evaluates through its right counterpart
  ([#562](https://github.com/discopy/discopy/issues/562)).
- `Tensor.spider_factory` returns its array on the active backend instead
  of always on NumPy, so diagrams with spiders evaluate — and
  differentiate — under the PyTorch backend
  ([#582](https://github.com/discopy/discopy/issues/582)).
- `trace(0)` is the identity, i.e. the vanishing axiom, rather than a
  morphism with empty `dom` and `cod`: `x[:-n]` is the empty prefix at
  `n == 0`, which emptied the boundary of `Hypergraph.trace` and of both
  `python.Function.trace`, and made `rigid.Diagram.curry(0, left=True)`
  curry the whole domain
  ([#578](https://github.com/discopy/discopy/issues/578)).
- Closed and biclosed diagrams containing a `Copy`, `Merge`, `Swap`,
  `Permutation`, `Braid` or `Twist` can be drawn: the `markov`, `symmetric`,
  `braided` and `balanced` functor branches now check that the codomain has
  the structure before using it, the way `biclosed.Functor` already did for
  `ev`, `exp` and `curry`
  ([#491](https://github.com/discopy/discopy/issues/491),
  [#548](https://github.com/discopy/discopy/issues/548)).
- `Double`'s `H*` structure is built by transposition instead of the dagger,
  which wrongly conjugated complex structure constants — invisible on the
  real examples of #405, wrong for `taft(3)`
  ([#484](https://github.com/discopy/discopy/pull/484)).

### Performance

- `CMap.permutation` encodes a permutation as boundary wiring — one
  involution, no boxes — instead of inheriting `abc.SymmetricCategory`'s
  default, which composes one swap per inversion and made
  `CMap.from_diagram` quadratic in the width of every permutation layer
  it met. On a string-matching circuit with 147 boxes and one
  frontier-wide permutation per alignment, `to_map` drops from 94 to 4
  seconds, and the wiring is checked identical to the swap-composed one.
- The elements of a Hopf algebra (`drinfeld_element`, `pivotal_element`,
  `ribbon_element`) contract each structural generator once through the
  cached `Algebra.arrays` and solve for the pivot with a thin SVD, so that
  `Double(taft(3)).ribbon_element` takes under a second instead of twenty
  ([#484](https://github.com/discopy/discopy/pull/484)).
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
- `CMap.from_diagram` is linear rather than quadratic in the number of
  boxes: `CMap.from_glued` glues the image of each box onto a scan of
  open wires in a single pass, instead of folding the images with
  `then` and re-validating the whole prefix at every step. This speeds
  up `Diagram.eval` on every tensor backend
  ([#525](https://github.com/discopy/discopy/pull/525)).

### Project

- The `TODO.md` rule of `RULES.md` is split in two: creation stays point 1,
  and a new point 2 has the agent delete its own `TODO.md` once every
  point is `[x]` or filed as an issue, taking the pull request out of draft:
  the style reviewer gives it a first pass before a human deep-reads it.
  A round of review feedback — bot or human — starts a fresh `TODO.md`,
  deleted again when the round is done; nitpicks are just fixed and
  resolved. Rule 4, only talk when prompted, is removed
  ([#608](https://github.com/discopy/discopy/pull/608)).
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
