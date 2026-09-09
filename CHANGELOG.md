# Changelog

All notable changes to DisCoPy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since [`1.2.2`](https://github.com/discopy/discopy/releases/tag/1.2.2).

### Added

- `abc.Nat`, a concrete dataclass for the free monoid on one generator
  (`n: int` with addition as `tensor`), and `abc.PRO`/`abc.PROB`/`abc.PROP`,
  the `MonoidalCategory`/`BraidedCategory`/`SymmetricCategory` whose objects
  are `Nat` — `PROB(PRO, BraidedCategory[Nat, C1])` and
  `PROP(PROB, SymmetricCategory[Nat, C1])`, mirroring how
  `abc.SymmetricCategory` already extends `abc.BraidedCategory` directly.
  `abc.Nat` also gets `__index__` (so `range(n)`/`int(n)` work whether `n`
  is a plain `int` or a `Nat`) and its `tensor` now returns `NotImplemented`
  for a non-`Nat` argument, like `monoidal.FreeMonoid.tensor` already does
  — needed to let `@` fall back to the other operand's `__rmatmul__` for
  whiskering, e.g. `Nat(1) @ some_morphism`, which previously crashed with
  `AttributeError` instead of building the identity on `Nat(1)` first.
  `python.finset.Function`/`Permutation.ob` changes from a raw `int` to
  `Nat`, its `dom`/`cod` now genuinely `Nat` instances (auto-cast from `int`
  at construction, the same convenience `monoidal.Diagram` already gives
  any `ob = Nat` subclass) rather than merely claiming to be one without
  the objects to match; `Permutation` inherits `abc.PROP` on the strength
  of that, its first genuine user
  ([#709](https://github.com/discopy/discopy/issues/709)).
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
  in production. On its first runs shellcheck found the `A && B || C` in
  `benchmark.yml`'s summary step, now an `if`
  ([#645](https://github.com/discopy/discopy/pull/645)).
- `.github/actions/setup`, one composite action for installing uv, Python,
  the project and, for the jobs that draw, Graphviz. The three `build.yml`
  jobs called for it four times between them and the Graphviz incantation
  was byte-identical twice. `benchmark.yml` keeps its own steps: it checks
  out two arbitrary commits and one of them predates this action
  ([#645](https://github.com/discopy/discopy/pull/645)).
- `.github/dependabot.yml`, grouping the monthly GitHub Actions updates
  into one pull request, now that every action is pinned by commit
  ([#645](https://github.com/discopy/discopy/pull/645)).
- `Diagram.to_compact` and `CMap.to_compact`, bending curry bubbles into
  coevaluation and feedback. Since a biclosed category has no trace, the
  `biclosed` method lands in `CMap`, which is compact whatever hosts it,
  while the `closed` one stays in diagrams. Unlike `rigid.to_rigid` and
  `interaction.Int`, this keeps the exponential atomic and bends the wire
  with `biclosed.Coeval`, the transpose of `Eval`, which a biclosed
  category only has when its exponential is read at a reflexive object
  ([#532](https://github.com/discopy/discopy/pull/532)).
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

### Changed

- `abc.TracedCategory` inherits from `abc.FeedbackCategory`, which moves
  down the hierarchy from `MarkovCategory` to `MonoidalCategory`: a traced
  category is a feedback category whose delay is trivial, so
  `TracedCategory` implements `delay` as the identity and `feedback` as the
  trace over `mem` — stated once for every traced carrier, i.e. the free
  traced, balanced, symmetric, markov, pivotal, ribbon, compact and
  frobenius diagrams, `Drawing` and `CMap`, while `stream.Stream` now
  declares the `FeedbackCategory` it already implements.
  `Hypergraph` keeps its `MonoidalCategory` base on purpose: a hypergraph
  is parameterised by its host category, and over a feedback host such as
  `feedback.Hypergraph` its delay is not trivial, so it must not inherit
  the trace-based `feedback` — `feedback.Functor` keeps `Feedback` bubbles
  as opaque boxes exactly when its codomain has no `feedback` method, which
  is how `feedback.Equation` compares diagrams. `monoidal.Ty.delay` is
  the identity, overridden by `feedback.Ty`, the same way `unwind` is
  trivial until `rigid.Ty` overrides it. The note in `discopy.feedback`
  showing that every traced category is a feedback category used to
  monkey-patch `symmetric`; its doctest now runs on the built-in methods.
  The free `feedback.Diagram` and `feedback.Ty` cannot conversely become
  base classes of `traced.Diagram` and `traced.Ty`: they are already their
  subclasses through `markov`, since the free feedback category is
  symmetric with a non-trivial delay, so the concrete classes inherit the
  other way around and the shared interface lives in `abc`
  ([#710](https://github.com/discopy/discopy/issues/710)).
- `monoidal.PRO` (and its counterparts `rigid.PRO`, `pivotal.PRO` and
  `frobenius.PRO`) is renamed to `Nat`: it is the free monoid on one
  generator, natural numbers with addition as tensor, and its unary
  encoding was already exposed through the sequence protocol
  (`len`, iteration and slicing, e.g. `Nat(3)[:1] == Nat(1)`), just under
  the wrong name — `PRO` is the name for the monoidal category with `Nat`
  as objects, see `abc.PRO` above. `abc.Nat` carries the concrete
  behaviour (its dataclass field `n`, `tensor` as addition, the sequence
  protocol), so `monoidal.Nat` only adds what a `Ty` needs on top: `dom`,
  `cod`, `inside`, serialisation and the whiskering-aware `tensor` that
  raises on a mismatched `Ty` rather than silently reinterpreting it.
  `monoidal.Functor.__call__` maps a `Nat` by mapping its single generator
  once and folding that image `other.n` times with `+`, rather than mapping
  each of the `n` identical atoms separately: a `Nat` is a unary encoding,
  so every atom is the same generator and its image need only be computed
  once. The fold starts from the image's own unit (`image[:0]`) rather than
  the declared codomain unit `cod.ob()`, since the latter can be a supertype
  of the image — `Diagram.to_hypergraph` on a `Nat`-typed permutation maps a
  `Nat` boundary through a functor whose `cod.ob` is the category's generic
  `Ty`, and `Ty() @ Nat` is refused. The fold is with `+` like the
  pre-existing `Dim`/`Ty` branches, so an arbitrary codomain's objects need
  only support `+`, e.g. `python.Function.ob = tuple[type, ...]`, whose
  images are plain tuples with a `+` but no `__matmul__` at all. The old names still work
  through a `DeprecationWarning`, via a new `utils.deprecated_alias` taking a
  mapping of every name a module deprecates. `utils.deprecated_ob`, the
  single-purpose `Ob`→`Wire` wrapper it generalises, is removed: its six call
  sites (`biclosed`, `braided`, `compact`, `feedback`, `grammar.pregroup`,
  `quantum.circuit`) now call `deprecated_alias(__name__, {"Ob": "Wire"})`
  directly, the same as `rigid`/`pivotal`/`frobenius`/`monoidal` already do
  for their `PRO`→`Nat` alias
  ([#709](https://github.com/discopy/discopy/issues/709)).
- Matplotlib SVGs adapt to the page behind them: they are saved on a
  transparent canvas and open with a `prefers-color-scheme: dark` media
  query that turns the elements drawn black on that canvas — wires, braids,
  wire labels, spiders and their labels, control dots — white on a dark
  page, so a single SVG file reads on both light and dark backgrounds.
  Elements whose readability does not depend on the page keep their static
  colours: box interiors stay white with black labels, coloured regions
  keep their fill and the black strokes over them. White spiders, e.g. the
  symbol of an `Equation`, are drawn unfilled so they leave no white patch
  on a non-white page, and raster formats keep their white background since
  they cannot adapt. The docs let content images follow the theme toggle by
  setting their `color-scheme`, which propagates into the SVG media query,
  instead of painting a white plate behind them in dark mode
  ([#453](https://github.com/discopy/discopy/issues/453), superseding the
  static outlines of
  [#497](https://github.com/discopy/discopy/pull/497)). The hand-drawn
  snake equation of the README header adapts the same way, replacing its
  separate `snake-equation-dark.svg`, and the unreferenced
  `frobenius-axioms.svg` is deleted.
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
  API is `urllib`, from the standard library.
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

### Removed

- `cat.Bubble.dagger`: a bubble's dagger was inherited from `Box.dagger`,
  which reconstructs with `type(self)(name, cod, dom, ...)` — positional
  arguments `Bubble.__init__` reads as `*args`, so it crashed with
  `AttributeError` on the very first (non-arrow) argument. `Bubble` now
  daggers each of its `args`, swaps `dom`/`cod` and carries `data`/`is_dagger`
  through like `Box.dagger` does
  ([#55](https://github.com/discopy/discopy/issues/55)).
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
- The in-house style reviewer — `.github/style-review/` (the `review.py`,
  `post.py`, `history.py`, `thread.py` and `github.py` scripts and their
  `prompt.md`), the `style-review.yml` workflow, and their tests under
  `.github/tests/` — is retired in favour of CodeRabbit, configured by a
  new `.coderabbit.yaml` that restates `STYLE.md` as per-path review
  instructions. It was built around our own open-weights model behind an
  OpenAI-compatible gateway, and around a cross-round `accepted`/`declined`/
  `open` tally kept in hidden review bodies; CodeRabbit is free for public
  repositories, so the gateway (and the `STYLE_REVIEW_BASE_URL`/`_MODEL`
  variables and `STYLE_REVIEW_API_KEY` secret it read) is no longer needed.
  Correctness review is unchanged — cubic keeps that lane — but the two
  reviewers now run as independent GitHub Apps on pull request events, so
  the style→correctness hand-over the workflow orchestrated (the source of
  #634/#645/#676) is gone rather than reimplemented. The `no-todo-on-main`
  draft gate stays: a draft carries its `TODO.md` and CodeRabbit skips
  drafts, so deleting `TODO.md` still hands a pull request to the style
  reviewer first.

### Fixed

- `Hypergraph.rotate` exchanged the two boundaries of the hypergraph and
  replaced each box by its rotation, but left the *ports* of those boxes
  and the spiders where they were: the wires reading a box's domain went
  on reading its domain although the rotated box's domain is its old
  codomain, and a spider typed `a` stayed `a` under a rotation that made
  every port around it `a.r`. Both are invisible on an endomorphism of a
  self-dual type, which is most of what the drawing and conversion tests
  rotate — `test_Hypergraph_rotate` rotated the identity and nothing
  else. Anything else raised: a bare `ValueError` from
  `Hypergraph.__init__` when the two arities differ, an `AxiomError` on
  the spider types when they do not. `.l` and `.r` are involutions again
  ([#716](https://github.com/discopy/discopy/issues/716)).
- Region painting computes the exact extents of each coloured region —
  polygons bounded by the wires on both sides, subdivided per height band —
  instead of overpainting everything to the right of each wire up to the
  full canvas width: translucent colours are no longer painted twice where
  two regions of the same colour are adjacent, white regions are not
  painted at all, so they erase to the background, and neither is the
  inside of a box, which is a 2-cell rather than a region, so no colour
  can bleed out around its border
  ([#521](https://github.com/discopy/discopy/issues/521)).
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
- A bubble whose inside and outside have a different number of wires keeps
  its boundary. Drawing the sides of a square frame with zero width is now
  the business of `Drawing.slot` and `Drawing.frame`, which have the colours
  of the regions they separate to show the edge in their place, rather than
  of every bubble drawn as a square, which has none and so came out with no
  visible outline at all
  ([#520](https://github.com/discopy/discopy/issues/520),
  [#569](https://github.com/discopy/discopy/issues/569)).
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

- The elements of a Hopf algebra (`drinfeld_element`, `pivotal_element`,
  `ribbon_element`) contract each structural generator once through the
  cached `Algebra.arrays` and solve for the pivot with a thin SVD, so that
  `Double(taft(3)).ribbon_element` takes under a second instead of twenty
  ([#484](https://github.com/discopy/discopy/pull/484)).
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
- `Hypergraph.from_diagram` is linear rather than quadratic in the number
  of layers, mirroring `CMap.from_glued`: the new `Hypergraph.from_glued`
  glues the image of every box onto a scan of open wires with a single
  union-find pass, instead of folding the images with `then`, which
  recomputes the pushout and relabels every spider and box built so far
  at each layer. A closed loop left by gluing a cap directly onto a cup
  survives as a scalar spider, since it is never referenced by the
  scan and would otherwise vanish silently. This speeds up
  `symmetric.Equation`, `compact.Equation`, `frobenius.Equation`,
  `Hypergraph.simplify` and `Diagram.foliation`, all of which go through
  `Diagram.to_hypergraph`
  ([#623](https://github.com/discopy/discopy/issues/623)).
- `CMap.ports` is a `cached_property`, confirmed with a regression test
  rather than assumed from `CMap`'s immutability: `Hypergraph.from_map`
  reads it once per box, so a plain `@property` rebuilding the whole port
  list on every access made `CMap.to_hypergraph` quadratic in the number
  of boxes, 226 s at 3200 boxes. It is now linear, e.g. 65.6 ms at 800
  boxes and 294.9 ms at 3200, down from 5.4 s and 226 s
  ([#624](https://github.com/discopy/discopy/issues/624)).

### Project

- The docs build on Sphinx 7.4 rather than 7.2, whose `stringify_annotation`
  handled a `TypeVar` but not a `ParamSpec`, so a signature such as
  `Callable[Concatenate[type, P], T]` crashed autodoc on Python 3.14, where
  `typing.get_type_hints` resolves the PEP 695 type parameter. The pin and
  the lock move, `myst-parser == 2.0.*` allowing any Sphinx below 8, and the
  `drawing`, `grammar`, `python` and `quantum` API pages list their
  submodules without the module prefix, which Sphinx 7.4 warns against
  under `automodule`
  ([#722](https://github.com/discopy/discopy/issues/722)).
- `CONTRIBUTING.md`'s LLM guidelines require an LLM contribution to be
  authored under a GitHub handle separate from the human who prompted it,
  and a pull request authored by an LLM to be approved by at least one
  human other than the one who prompted it.
- `.claude/hooks/session-start.sh`, registered in `.claude/settings.json`
  as a `SessionStart` hook for Claude Code on the web, syncs the full
  development environment before the session starts, so that the linter
  and the whole test suite run as `CONTRIBUTING.md` says; without the
  registration the script is inert. When `download.pytorch.org`, the index
  `pyproject.toml` pins torch to on Linux, is not reachable from the
  session, it syncs everything but torch and installs the locked version
  from PyPI instead, whose wheels run on the CPU. Every agent session so
  far ran `pytest --skip-extra` and reported the torch tests skipped.
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
