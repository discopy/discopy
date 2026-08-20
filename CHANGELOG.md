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
- The pivotal structure of `Rep(H)`: `HopfAlgebra.drinfeld_element`,
  `pivotal_element` and `ribbon_element`, cached single tensors named after
  the literature (Reshetikhin–Turaev; Kassel; Radford), with pivotal cups
  and caps twisting the dual leg so all four orientations are intertwiners.
  `taft(n)`, the smallest algebras with a pivot of order `n` (Sweedler's
  algebra is `n = 2`), realise the Kauffman–Radford ribbon criterion
  ([#484](https://github.com/discopy/discopy/pull/484)).

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

### Fixed

- A boxless `monoidal.Layer` can no longer be placed inside a `Diagram`:
  `Diagram.__init__` raises `ValueError` for a layer with no box, restoring
  the invariant that every layer holds at least one box and that the identity
  diagram is the empty sequence of layers. Such a layer is the internal unit
  of `Layer.tensor`, built by `Layer.id` and merged away by `Layer.normalise`;
  put inside a diagram by hand it survived `normal_form` and made `foliation`
  and `draw` raise. The check is gated on `_scan`, so the internal fast paths
  that build layers by construction are unaffected
  ([#599](https://github.com/discopy/discopy/issues/599)).
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
