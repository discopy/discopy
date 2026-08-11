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

- `test/neural/test_sudoku_act_e2e.py`, an end-to-end check that trains model
  C through the real ACT loop on the Palm et al. (2018) benchmark and then
  reads it with the noise study's own `latent_stats` and `run_segment`: the
  first test in which the model is good enough for "noise costs board
  accuracy" to be falsifiable. It is marked `neural_e2e` and deselected by
  default — three minutes, a GPU and a cached benchmark — so run it with
  `pytest -m neural_e2e` when `discopy.neural` changes.

### Fixed

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
- Tensor networks are contracted with `opt_einsum` when the number of
  indices exceeds `numpy.einsum`'s 52-index limit
  ([#448](https://github.com/discopy/discopy/pull/448)).
- `grammar.categorial.cat2ty` reads a fully parenthesized category such as
  `(S\NP)` as a category rather than an atom, strips CCGbank features
  wherever they occur rather than on atoms only, and associates slashes to
  the left as CCG does
  ([#528](https://github.com/discopy/discopy/issues/528)).
- Hypergraph hash
  ([#387](https://github.com/discopy/discopy/pull/387)).
- Bubble drawing
  ([#431](https://github.com/discopy/discopy/pull/431)).
- Controlled gate drawing: the control wire is anchored on the indexed
  input of the controlled box rather than its first one, so gates with a
  classical wire or a distance other than one are drawn on the right wires
  ([#439](https://github.com/discopy/discopy/pull/439)).

### Performance

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
