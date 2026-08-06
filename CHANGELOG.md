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

### Changed

- `CMap` conversion now preserves concrete diagram categories, compares maps
  through their underlying hypergraphs, and separates acyclicity from box order
  with `is_acyclic`, `is_topologically_ordered`, and `topological_order`
  ([#532](https://github.com/discopy/discopy/pull/532)).
- `CMap` follows the `Hypergraph` terminology with `is_monogamous` and
  `make_monogamous`, `is_causal` meaning monogamous, acyclic and topologically
  ordered ([#532](https://github.com/discopy/discopy/pull/532)).
- `ev`, `curry` and `uncurry` are defined once in `abc.RigidCategory` from cups
  and caps, and `uncurry` once in `abc.BiclosedCategory` from evaluation, so
  that rigid diagrams and `CMap` share the same definitions rather than
  duplicating them. Rigid diagrams gain `uncurry` and `BiclosedCategory.uncurry`
  gains an argument `n` for the number of objects to uncurry
  ([#532](https://github.com/discopy/discopy/pull/532)).
- `CMap.curry` and `CMap.uncurry` use the compact structure of the wiring when
  the host category is rigid, otherwise the currying and evaluation stay
  explicit boxes ([#532](https://github.com/discopy/discopy/pull/532)).
- CMap planarity is now diagnostic rather than a downgrade constraint
  ([#532](https://github.com/discopy/discopy/pull/532)).
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

### Fixed

- Pivotal diagram-to-map conversion now encodes cups and caps as `CMap`
  wiring rather than keeping them as boxes
  ([#532](https://github.com/discopy/discopy/pull/532)).
- Rigid `CMap` downgrade now distinguishes left and right adjoints, rejecting
  bends whose crossed wiring would require an unavailable swap
  ([#532](https://github.com/discopy/discopy/pull/532)).
- Tensor networks are contracted with `opt_einsum` when the number of
  indices exceeds `numpy.einsum`'s 52-index limit
  ([#448](https://github.com/discopy/discopy/pull/448)).
- Hypergraph hash
  ([#387](https://github.com/discopy/discopy/pull/387)).
- Bubble drawing
  ([#431](https://github.com/discopy/discopy/pull/431)).

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
