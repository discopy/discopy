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
- `discopy.kleisli`, the Kleisli category of a `Monad`, with the channel,
  multiplicative and additive instances. `Monad` carries an optional
  `iterate` operator, i.e. the structure of an Elgot monad, so that
  `kleisli.additive.Channel.trace` is defined exactly for the monads that
  supply one, with Dal Lago-Hoshino's token machines as a worked example
  where conditioning is a trace
  ([#443](https://github.com/discopy/discopy/pull/443)).

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

### Fixed

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
