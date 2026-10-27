# Changelog

All notable changes to DisCoPy are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since [`1.2.2`](https://github.com/discopy/discopy/releases/tag/1.2.2).

### Added

- `discopy.abc`, the abstract base classes of the hierarchy, from `Category`
  to `PROP`, `Monoid` and `Pregroup` ([#326], [#331], [#727]).
- `discopy.testable`, laws stated once on `abc` and checked by the Hypothesis
  matrix in `proptest/` on every level of the hierarchy, each free category
  drawing its terms from its own factories, with the laws of every doctrine
  on `abc` and the shapes of their arguments ([#658], [#817], [#818], [#829]).
- `discopy.cmap`, combinatorial maps for diagrams in compact categories, with
  `genus`, `is_planar` and `to_compact` ([#338], [#532], [#690]).
- `discopy.para`, parametric and coparametric maps over any symmetric
  category ([#559], [#575]).
- `discopy.hopf`, ribbon semantics in the representations of a
  finite-dimensional Hopf algebra and its Drinfeld double ([#405], [#484]).
- `grammar.abstract`, abstract categorial grammars ([#400]), and
  `categorial.Ty` with type raising and crossed composition ([#332], [#341]).
- Lambda terms in `biclosed` and `closed`, with products, `Let` and
  capture-avoiding substitution, to and from diagrams ([#322], [#332], [#400],
  [#489]).
- 2-categories, i.e. coloured regions, `None` where there is no colour, drawn
  with an opt-in legend ([#354], [#355], [#357], [#754], [#830]).
- `symmetric.Permutation` and `symmetric.Layer`, with `Swap` the two-wire
  permutation ([#362], [#594]).
- `monoidal.List`, the free monoid on a type ([#747]).
- `cat.Transformation`, natural transformations between functors ([#351]).
- `Equation` at every level, true when its terms are equal up to that level's
  axioms, e.g. `symmetric.Equation` up to hypergraph isomorphism ([#415]).
- `Sum` in `traced`, `pivotal`, `compact` and `frobenius` ([#812]).
- Ribbon diagrams drawn as dual rails ([#358], [#360]).
- Rich display of diagrams and drawings in Jupyter ([#445]).

### Changed

- DisCoPy requires Python 3.12 and is packaged with uv ([#311]).
- `closed` is the closed *symmetric* monoidal category; the planar one, with
  `Over` and `Under`, is `biclosed` ([#322]).
- An arrow class is its own category, so a functor takes `cod=Tensor` where
  it took `cod=Category(Dim, Tensor)` ([#326]).
- `Functor(ob=, ar=)` is `Functor(ob_map=, ar_map=)`, and `Arrow.ty_factory`
  and `Arrow.factory` are `Arrow.ob` and `Arrow.ar` ([#329], [#330], [#369]).
- `drawing.Equation` is `monoidal.Equation` ([#415]).
- Symmetric categories build their swaps with `swap_factory` ([#440]), and
  `Hypergraph[C]` replaces `hypergraph_factory` ([#379], [#532]).
- A `monoidal.Layer` is a list of boxes and types with at least one box,
  rather than an odd-length list alternating them ([#438], [#590], [#603]).
- `+` is no longer the tensor of types, use `@` ([#747]).
- `python.Function` types its wires with `List[type]` rather than tuples,
  `python.finset` and `Matrix` with `Nat` rather than `int` ([#727], [#747]).
- The quantum `SWAP` is a gate, kept by `to_tk`; the symmetry is still
  `Circuit.swap` ([#594]).
- Tensor networks are contracted in one `einsum` through `CMap`, with
  `opt_einsum` past 52 indices ([#402], [#448], [#812]).
- Drawings fit their labels ([#364], [#365]), are reproducible ([#457],
  [#469]), and as SVG adapt to dark mode ([#725]).
- Faster `Ty` construction ([#420]), `Hypergraph` equality ([#353]),
  `Hypergraph.from_diagram` ([#644]) and `quantum.Channel` ([#812]).

### Deprecated

- Every `Ob` but `cat.Ob` is `Wire` ([#566]), and `PRO` is `Nat` ([#727]):
  the old names still work, with a `DeprecationWarning`.

### Removed

- Parameters nothing read: `Tree.to_diagram(contravariant)`,
  `Tree.from_nltk(lexicalised, word_types)`, `Drawing.id(length)`,
  `markov.Diagram.discard(n)` and `Backend(linewidth)` ([#768]).

### Fixed

- `Hypergraph`: its hash, `simplify`, `rotate`, and `to_diagram` on
  non-self-dual, `Nat` and `Dim` types ([#324], [#387], [#393], [#622],
  [#719]).
- A box and the diagram equal to it have the same hash ([#306]).
- `frobenius.Functor` keeps the phase of a spider, e.g. ZX ([#622]).
- `trace(0)` is the identity ([#588]).
- `Bubble.dagger` ([#629]) and the drawing of bubbles ([#431], [#564]).
- `str(Ty(""))` ([#312]), and `Eval` and `Curry` read back from their `repr`
  ([#400]).
- `discopy.quantum`: `CQ.__str__`, `Measure(override_bits=True)`, `Encode`
  types, `CRz`/`CRx`/`CU1` at any distance, `Sqrt.dagger` and
  `Channel[float]` ([#812]).
- `Matrix.trace(left=True)`, `Tensor.to_quimb(dtype)` and `jacobian(params)`
  ([#768]), and spiders on PyTorch ([#584]).
- `python.additive.Function.trace` on wires of different lengths ([#561]),
  and `python.Function.tensor` is variadic ([#489]).
- `grammar.categorial.cat2ty` on brackets, features and associativity
  ([#529]).
- Drawing controlled gates ([#439]) and discards ([#514]).
- `Braid`, `Trace`, `Twist`, `Copy`, `Merge`, `Spider`, `Feedback` and `Dim`
  read back from their `repr` and their tree ([#817]).

### Project

- `AGENTS.md`, `RULES.md` and `STYLE.md` for agents, and an LLM contribution
  needs its own handle and a human approval other than its prompter's
  ([#378], [#720]).
- Notebooks are marimo ([#404]), doc images are SVG drawing tests ([#419],
  [#470]), and `test/` mirrors `discopy/` ([#403]).
- `benchmark/` and `proptest/` suites, run on `main` and on pull requests
  labelled `benchmark` or `proptest` ([#346], [#459], [#658]).
- `pylint` fails on unused names and on a score below `fail-under` ([#768]).
- CodeRabbit reviews style, and a pull request is a draft while it has a
  `TODO.md` ([#485], [#708]).
- The notebook pages of the docs are titled by their first heading ([#825]).
- Every pull request gets a preview of the docs from Read the Docs ([#824]).

## [1.2.2] - 2025-12-19

See the [GitHub release](https://github.com/discopy/discopy/releases/tag/1.2.2).

## Older releases

See the [GitHub releases page](https://github.com/discopy/discopy/releases)
for the changelog of `1.2.1` and earlier.

[#306]: https://github.com/discopy/discopy/pull/306
[#311]: https://github.com/discopy/discopy/pull/311
[#312]: https://github.com/discopy/discopy/pull/312
[#322]: https://github.com/discopy/discopy/pull/322
[#324]: https://github.com/discopy/discopy/pull/324
[#326]: https://github.com/discopy/discopy/pull/326
[#329]: https://github.com/discopy/discopy/pull/329
[#330]: https://github.com/discopy/discopy/pull/330
[#331]: https://github.com/discopy/discopy/pull/331
[#332]: https://github.com/discopy/discopy/pull/332
[#338]: https://github.com/discopy/discopy/pull/338
[#341]: https://github.com/discopy/discopy/pull/341
[#346]: https://github.com/discopy/discopy/pull/346
[#351]: https://github.com/discopy/discopy/pull/351
[#353]: https://github.com/discopy/discopy/pull/353
[#354]: https://github.com/discopy/discopy/pull/354
[#355]: https://github.com/discopy/discopy/pull/355
[#357]: https://github.com/discopy/discopy/pull/357
[#358]: https://github.com/discopy/discopy/pull/358
[#360]: https://github.com/discopy/discopy/pull/360
[#362]: https://github.com/discopy/discopy/pull/362
[#364]: https://github.com/discopy/discopy/pull/364
[#365]: https://github.com/discopy/discopy/pull/365
[#369]: https://github.com/discopy/discopy/pull/369
[#378]: https://github.com/discopy/discopy/pull/378
[#379]: https://github.com/discopy/discopy/pull/379
[#387]: https://github.com/discopy/discopy/pull/387
[#393]: https://github.com/discopy/discopy/pull/393
[#400]: https://github.com/discopy/discopy/pull/400
[#402]: https://github.com/discopy/discopy/pull/402
[#403]: https://github.com/discopy/discopy/pull/403
[#404]: https://github.com/discopy/discopy/pull/404
[#405]: https://github.com/discopy/discopy/pull/405
[#415]: https://github.com/discopy/discopy/pull/415
[#419]: https://github.com/discopy/discopy/pull/419
[#420]: https://github.com/discopy/discopy/pull/420
[#431]: https://github.com/discopy/discopy/pull/431
[#438]: https://github.com/discopy/discopy/pull/438
[#439]: https://github.com/discopy/discopy/pull/439
[#440]: https://github.com/discopy/discopy/pull/440
[#445]: https://github.com/discopy/discopy/pull/445
[#448]: https://github.com/discopy/discopy/pull/448
[#457]: https://github.com/discopy/discopy/pull/457
[#459]: https://github.com/discopy/discopy/pull/459
[#469]: https://github.com/discopy/discopy/pull/469
[#470]: https://github.com/discopy/discopy/pull/470
[#484]: https://github.com/discopy/discopy/pull/484
[#485]: https://github.com/discopy/discopy/pull/485
[#489]: https://github.com/discopy/discopy/pull/489
[#514]: https://github.com/discopy/discopy/pull/514
[#529]: https://github.com/discopy/discopy/pull/529
[#532]: https://github.com/discopy/discopy/pull/532
[#559]: https://github.com/discopy/discopy/pull/559
[#561]: https://github.com/discopy/discopy/pull/561
[#564]: https://github.com/discopy/discopy/pull/564
[#566]: https://github.com/discopy/discopy/pull/566
[#575]: https://github.com/discopy/discopy/pull/575
[#584]: https://github.com/discopy/discopy/pull/584
[#588]: https://github.com/discopy/discopy/pull/588
[#590]: https://github.com/discopy/discopy/pull/590
[#594]: https://github.com/discopy/discopy/pull/594
[#603]: https://github.com/discopy/discopy/pull/603
[#622]: https://github.com/discopy/discopy/pull/622
[#629]: https://github.com/discopy/discopy/pull/629
[#644]: https://github.com/discopy/discopy/pull/644
[#658]: https://github.com/discopy/discopy/pull/658
[#690]: https://github.com/discopy/discopy/pull/690
[#708]: https://github.com/discopy/discopy/pull/708
[#719]: https://github.com/discopy/discopy/pull/719
[#720]: https://github.com/discopy/discopy/pull/720
[#725]: https://github.com/discopy/discopy/pull/725
[#727]: https://github.com/discopy/discopy/pull/727
[#747]: https://github.com/discopy/discopy/pull/747
[#754]: https://github.com/discopy/discopy/pull/754
[#768]: https://github.com/discopy/discopy/pull/768
[#812]: https://github.com/discopy/discopy/pull/812
[#817]: https://github.com/discopy/discopy/pull/817
[#818]: https://github.com/discopy/discopy/pull/818
[#824]: https://github.com/discopy/discopy/pull/824
[#825]: https://github.com/discopy/discopy/pull/825
[#829]: https://github.com/discopy/discopy/pull/829
[#830]: https://github.com/discopy/discopy/pull/830
