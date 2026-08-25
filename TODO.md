# TODO.md

> Give this one a shot https://github.com/discopy/discopy/issues/472

- [x] Merge the symmetric-layer PR (#362) branch into this branch and resolve its conflicts with main
- [x] Split `markov` into `markov` (comonoid: copy, discard) and `comarkov` (monoid: merge, unit)
- [x] Add `finset.Function` and a `markov.Function` box holding the opposite of a function between finite sets
- [x] Make `markov.Layer` alternate between function-opposites and generators, the same way `symmetric.Layer` alternates permutations and generators
- [ ] Add `frobenius.Cospan` and make `frobenius.Layer` alternate cospans and generators (see issue #472)

## Port onto the merged #362 (🌙 evening, 2026-08-01)

#362 was **squash**-merged as `a4f7a73`, so git has no common ancestry for it
and merging `main` conflicts in 16 files / 49 hunks. That is an artifact; the
real problem is that #362 kept changing after the state this branch merged
(`afbcc44`), and the merged design drops three things this branch is built on.
A 3-way merge with `afbcc44` as base cuts the conflicts to 15 hunks; the rest
is a port, not a merge.

- [ ] Decide whether the base `Category` keeps routing stubs. Merged #362
      **removed** `Category.permutation`; this branch adds the mirror
      `Category.function`, and `test/monoidal.py::test_identity_function`
      asserts `monoidal.Diagram.function([0, 1], x @ y) == Id(x @ y)`.
      Keeping one without the other leaves the base class asymmetric —
      it wants a ruling before an implementation
- [x] @evening-2026-08-25T01:01Z `symmetric.Layer.permutations` is not
      restored: the merged representation no longer alternates strictly (a
      generator layer can hold consecutive boxes with no plumbing between
      them), so an even-index view is not meaningful any more. Adapted
      `test/markov.py::test_Layer` and `test/symmetric.py`'s
      `test_Layer_factory_ownership` to main's own idiom instead —
      `layer.boxes_or_types` and `layer.is_plumbing` — which already carry
      the same information.
- [x] @evening-2026-08-25T01:01Z `symmetric.Layer.plumbing` (main's actual
      name for the old `routing_factory` idea, from `Permutation` alone to
      `(monoidal.Ty, Permutation)`) is already exactly this extension point;
      only `Layer.normalise` and `Layer.is_plumbing` still hard-coded
      `Permutation`, generalised to read `cls.plumbing` / `self.plumbing`.
      `markov.Layer.plumbing = (monoidal.Ty, Function, symmetric.Permutation)`.
      The generalised `normalise` keeps main's `hasattr` guard (needed so
      unpickling a cyclic `box.inside == (Layer(box),)` does not crash on a
      structural box whose own state is not set yet).
- [x] @evening-2026-08-25T01:01Z `markov.Layer` and the module docstring
      rewritten: identity routing is a bare `Ty`, not a boxed identity
      `Function`; the doctest now checks `boxes_or_types`/`is_plumbing`
      instead of `layer[::2]`.
- [x] @evening-2026-08-25T01:01Z Took `main`'s current names, not this
      TODO's stale memory of them: `drawing_permutation` is gone (dropped
      from `config.py` and `drawing/drawing.py`'s `dagger`, replaced by
      `draw_as_permutation` + `permutation_indices`); `_is_crossing` is
      `config.is_crossing`; main never settled on `LAYERS_MUST_ALTERNATE` —
      it replaced the whole alternating invariant with
      `LAYERS_MUST_HAVE_A_BOX`, so `PERMUTATION_AT_ODD_INDEX` and
      `LAYERS_MUST_BE_ODD` are just dropped as dead.
- [x] @evening-2026-08-25T01:01Z Dropped `77eb4eb`: both `test/hopf.py`
      spots now read `from discopy import hopf, compact, tensor`, matching
      `main` (one dropped the extra `rigid` this branch had added, the
      other dropped the redundant `compact` this branch had added a second
      time); full suite passes with the imports as `main` has them.

## Port verification (2026-08-25, @evening)

- `uv run pflake8 discopy` is clean.
- `uv run coverage run -m pytest --skip-extra`: 693 passed, 51 skipped (the
  quantum/tensor extras this environment cannot install), 0 failed.
- Three latent bugs in already-merged, unreleased `main` code surfaced only
  once the port actually exercised them, and are fixed rather than routed
  around — logged in `CHANGELOG.md`'s `[Unreleased]` → `Fixed`:
  `abc.SymmetricCategory.permutation` tensoring objects with `@` where its
  own `tensor` helper uses `+`; `para.Hypergraph`/`para.Feedback` unable to
  instantiate because `abc.HypergraphCategory`/`FeedbackCategory` now also
  require `merge` (added `para.Comarkov`, mirroring `para.Markov`); and
  `drawing.Drawing.permutation` defined twice, the second (dead) copy
  shadowing the first.
- Not touched: the routing-stubs ruling point above, `frobenius.Cospan`
  (already deferred to #472).

---

# TODO — refactoring of PR #362 (Add symmetric.Layer)

Instruction from Alexis (@toumix), verbatim:

> There was some weird stuff happening in the tensor of layers which didn't
> make sense and overall the PR deserves some thorough refactoring (eg it was
> written before the new contributing guidelines and agents.md) please give it
> a go.

## Checklist

- [x] @codex-2026-07-24T12:46+0530 Refactor the PR around one explicit
      permutation-storage invariant, remove incidental complexity, and verify
      the result against focused and full tests.
- [x] Investigate the tensor-of-layers semantics: map every call path into
      `Permutation.tensor` (`abc.whisker`, `Layer.__matmul__`/`__rmatmul__`,
      direct calls) and pin down which branches are live.
- [x] Refactor `Permutation.then`/`tensor`/`dagger`/`__rmatmul__` so they all
      flow through the single chokepoint `Diagram.from_permutation`, with no
      case explosion; make the reason `@unbiased` does not fit visible in the
      structure (results leave the `Permutation` subtype, so the tail of the
      arguments is delegated to the result's own method).
- [x] Refactor `symmetric.Layer`: `__init__` (compact, validated), `cast`,
      drop `dagger` (subsumed by `monoidal.Layer` + `cat.Ob.dagger`), guard
      `merge` so `foliation()` of permutation layers stops crashing.
- [x] Move misplaced logic to `monoidal.Layer`: uniform `dagger`, fix the
      odd-slot validation bug in `__init__`, fix `free_symbols`/`subs` on
      foliated (5+ slot) layers.
- [x] Fix `symmetric.Functor.__call__` on `Permutation` with a
      length-changing ob map (currently a bare `ValueError`).
- [x] Style-guide sweep of the whole diff: no code comments, docstrings with
      doctests, short names, `eval(repr(x)) == x`, drawing backend comments.
- [x] Update docs and tests: module docstring, doctests, regression tests for
      foliation, whiskering, functors, `then()`/`tensor()` with no arguments.
- [x] Run `uv run pflake8 discopy` and `uv run coverage run -m pytest`, fix
      anything broken, record pre-existing failures.
- [x] @evening-2026-07-23T20:40 Fix coverage gate: delete dead braid-shadow
      code, test TikZ crossing + `Permutation` dunders.

## Deliberately left out (follow-ups agreed in review)

- `Swap` as a subclass of `Permutation` (own issue) — now
  [#444](https://github.com/discopy/discopy/issues/444), opened 2026-07-22 with the
  "new" dunder catching `(1, 0)" spelled out.
- Relaxation of the alternating-list `Layer` representation (#437).

Uniform storage of even slots was subsequently brought back into scope by the
corrected specification below.

## Guidance (🐦 birdsong, 2026-07-22)

- top of the dependency chain right now — #438 (layer-simplification, #437) is
  seeded and waiting for this to land before it starts, since it builds on the
  representation you land here. land this first, don't let it stall behind the
  lower-priority drafts.
- once merged, worth a beat to check whether #444 (Swap ⊂ Permutation) is now
  smaller/easier given the refactored chokepoint — not required, just likely.

## Verification (2026-07-22, @bridge-2026-07-22)

- `uv run pflake8 discopy` clean.
- `uv run coverage run -m pytest` on everything except quantum: 523 passed,
  0 failed. Excluded as environment-blocked (proxy forbids installing torch
  and pytket): `discopy/quantum`, `test/quantum`, `docs/notebooks/qnlp.ipynb`
  and 4 torch-only tests in `test/semantics/{tensor,matrix}.py` — all
  pre-existing, unrelated to this refactoring.
- Found and fixed a regression the PR had introduced: the README cooking
  example (`test/drawing/drawing.py::test_crack_two_eggs_at_once`) failed
  because `Layer.__eq__` compared classes asymmetrically; layer equality is
  now structural.
- Found and fixed `dumps`/`loads` breaking on `Permutation` boxes
  (`to_tree`/`from_tree` added).

## Codex refactor pass

Instruction from Alexis, verbatim:

> Refactor this symmetric Layer PR after Claude did a messy job
> https://github.com/discopy/discopy/pull/362

This section records the discarded first pass. Its ordinary-box invariant was
superseded by the corrected specification below.

The audit rejected the proposed `symmetric.Layer` representation. Putting a
`Permutation` in a `monoidal.Layer` type slot made equal diagrams behave
differently under composition and tensor, and broke `boxes`, `offsets`,
`encode`, `normalize`, substitution, compact rotation, and category factories.

The refactored invariant is that `Permutation` is an ordinary `Box` in an
ordinary box slot. Composition and tensor use the ordinary `Diagram`
operations so strict associativity is preserved, identity permutations are
empty diagrams, and semantic equality with swap networks is expressed by
`Equation`. Drawing metadata renders the ordinary box as a compact band
without rewriting the drawing graph.

- [x] Add regressions for setoid congruence, offsets/encoding, factories,
      length-changing functors, compact rotation, drawing graph integrity, and
      the finite-set `Sequence` contract.
- [x] Make `finset.Function` a real `Sequence`; permutation indexing now uses
      normal Python bounds instead of modulo wraparound.
- [x] Remove generated asset churn; let the `docs-static` job regenerate it.
- [x] Merge current `main` and run the full lint/test/coverage suite.
- [x] Prepare replacement PR title and description. The GitHub integration
      rejected the metadata update with HTTP 403, and `gh` has no authenticated
      host in this environment.

## Verification (2026-07-24, @codex)

- Merged `origin/main` at `b365bfa4`.
- `uv run pflake8 discopy` is clean.
- Post-audit focused suites: 108 tests and 57 doctests passed.
- Full suite: 766 passed, 1 skipped; the only 4 failures require the external
  Graphviz `dot` executable, which is not installed in this environment.
- Coverage after the full run: 98%.
- Exhaustive permutation and compact-rotation laws passed through arity 5;
  serialization and category-factory ownership passed across symmetric,
  compact, Markov, and inherited descendant categories.
- Native permutations survive foliation as boxes, and a 1,100-wire reverse
  permutation converts directly to a hypergraph without recursive swaps.

## Unrelated pre-existing drawing issues observed

- `Drawing.validate_attributes()` reaches `set(...) + set(...)` and raises
  `TypeError` on otherwise valid drawings.
- Daggering a multi-box `Drawing` can fail validation because relabeling does
  not preserve the box-node order expected by `validate_attributes()`.

## Corrected specification (2026-07-24)

Instruction from Alexis, verbatim:

> You misunderstood the spec: we want permutations everywhere, not
> permutations as ordinary boxes so a layer is an alternation of permutation,
> generator, permutation. In a first iteration the swaps are distinct from
> (1, 0) permutations and are considered as generators so that we don't need
> to change much of the code.

A symmetric layer represents
`p_0 @ f_1 @ p_1 @ ... @ f_n @ p_n`, where each `p_i` is a permutation and
each `f_i` is a non-permutation generator. `Swap` remains distinct from
`Permutation(..., [1, 0])` and occupies a generator slot.

- [x] @codex-2026-07-24T15:55+0530 Rework the PR around permutation-valued
      layer slots, preserving the existing generator treatment of swaps.
- [x] Normalise every even slot, including identities, to the category's
      concrete `Permutation` factory; reject permutations in odd slots.
- [x] Keep `Swap` as a distinct odd-slot generator and preserve its semantic
      equality with the corresponding permutation only through `Equation`.
- [x] Canonicalise identity and adjacent permutation-only layers so
      composition, tensor, dagger and whiskering respect diagram equality.
- [x] Preserve compact rotation, feedback delay, functor application, drawing,
      hypergraph conversion and descendant-category factories.
- [x] Make offset-based operations fail explicitly on non-identity routing
      rather than silently treating a permutation as a type.
- [x] Upgrade legacy JSON and pickle representations without rebuilding
      incomplete cyclic `Box`/`Layer` objects during unpickling.

## Corrected-spec verification (2026-07-24, @codex)

- `uv run pflake8 discopy` is clean.
- `uv run coverage run -m pytest`: 782 passed, 1 skipped.
- Coverage is 98%.
- Exhaustive permutation composition, dagger and compact-rotation laws passed
  through arity 5; tensor laws passed through arity 3 on each side.
- Current and earlier PR-era symmetric box and diagram JSON/pickle
  representations normalise to permutation-valued layer slots.

## Review follow-up (2026-07-24)

Instruction from Alexis, verbatim:

> added some review on symmetric.Layer, you haven't done a much better job
> than Claude I must say

- [x] @codex-2026-07-24T20:02+0530 Address every unresolved actionable
      review thread by simplifying `symmetric.Layer`, removing redundant
      hierarchy-specific factories and private validation machinery, and
      rerunning the full verification suite.

The review supersedes the earlier constructor canonicalisation and migration
work: `Diagram` no longer has a custom constructor or state hook, `Layer` has
no private conversion helpers, and sequential permutations are compared
semantically with `Equation`. The permutation factory for a generator layer is
derived from the generator's category, so Markov no longer defines a redundant
`Layer` subclass.

## Review-follow-up verification (2026-07-24, @codex)

- `pflake8 discopy` is clean.
- Full non-notebook suite: 772 passed, 1 skipped; coverage is 98%.
- The seven configured notebooks were deselected because the app sandbox
  forbids the local sockets needed to start their kernels.

## Structural review follow-up (2026-07-25)

Instruction from Alexis, verbatim:

> Go through the discopy PRs you own and follow the agents/EVENING.md prompt i.e. go through the reviews and implement them

A symmetric layer presents
`s_0 @ f_1 @ s_1 @ ... @ f_n @ s_n`, where each `f_i` is a
non-permutation generator (including `Swap`) and each `s_i` is either a type,
representing identity routing, or a non-identity native `Permutation`.
Forgetting the structural distinction expands every non-identity `s_i` to an
ordinary box between empty types and coalesces each identity `s_i` into its
adjacent type slot. Thus every actual `Permutation` satisfies the ordinary
`Box` invariant, and `boxes`, `offsets`, encoding and box-indexed rewrites use
one ordinary boxes-and-types view. Category-specific permutation ownership
comes from the generator or permutation itself; only categories which add
layer behaviour, such as compact rotation, define a `Layer` subclass.

- [WIP] @evening-2026-07-25T11:15+0200 Implement the new structural review: store identity routing in `Layer`
      rather than invalid identity boxes; expose native permutations as real
      boxes with offsets to `normalize`, `interchange` and `substitute`;
      collapse structural checks to `Layer.is_structural`; simplify factory
      selection without hierarchy-wide `Layer` subclasses; add the
      unequal-arity offset regression and concise docs; run focused tests,
      `pflake8`, and the full coverage suite.
