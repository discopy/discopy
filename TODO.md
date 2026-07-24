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
