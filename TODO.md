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
- Uniform storage of even slots ("everything is a permutation") and the
  relaxation of the alternating-list `Layer` representation (#437).

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
