# TODO — refactoring of PR #362 (Add symmetric.Layer)

Instruction from Alexis (@toumix), verbatim:

> There was some weird stuff happening in the tensor of layers which didn't
> make sense and overall the PR deserves some thorough refactoring (eg it was
> written before the new contributing guidelines and agents.md) please give it
> a go.

## Checklist

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
- [WIP] @bridge-2026-07-23 05:54 Fix coverage gate: delete dead braid-shadow
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
