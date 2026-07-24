# TODO — refactoring of PR #362 (Add symmetric.Layer)

Instruction from Alexis (@toumix), verbatim:

> There was some weird stuff happening in the tensor of layers which didn't
> make sense and overall the PR deserves some thorough refactoring (eg it was
> written before the new contributing guidelines and agents.md) please give it
> a go.

## Checklist

- [WIP] @bridge-2026-07-22 Investigate the tensor-of-layers semantics: map every call path into
      `Permutation.tensor` (`abc.whisker`, `Layer.__matmul__`/`__rmatmul__`,
      direct calls) and pin down which branches are live.
- [ ] Refactor `Permutation.then`/`tensor`/`dagger`/`__rmatmul__` so they all
      flow through the single chokepoint `Diagram.from_permutation`, with no
      case explosion; make the reason `@unbiased` does not fit visible in the
      structure (results leave the `Permutation` subtype, so the tail of the
      arguments is delegated to the result's own method).
- [ ] Refactor `symmetric.Layer`: `__init__` (compact, validated), `cast`,
      drop `dagger` (subsumed by `monoidal.Layer` + `cat.Ob.dagger`), guard
      `merge` so `foliation()` of permutation layers stops crashing.
- [ ] Move misplaced logic to `monoidal.Layer`: uniform `dagger`, fix the
      odd-slot validation bug in `__init__`, fix `free_symbols`/`subs` on
      foliated (5+ slot) layers.
- [ ] Fix `symmetric.Functor.__call__` on `Permutation` with a
      length-changing ob map (currently a bare `ValueError`).
- [ ] Style-guide sweep of the whole diff: no code comments, docstrings with
      doctests, short names, `eval(repr(x)) == x`, drawing backend comments.
- [ ] Update docs and tests: module docstring, doctests, regression tests for
      foliation, whiskering, functors, `then()`/`tensor()` with no arguments.
- [ ] Run `uv run pflake8 discopy` and `uv run coverage run -m pytest`, fix
      anything broken, record pre-existing failures.

## Deliberately left out (follow-ups agreed in review)

- `Swap` as a subclass of `Permutation` (own issue).
- Uniform storage of even slots ("everything is a permutation") and the
  relaxation of the alternating-list `Layer` representation (#437).
