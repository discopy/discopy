# TODO — PR #415 (Add cat.Equation)

Worklist seeded by Daylight from unresolved review threads on this ALEXIS_GH-authored,
approved PR (unresolved thread = a (C)-approval per ROUTINE.md: an unedited toumix
instruction comment). Each point below quotes its triggering comment verbatim.

- [x] @daylight-2026-07-22T1400 — [discussion_r3629279806](https://github.com/discopy/discopy/pull/415#discussion_r3629279806)
  + [discussion_r3629284620](https://github.com/discopy/discopy/pull/415#discussion_r3629284620)
  (`discopy/symmetric.py:90`): "Even worse than a private method, adding a private
  class I had not seen that kind of nonsense before!" / "none of this is needed
  anymore because the up_to method of Equation never needs to be a functor it can
  be any callable e.g. the method to_hypergraph directly." — deleted the private
  `_ToHypergraph` descriptor class; `Diagram.to_hypergraph` is now a plain instance
  method, and every `up_to = Diagram.to_hypergraph` site (`symmetric.py`,
  `compact.py`, `markov.py`, `frobenius.py`, `feedback.py`) is wrapped
  `staticmethod(...)` — plain functions stored as a class attribute of an unrelated
  class rebind `self` on instance access (verified this throws `TypeError` without
  the wrapper), `staticmethod` is the standard idiom to keep it a bare 1-arg
  callable, matching "any callable" literally with no custom descriptor needed.
- [x] @daylight-2026-07-22T1400 — [discussion_r3629294798](https://github.com/discopy/discopy/pull/415#discussion_r3629294798)
  (`discopy/cat.py:1102`, suggested diff): collapsed `Equation.__bool__`'s
  if/list-comprehension into the suggested one-liner ternary + `map`; the
  suggestion's `map(self.up_to, term)` was a typo (undefined `term`, singular) —
  applied as `map(self.up_to, self.terms)`, the only reading consistent with the
  surrounding code and the variable it assigns to.

## Void / skipped (reported to Alexis, not implemented)
- [discussion_r3620232449](https://github.com/discopy/discopy/pull/415#discussion_r3620232449)
  (`README.md`, "This should be an assert Equation") is VOID per `ROUTINE.md`
  INTEGRITY — edited 1s after posting (07:29:25Z → 07:29:26Z). The thread's second
  comment ("we removed quotient") is unedited but reads as explanation, not an
  instruction on its own — skipped pending a fresh comment.
- [discussion_r3620219296](https://github.com/discopy/discopy/pull/415#discussion_r3620219296)
  (`discopy/messages.py`, "This is overkill it's only ever used once let's move it
  back to the drawing module") — already done, one minute after the comment, by
  commit `95f8514` ("Inline drawing.Equation deprecation string at the warning
  site"); the thread is just stale (marked `is_outdated`). Resolved on GitHub with
  a pointer to that commit, no new code needed.

## Run `uv run pflake8 discopy` and `uv run coverage run -m pytest`
- [x] pflake8 clean; doctests for `cat`/`monoidal`/`symmetric`/`compact`/`markov`/
  `frobenius`/`feedback`/`hypergraph` and all of `test/syntax/` green; full suite
  (excluding quantum/tensor optional deps unavailable in this sandbox — sympy/jax/
  pytket/tensornetwork, same 51 pre-existing failures with or without this change,
  verified via `git stash`) unchanged.
