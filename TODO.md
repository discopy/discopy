# TODO

Prompt ([#373](https://github.com/discopy/discopy/issues/373), verbatim):

> As a baseline for more fancy evaluators we want an implementation of syntactic beta reduction with capture avoiding substitution. The result of the normal form should be as a Böhm tree stored in the following class:
>
> ```python
> class BohmTree:
>   cod: Ty
>   variables: tuple[Variable, ...]
>   head: int
>   args: tuple[Optional[BohmTree], ...]
> ```
>
> where `0 <= head < len(variables)` is the index of the head variable of type `variables[head].cod == XS[0] >> ... >> X[k]` for `len(args) == k - 1`, `X[i] == args[i].cod` and `X[k] == cod`.
>
> - It should be possible to give a number of beta redexes as budget and return an incomplete Bohm tree (hence the `Optional`).
> - It should be easy to extend to different kinds of reduction strategies, with leftmost-outermost as the default.
> - It should come with a method for going back to the standard term syntax, to validate that e.g. normalisation is idempotent.
> - It should come with unit tests for standard constructions e.g. addition, multiplication, exponentiation of Church numerals. Since the latter is not simply typed we can add a "unitype" which is its own exponential.
> - It should preserve names of variables as much as possible, i.e. when it can avoid capture.

---

- [x] Implement capture-avoiding substitution on `closed.Term`, preserving variable names when capture can be avoided
- [x] Implement the `BohmTree` class as specified (`cod`, `variables`, `head`, `args`)
- [x] Beta reduction with a redex budget returning an incomplete Böhm tree, extensible reduction strategies with leftmost-outermost as default
- [x] Method back to standard term syntax; test that normalisation is idempotent
- [x] Unit tests: addition, multiplication, exponentiation of Church numerals, with a "unitype" that is its own exponential
- [x] Run `pflake8 discopy` and `coverage run -m pytest`
- [x] Replace `args: tuple[Optional[BohmTree], ...]` with lazy components via a `@cached __getitem__(key) -> BohmTree` (per Alexis's #373 comment, 2026-07-22) — done in 130935c: `BohmTree` now stores `strategy`+`spine` instead of `args`; `tree[i]` reduces `spine[i]` on first access via the shared `strategy` (same mutable budget threaded through the whole tree) and caches it, raising `ValueError` if budget is exhausted rather than returning `None`. `Strategy.arguments` (which used to eagerly compute results) became `Strategy.order` (returns the visiting order as indices) so that code forcing a whole tree (`BohmTree.to_term`) still respects a custom strategy's budget allocation — e.g. `RightmostFirst` still gets its intended argument reduced first even though nothing is eager anymore. Tests in `test/syntax/closed.py` (`test_reduce_budget`, `test_reduce_strategy`) rewritten to index/force explicitly instead of comparing pre-populated tuples; docstrings/doctests in `closed.py` updated to match. Full verification: `pflake8 discopy` clean, `coverage run -m pytest test/syntax/closed.py --doctest-modules discopy/closed.py` 18 passed, full suite 533 passed (all failures pre-existing environment-blocked quantum/tensor deps, unrelated). One design note for your review at sign-off: `BohmTree` equality/repr now include `strategy` by value, so two trees over the same spine but with different remaining budget are not `==` — this is intentional (budget is now part of a node's identity, not invisible) but worth a look.

## Guidance (🐦 birdsong, 2026-07-22)

- `Term` already lives in `discopy/closed.py` — put `BohmTree` and the reduction code
  there too, not a new top-level module.
- low collision risk, no other in-flight PR touches `closed.py` — safe to start now,
  no need to wait on the other five drafts.
- "unitype which is its own exponential" — check `closed.Ty`'s `exp`/`__lshift__` for
  the cleanest way to build a self-exponential type before inventing a new mechanism.
- style guide: no comments explaining the substitution logic, a docstring with a
  doctest instead.
