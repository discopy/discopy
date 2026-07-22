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

- [ ] Implement capture-avoiding substitution on `closed.Term`, preserving variable names when capture can be avoided
- [ ] Implement the `BohmTree` class as specified (`cod`, `variables`, `head`, `args`)
- [ ] Beta reduction with a redex budget returning an incomplete Böhm tree, extensible reduction strategies with leftmost-outermost as default
- [ ] Method back to standard term syntax; test that normalisation is idempotent
- [ ] Unit tests: addition, multiplication, exponentiation of Church numerals, with a "unitype" that is its own exponential
- [ ] Run `pflake8 discopy` and `coverage run -m pytest`

## Guidance (🐦 birdsong, 2026-07-22)

- `Term` already lives in `discopy/closed.py` — put `BohmTree` and the reduction code
  there too, not a new top-level module.
- low collision risk, no other in-flight PR touches `closed.py` — safe to start now,
  no need to wait on the other five drafts.
- "unitype which is its own exponential" — check `closed.Ty`'s `exp`/`__lshift__` for
  the cleanest way to build a self-exponential type before inventing a new mechanism.
- style guide: no comments explaining the substitution logic, a docstring with a
  doctest instead.
