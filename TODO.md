# Fix the free-variable order in biclosed.Application

> Follow toumix/desire/EVENING.md

No human prompt named this work. It came out of the audit
[#545](https://github.com/discopy/discopy/pull/545) left open — "the same question, *does any test
exercise this path?*, is worth asking of `biclosed.py`, which I have not audited" — taken up because
every other point in the queue is `[x]` and waiting on a human, and because `discopy/biclosed.py` is
touched by no queued PR except #511, which only rewrites the `Constant` docstring.

Closes [#550](https://github.com/discopy/discopy/issues/550).

## The bug

`Application.__check_dom__` builds `freevars` in the reverse order to the `dom` it returns two lines
below. `Abstraction.__check_dom__` indexes one against the other, so it strips the wrong end of
`dom`: `eval` is not well-typed on any term with two free variables coming from an application, and
legitimate eta-expansions are refused while illegitimate abstractions are accepted.

`eval`, `Application.constants` and `closed.Application` (which derives `dom` *from* `freevars`) all
agree with `dom`, so `freevars` is the outlier and the fix is to swap its two branches.

This is the `biclosed` analogue of [#544](https://github.com/discopy/discopy/issues/544), the same
disagreement between a free-variable list and the wires it indexes, one level down in `closed`.

## Points

- [x] 1. Swap the two `freevars` branches in `biclosed.Application.__check_dom__` so they match
      `dom` and `constants`.
- [x] 2. `test_Term_linear_planar` asserted that the two eta-expansions `Abstraction(var, fvar(var))`
      and `Abstraction(var, var(gvar, left=True), left=True)` raise. They are planar and well-typed,
      so that assertion encoded the bug — replaced by their `dom`/`cod`. The two genuine linearity
      and planarity cases above it are untouched.
- [x] 3. `test_Application_freevars_order`: `dom == Ty().tensor(*[v.cod for v in freevars])` on an
      application in each handedness.
- [x] 4. `test_Abstraction_well_typed`: `eval` agrees with the term on `dom` and `cod` for a nested
      pair of binders, and the abstraction of a non-outermost variable is refused.
- [x] 5. `CHANGELOG.md` entry under `[Unreleased]`.
- [x] 6. `uv run pflake8 discopy` clean and `uv run coverage run -m pytest --skip-extra` green
      (626 passed, 51 skipped — 624 before, plus the two new tests).

## Not done here

`closed.Application.__check_dom__` keeps `func.freevars + args.freevars` in *both* handednesses. It
is self-consistent, since it derives `dom` from that list rather than from the sub-`dom`s, but it
disagrees with `biclosed` on `left=True`. Whether the two should share one convention is a design
call, not a bug, and it touches the file #442, #511 and #545 are all queued on — left alone.
