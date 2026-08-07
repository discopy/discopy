# Fix and test non-linear closed terms

> oh oh, clearly we need some better testing for non-linear terms! please open a PR for this bug
> before we tackle the actual round-trip

"This bug" is #541. Measuring around it turned up three more in the same two methods, all of them
reachable from ordinary terms and none of them covered by a test. `test/closed.py` had four tests
and none built a non-linear term.

## The four bugs

- **#542** — `closed.Application.__check_dom__` ends with `self.ob.tensor(...)`, an unbound call on
  the class. With at least one free variable the first type binds to `self` and the answer is right
  by accident; with none it raises, so `f(x)` for two constants does not build.
- **#543** — the same method takes `list(set(...))` on the overlap branch, so free-variable order,
  and hence `dom`, depends on `PYTHONHASHSEED`. The overlap branch *is* the non-linear branch.
- **#541** — `closed.Abstraction.eval` calls `.index()` unconditionally, so abstracting a variable
  absent from the body raises instead of discarding it.
- **#544** — the permutation in the same method is the identity whenever the abstracted variable is
  not already first, so nested abstractions curry the wrong wire and `eval` loses `dom` and `cod`.
  Not a non-linear bug: `A(lambda a: B(lambda b: g(a)(b)))` is linear and hits it.

#544 was found last and is the largest — `eval` is meant to be a functor, and it is not well-typed
on any term with two nested abstractions. It is fixed here because the fix is one line and the
other three cannot be tested end-to-end while it is broken.

## Points

- [x] 1. `self.ob().tensor(...)` so an application with no free variables builds (#542).
- [x] 2. First-occurrence order via `dict.fromkeys` instead of `set`, unifying the overlap and
      linear branches — they agree on every term where both applied (#543).
- [x] 3. Discard the abstracted variable when it does not occur in the body (#541).
- [x] 4. `[i] + [j for j in range(n) if j != i]` as the permutation, so `curry` binds the abstracted
      wire (#544).
- [x] 5. Tests: application without free variables, free-variable ordering, abstraction of an unused
      variable, `dom`/`cod` preservation across nested/unused/copied terms, and copy/discard
      appearing in the evaluated diagram.
- [x] 6. `CHANGELOG.md` entry under `[Unreleased]`.
- [x] 7. `uv run pflake8 discopy` and `uv run coverage run -m pytest` green.

## Not in scope

The term-to-diagram roundtrip itself, which is #540 and waits on a naming ruling. This branch is
the prerequisite: #541 and #544 both block the roundtrip, the first because the discard case cannot
be round-tripped while it raises, the second because a roundtrip cannot be stated at all while
`eval` does not preserve `cod`.
