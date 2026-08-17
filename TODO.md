# Property-based testing

No verbatim prompt: this PR was opened 2026-06-22, before `RULES.md` rule 1 existed, and its
description is a generated summary rather than anything a human typed. It gets a `TODO.md` now
because USER ruled on [desire#76](https://github.com/toumix/desire/issues/76), verbatim:

> adoption adds a todo

That ruling is what unsticks this branch. `no-todo-on-main` marks a PR ready only when a push
*deletes* a `TODO.md`, so a branch that never had one can never leave draft, however green it is.
This file restores the normal gate: USER deletes it, the guard marks the PR ready.

## State

All three sign-off criteria otherwise hold, re-checked 2026-08-17: 0 behind `main` (`48eea53`),
CI green, no thread waiting on an agent. The Hypothesis matrix lives in `proptest/`, outside
pytest's `testpaths`, and runs on `main`, on manual dispatch, and on this PR via its `proptest`
label.

## Points

Cubic reviewed the branch on 2026-08-16 and raised ten findings. Two are settled below; the other
eight are the remaining work, none of them blocking.

- [x] **The P1 is a false positive.** Cubic reads `Arguments.bifunctoriality` and friends as
      returning an unpacked N-tuple that `axiom(*arguments)` would splat onto a single structured
      parameter, raising `TypeError: too many positional arguments`. They do not: every one of them
      ends on a trailing comma, so they return a **1-tuple wrapping** the N-tuple and the splat
      passes exactly one argument.

      ```python
      >>> args = Arguments.bifunctoriality(monoidal.Diagram)
      >>> len(args), len(args[0])
      (1, 4)
      ```

      `test/abc.py` is 272 passed, 32 skipped, 16 xfailed on this head. Had the claim been right
      the suite would have been red for eight weeks.
- [x] `all_axioms` binding each axiom twice is cosmetic and true; folded in with the next edit to
      that generator rather than as a commit of its own.
- [ ] `pivotal.Diagram.to_hypergraph` calls `self.normal_form()` and **throws the result away**,
      then converts the original `self`. Diagram methods are pure by `STYLE.md`, so the line is
      dead. Either rebind to the normal form or drop the call — deciding which is the point, since
      the two differ for a non-normal input.
- [ ] `FeedbackCategory.feedback(self, dom, cod, mem)` declares all three required, but
      `feedback_joining` calls `f.feedback(mem=mem)` and then `f.feedback()` with no arguments at
      all. Cubic flagged `feedback_joining`; `feedback_vanishing` on the line above has the same
      shape and was missed. Either the abstract signature gives `dom`/`cod` defaults or both axioms
      pass them.
- [ ] `Arrow.strategy` returns a single generator and skips recursive composition whenever `dom` or
      `cod` is given, so every boundary-constrained property test runs on generators only. Build
      that branch recursively, respecting `min_leaves`/`max_leaves`.
- [ ] `Layer.strategy` drops the `exclude` set that `Diagram.strategy` threads into it, so the
      no-reuse contract is inert. Apply it when building candidate boxes, or delete the parameter.
- [ ] `CMap.strategy` converts via `diagram.to_map()`, which yields the category's default map type
      rather than `cls`, so a subclass is never actually exercised. Use `cls.from_diagram`.
- [ ] `pyproject.toml` excludes `"def strategy"` from coverage globally. The strategies are runtime
      code and this is the PR that adds them, so the exclusion hides the new surface from the gate
      it has to pass. Drop it and mark individual non-runtime helpers `# pragma: no cover`.
- [ ] `test/cmap.py:39` asserts `module.CMap.axioms` and `module.Diagram.axioms` enumerate the same
      names, which holds by construction — `CMap.axioms` is built from `cls.category.axioms`. It
      cannot fail. Assert axiom *satisfaction* through the new property checks instead.
- [ ] `test/testing.py:84` passes `dom=`, which makes `extend_strategy` return `base` unchanged, so
      the extension it exists for is never run. Add a case without `dom`/`cod`.

## Open question for USER, not blocking

The `check_*` methods land in `discopy/abc.py`, so every category class in the library gains ~15
test-only methods on its public surface. Defensible under `STYLE.md`'s "we expose the interface of
every subprocedure as methods that can be tested and reused", and it is the one place where test
scaffolding becomes API — raised on the PR 2026-07-25 and never answered. The alternative is
Hypothesis strategies living in `proptest/` only. Worth a yes or no before this merges, since it is
much cheaper to move now than after release.
