# Property-based testing

## Current prompt

> don't push on the cmap branch, focus on proptesting. implement your changes not in fable bug investigation branch, moving bug reproductions from the test/fable/*.py files to test/test_properties.py, then once every relevant bug was classified and implemented as part of the general test suite, upstream the changes in the principal development branch for property testing

> feel free to implement new axioms in the missing categories

- [ ] Fix the property suite and integrate the classified general properties on this branch.

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

Cubic reviewed the branch on 2026-08-16 and raised ten findings. All are now settled below: eight
were fixed and two were false positives.

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

      `test/abc.py` passes on this head. Had the claim been right the suite would have been red for
      eight weeks.
- [x] `all_axioms` binding each axiom twice was cosmetic and true; the generator now yields the
      already-bound axiom.
- [x] **The pivotal finding is a false positive.** `normal_form()` is called for its exception: it
      rejects planar diagrams that are not boundary-connected, while conversion intentionally
      preserves the original presentation. Two differently nested disconnected circles convert
      directly to the same hypergraph, and a regression test now documents that the public method
      rejects this unfaithful conversion.
- [x] `FeedbackCategory.feedback(self, dom, cod, mem)` declared all three arguments required even
      though both implementations and axioms infer them. The abstract signature now gives all three
      `None` defaults, matching `feedback.Diagram` and `stream.Stream`.
- [x] `Arrow.strategy` now applies exact boundaries to recursively generated paths, preserving
      `min_leaves` and `max_leaves`; a regression test finds a constrained composite.
- [x] `Layer.strategy` now filters candidate boxes through `exclude`, and `symmetric.Layer` forwards
      the set to its base distribution. Testing this also exposed and fixed its stale `Layer.cast`
      call, which broke unconstrained generation after `Layer.cast` was removed.
- [x] `CMap.strategy` now converts with `cls.from_diagram`, and a custom-subclass regression test
      checks the sampled value's exact type.
- [x] The global `"def strategy"` coverage exclusion is gone. Concrete strategy smoke tests keep the
      default suite at 98%; only the abstract `Strategy.strategy` declaration has a local pragma.
- [x] The tautological CMap axiom-name comparison now evaluates unitality and associativity on
      concrete maps; the property matrix continues to check every declared map axiom.
- [x] `test_extend_strategy` now finds both a free box and a twist from the extended distribution,
      while retaining its boundary-guard assertion.
- [x] Merged `main` (`5fa95f6e`), keeping both `CMap.from_generator` and the new
      `CMap.from_glued`, then took the simplifications the merged changes allow: the
      `axiom_status` entries that `Category.axiom_equality` already resolves through the MRO
      are gone, and `benchmark.generators.single_layer_tensor` builds its layer from boxes
      alone now that `monoidal.Layer` normalises its own plumbing.

## Open question for USER, not blocking

The `check_*` methods land in `discopy/abc.py`, so every category class in the library gains ~15
test-only methods on its public surface. Defensible under `STYLE.md`'s "we expose the interface of
every subprocedure as methods that can be tested and reused", and it is the one place where test
scaffolding becomes API — raised on the PR 2026-07-25 and never answered. The alternative is
Hypothesis strategies living in `proptest/` only. Worth a yes or no before this merges, since it is
much cheaper to move now than after release.
