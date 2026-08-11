# TODO

Prompt ([#437](https://github.com/discopy/discopy/issues/437), verbatim):

> Instead of lists of odd length of length 3 minimum alternating with type, box, type, etc. we want the following way of defining layers: a  monoidal layer holds a list of boxes and non-empty types with at least one box and no two consecutive types.
>
> Whiskering a layer with a type on the left (right) only appends to the list if the type is non-empty and the left-most (right-most) element of the layer's internal list is a box.
>
> Tensoring a layer that ends with a type with a layer that begins with a type should tensor them so that the resulting layer has the length given by the sum of the lengths of the two layers minus 1.
>
> Initialising a new layer should scan=True by default to go through the list and tensor consecutive types to enforce the invariant, but when the layer is constructed by one of the methods above we use scan=False because we know we are preserving the invariant, so tensor a list of n layers takes linear time rather than quadratic.
>
> The same logic applies to #362 by replacing "type" with "permutation" in the discussion above: a symmetric layer holds a list of boxes and non-empty permutations with no two consecutive permutations with the following condition either a) it has at least one box or b) it is a singleton list of a non-identity permutation.

---

- [x] Redefine `monoidal.Layer` on the new representation: boxes and non-empty types, at least one box, no two consecutive types
- [x] Whiskering appends the type only when it is non-empty and the outermost element is a box, otherwise merges it into the boundary type
- [x] Tensoring layers merges a trailing type with a leading type (resulting length = sum of lengths − 1)
- [x] Constructor defaults to `scan=True` (rescans the list, merging consecutive types to restore the invariant); every internal call site that already preserves the invariant constructs with `scan=False`, so tensoring `n` layers is linear rather than quadratic
- [x] `symmetric.Layer` (#362): the same invariant with "permutation" in place of "type", except it may also be a singleton list holding one non-identity permutation and no boxes
- [x] Adjust `dom`/`cod`/`name` computation and `boxes_and_offsets` to the new representation
- [x] Sweep dependent code (drawing, foliation, `symmetric.Layer` from #362) and update doctests + README
- [x] Run `pflake8 discopy` and `coverage run -m pytest`

## Guidance (🐦 birdsong, 2026-07-22)

- wait for #362 (symmetric-layer refactor, branch `claude/discopy-main-work-xu4vkj`) to land
  first. its own TODO defers this exact representation change to here, and
  `symmetric.Layer` subclasses `monoidal.Layer` — start now, rebase twice.
- #362 just fixed a `Layer.merge` crash on permutation layers. keep it fixed, add a
  regression test so this change can't reintroduce it.
- `boxes_and_offsets` feeds the drawing backend directly. no behaviour change for
  diagrams that already round-trip, or every drawing test breaks.
- eval(repr(x)) == x still has to hold on the new representation.

## Guidance (🌤️ daylight, 2026-07-22)

- #362 landed its refactor as of today, all points [x], undrafted, just waiting
  on your merge — so "wait for #362" now means wait for the merge, not the work.
  one rebase should do it.
- #444 (Swap ⊂ Permutation) got deferred out of #362, separate issue, doesn't
  block you.

## Guidance (🐦 birdsong, 2026-07-23)

- Alexis edited issue #437 (2026-07-22 16:00Z) to add the `scan=True`/`scan=False`
  linear-time requirement and the precise `symmetric.Layer` invariant (case a/b) —
  both added as their own checklist points above, and the verbatim prompt refreshed
  to match. Per `RULES.md` rule 1 this refresh is authorized by his direct
  instruction in today's session ("make sure the todos reflect any changes to the
  issues").

## Review follow-up (2026-07-25)

Instruction from Alexis, verbatim:

> Go through the discopy PRs you own and follow the agents/EVENING.md prompt i.e. go through the reviews and implement them

The branch should be current `main` plus this trusted checklist. Generated
documentation assets remain inherited from `main` and are left to docs-static.

- [x] Restart the branch from current
  `main` while preserving this checklist.

Verification: restored all generated assets, then merged current `main`
append-only per `RULES.md`; the PR diff is now only `TODO.md`.
`pflake8 discopy` passes. Non-optional tests: 319 passed; 4 require missing
SymPy/Torch dependencies. Full collection is blocked by the same optional
quantum and tensor dependencies.

## Sign-off follow-up (🌙 evening, 2026-08-01)

#362 merged on 2026-07-29, which was the last thing this branch waited on, and
`CHANGELOG.md` landed after this checklist was written (#487): `AGENTS.md` now
asks for an `[Unreleased]` entry on every user-facing change, and the new
`Layer` representation is one.

- [x] Merge current `main` and add the `CHANGELOG.md` entry for the new
  `Layer` representation

Verification: merged `main` (`e80ea38`) append-only, no conflicts.
`pflake8 discopy` clean; `pytest --skip-extra` gives 623 passed, 51 skipped.
Every point above is `[x]`, so this is ready for sign-off as soon as a human
deletes `TODO.md`.

## Review follow-up (2026-08-06)

Instruction from Alexis, verbatim:

> Simplify layers https://github.com/discopy/discopy/pull/438
> I added some comments

- [x] Address all five unresolved review comments: define plumbing in the
  documentation, expose the layer normalisation steps, and reuse them during
  deserialisation.

Verification: `pflake8 discopy` is clean and the 101 focused monoidal,
symmetric, compact and Markov tests pass. GitHub build run `31132904791`
passed docs, lint and the full test suite on Python 3.12, 3.13 and 3.14.
The branch includes current `main` at `ed4c0b3d`.

## Review follow-up (2026-08-07, second round)

Instruction from Alexis, verbatim:

> simplify layers https://github.com/discopy/discopy/pull/438

The nine unresolved review comments from the 10:34–10:51 UTC round, plus the
08:38 follow-up on the resolved thread asking to rename `scan`:

- [x] Rename `scan` to
  `normalise` and give `__init__` no pass over `inside` when it is `False`:
  `name`, `dom` and `cod` become lazy, the box-presence check moves to the
  `normalise=True` path, and the flag's docstring explains the type checking
  and the quadratic blowup that skipping it avoids
- [x] Make the type
  checking of `check` explicit in its loop instead of tensoring for the side
  effect, same for the empty-type cases of `__matmul__`/`__rmatmul__`
- [x] Flatten
  `normalise` to two cases: append or tensor with the last element
- [x] Drop
  `Layer.tensor`, keeping only `__matmul__` and `__rmatmul__`, and update
  the tests that used it
- [x] Answer the
  `boxes_and_types` compatibility question with a measured estimate, filed
  as an issue

Verification: `pflake8 discopy` is clean and `pytest --skip-extra` gives
627 passed, 51 skipped. Chaining 4000 layers with `@` takes 0.47s against
122s before this round (the old `@` re-tensored types and names at every
step); the compatibility question is answered on the thread and filed as
[#547](https://github.com/discopy/discopy/issues/547).

## Review follow-up (2026-08-07, third round — recorded 🐦 birdsong 2026-08-11)

The 15:34–16:18 UTC round landed **after** the last work commit (`5615254`, 12:56 UTC) and no
turn picked it up: four days, four trusted instructions, none applied. Recorded here unclaimed.

Alexis, replying to daydream6728 on `Layer.cast`, verbatim:

> ha yes good point!

on daydream6728's *"If this is expected to be true for all subclasses of monoidal.Layer, then we
can probably remove the cast method altogether"*.

Alexis, on the `len(...) != 3` guard in `interchange`, verbatim:

> yes that's a mistake indeed!

on daydream6728's *"Can't we have less than 3 items if e.g. a box has no type to its left
`Layer(Ty(), box, right)`? From what i understand, its normal form will be `Layer(box, right)`
since the odd constraint disappeared."*

Alexis, on deferring `boxes_and_types` to [#547](https://github.com/discopy/discopy/issues/547),
verbatim:

> not sure
> let's get rid of it and simplify the 8 methods

Alexis 🚀'd daydream6728's comment (2026-08-07T16:15:34Z), verbatim:

> `Layer` could inherit `ColouredMonoid` and get that for free

- [ ] Remove `Layer.cast`, since every subclass of `monoidal.Layer` satisfies what it casts for
- [ ] Fix the `any(len(layer.boxes_and_types) != 3 ...)` guard in `Diagram.interchange`: a layer
  whose box has empty plumbing on one side normalises to two components, so the check rejects
  diagrams it should accept
- [x] ~~Drop `Layer.boxes_and_types` and port the eight call sites in six modules to the new
  representation, i.e. bring #547 into this pull request and close it with this one~~
  **Cancelled by Alexis on 2026-08-11**, see below: it stays in
  [#547](https://github.com/discopy/discopy/issues/547) and out of this pull request
- [ ] Let `Layer` inherit `ColouredMonoid` rather than restate what it provides

Note for whoever claims these: this roughly doubles the diff of the largest pull request in the
queue, so the third point is worth splitting off if the review cost is judged too high — that is a
question for Alexis, not a decision to take here.

## The third point is cancelled (🌙 evening, 2026-08-11)

Alexis answered exactly that question the same night, on
[memory#55](https://github.com/toumix/memory/pull/55), verbatim:

> nah you're right let's keep the PR small and delay the simplify layers refactoring of previous
> algorithms

Confirmed by Alexis an hour later, unprompted and naming the issue, verbatim:

> yes i meant you're right about this issue, we should postpone to a later PR
> https://github.com/discopy/discopy/issues/547

This reverses the *"let's get rid of it and simplify the 8 methods"* quoted above and reinstates
the agent's original answer on the thread — port the eight call sites in
[#547](https://github.com/discopy/discopy/issues/547), not here. "you're right" is agreeing with
that measurement, and "the simplify layers refactoring of previous algorithms" is his own phrase
from the question that opened the thread, *"refactor the algorithms that depend on the old
representation"*.

So `Layer.boxes_and_types` **stays** in this pull request. The other three points above are
untouched by this and remain open.
