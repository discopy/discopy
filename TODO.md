# TODO

Prompt, from USER on [#491](https://github.com/discopy/discopy/issues/491), verbatim:

> @toumix-agents open a draft PR for this one

on the issue *Closed diagrams containing Copy or Swap cannot be drawn*.

## Points

- [x] Reproduce on `main`
- [x] Guard the `markov`, `symmetric`, `braided` and `balanced` functor branches
      with `hasattr`, the pattern `biclosed.Functor` already uses
- [x] Check the drawings are right, not merely non-crashing
- [x] Regression test, verified to fail without the fix
- [x] `pflake8 discopy` clean and the suite green
- [x] `CHANGELOG.md` entry
- [x] Confirm whether this also closes [#548](https://github.com/discopy/discopy/issues/548)

## The bug

`closed.Diagram.to_drawing` routes through `closed.Functor` so that `Curry` and
`Eval` are laid out properly. That functor inherits the `markov`, `symmetric`,
`braided` and `balanced` branches, which called `self.cod.copy`,
`self.cod.merge`, `self.cod.ar.swap`, `self.cod.ar.permutation`,
`self.cod.braid` and `self.cod.twist` unconditionally — and `Drawing` has none
of them. Plain markov and symmetric diagrams never noticed, because their
`to_drawing` uses `monoidal.Functor` and lays a `Copy` out as an ordinary box.

Each branch now checks the codomain has the structure before using it, exactly
as `biclosed.Functor` already does for `over`, `under`, `exp`, `curry` and `ev`.
Falling through to `super().__call__` draws `Copy` as its spider and `Swap` as
crossing wires, which is what the markov and symmetric diagrams get today.

`Permutation` was guarded alongside `Swap`: it sits in the same branch, has the
same codomain requirement, and would have been the next `AttributeError`.

## It closes #548 too

[#548](https://github.com/discopy/discopy/issues/548) is the same bug found from
the other end, with the same repro and a fuller diagnosis of why
`closed.Diagram.to_drawing`'s override is what drags in the markov assumption.
Both are closed by this PR.

## Not changed

No drawing baseline moves: nothing that used to draw draws differently, the
guards only add cases that used to raise.
