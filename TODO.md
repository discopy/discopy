Self-initiated fix during the nightly Evening cycle, no human prompt behind it.

Found while scanning `discopy/discopy` issues: issue #549 reports that
`closed.Context.dom` calls `self.category.ob.tensor(...)` on the *class*
`Ty` rather than an instance, i.e. an unbound method call. With at least one
variable in `self.inside` this happens to bind correctly by accident (the
first type slots into `tensor`'s own `self` parameter); with zero variables
it raises `TypeError: FreeMonoid.tensor() missing 1 required positional
argument: 'self'`. This mirrors the already-fixed sibling bug #542 in
`Abstraction.__check_dom__`, which already instantiates `self.ob()` before
calling `.tensor(...)`. Verified by reading `discopy/closed.py`: today the
bug is latent, since the only two construction sites of `Context`
(`Application.eval`'s `Context(self.freevars)`, guarded by non-empty
`overlap`, and `Abstraction.eval`'s `Context([self.var] + context.inside)`)
always produce a non-empty `inside`. The fix is still worth landing so the
bug doesn't surface later, and it needs a regression test that directly
constructs `Context([])` since no current code path reaches it.

- [x] Fix `Context.dom` to call `self.category.ob().tensor(...)` on an
      instance, matching the `Abstraction.__check_dom__` pattern.
- [x] Add a regression test in `test/closed.py` constructing `Context([])`
      and asserting `.dom == Ty()`, plus a non-empty case confirming the
      existing (accidentally-correct) behaviour still holds.
- [x] Add a one-line `### Fixed` entry to `CHANGELOG.md`'s `[Unreleased]`
      section referencing #549.
- [x] Run `uv run pytest test/closed.py` and `uv run pflake8 discopy`,
      confirm both clean.
- [x] Manually reproduce the bug on `main` (before the fix) to quote actual
      before/after output in the PR body.
- [x] Push the branch and open a draft PR against `main`, referencing #549.
