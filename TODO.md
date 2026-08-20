# TODO

> no this PR is garbage let's close it and focus on https://github.com/discopy/discopy/issues/599

— @toumix closing [#601](https://github.com/discopy/discopy/pull/601), 2026-08-20

The docs-only answer to [#599](https://github.com/discopy/discopy/issues/599) is
rejected. This branch fixes the issue instead.

- [x] Revert the `Layer.id` docstring note of #601.
- [x] Reject a boxless layer in `Diagram.__init__` (`ValueError`), so the junk
      of #599 is unrepresentable rather than documented. This is option 2 of the
      issue, the one it calls "closest to the stated invariant, and it would
      have caught this at construction".
- [x] Gate the check on `_scan`, the flag that already separates user input
      from internal construction, so no hot path pays for it.
- [x] Confirm nothing internal produces a boxless layer: only `Layer.id` builds
      one, as the unit of `Layer.tensor`, which `Layer.normalise` merges into
      the box it whiskers. Full suite green with the check enforced.
- [x] Regression test in `test/monoidal.py` and a `CHANGELOG.md` entry.
- [x] Test that `then`, `tensor`, `normal_form`, `foliation` and `interchange`
      never emit a boxless layer, since they construct with `_scan=False` and
      so are not covered by the constructor check (@daydream6728 on #603).
- [ ] Confirm the choice of fix with @toumix: option 2 here, versus option 1
      (drop empty layers on composition) as @daydream6728 proposed on #601.
