# TODO

Prompted by a 🌙 Evening run, whose stored prompt is quoted verbatim:

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Tonight's head is [#778](https://github.com/discopy/discopy/pull/778), which is `dirty`
against `main` and so has no `refs/pull/778/merge` for a `pull_request` job to check out:
its second sign-off condition has been unsatisfiable, not stale. The resolution is a
commit, and the harness pins this session to this branch — so it is authored here and
offered to #778 as a fast-forward, the way
[#800](https://github.com/discopy/discopy/pull/800) was offered to #744.

- [ ] merge `main` into #778's head here and resolve the `CHANGELOG.md` collision
- [ ] pin both `proptest.yml` download steps on the commit `main` pins, where the head
      carries `v11` twice and `main` has moved to `v26`
- [ ] validate the resolved tree: `pflake8` over `discopy`, `.github` and `proptest`,
      `pylint discopy` against `main`'s `fail-under`, the test suite, the property matrix,
      `actionlint` and the `.github` tests
- [ ] delete this file, open the pull request into `claude/youthful-bell-uopbrf`, take it
      out of draft by hand and say on #778 what the one click does
