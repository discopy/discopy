# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

[#795](https://github.com/discopy/discopy/pull/795) fixes the copy map of the
`Matrix` comonoid — `Matrix.copy(3, 2)` comes out with an empty middle row on
`main` and is not a function at all — and it has been `dirty` since
`2026-10-09`, nine commits behind `main` `afd24e1b`. A conflicted head has no
`refs/pull/795/merge`, so no `pull_request` job can run on it: its sign-off
condition 2 is **unsatisfiable rather than stale**, and that is a check that
needs the current base, which is `EVENING.md`'s own reason to merge.

The pin forbids pushing to #795's branch, so the resolution is authored here
and offered as a pull request into it, the fourth after
[#800](https://github.com/discopy/discopy/pull/800),
[#801](https://github.com/discopy/discopy/pull/801) and
[#807](https://github.com/discopy/discopy/pull/807) (desire#46).

- [ ] Resolve `origin/main` into #795 — `CHANGELOG.md` alone, measured
- [ ] Resolve it as a union of the two sides, additively, keeping the blank
      line between the entries that a marker-only deletion eats
- [ ] Grep the whole tree for conflict markers
- [ ] Validate the merged tree against `main`'s own numbers in this venv:
      `pflake8`, `pylint` against `fail-under`, `pytest`, and the doctests of
      the four `Matrix` classmethods this head documents
- [ ] Re-measure the three `MarkovCategory` copy laws on the merged tree, not
      only on the head as it stood: the claim is about the tree anybody merges
- [ ] Offer the resolution as a pull request into `claude/youthful-bell-qpanvx`,
      opened as a draft and undrafted by hand
