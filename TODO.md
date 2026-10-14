# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

`main` took [#489](https://github.com/discopy/discopy/pull/489) at
`2026-10-13T07:55:35Z` and [#627](https://github.com/discopy/discopy/pull/627)
at `07:28:48Z`. That turned eleven of our `main`-based heads `dirty`, and
[#761](https://github.com/discopy/discopy/pull/761) is the only one whose
conflict is in library code: #489 landed the downstream `tensor` patch that
#761 exists to replace upstream, on @daydream6728's own reading of it in
https://github.com/discopy/discopy/pull/489#discussion_r3896050565.

The pin forbids pushing to #761's branch, so the resolution is authored here
and offered as a pull request into it (desire#46).

- [ ] Resolve `origin/main` into #761 — `discopy/python/multiplicative.py`
- [ ] Decide what becomes of `test_tensor_unit_is_none`, which #489 landed to
      pin the convention #761 removes, and which auto-merged silently
- [ ] Grep the whole tree for conflict markers, the suite reading no `CHANGELOG.md`
- [ ] Validate the merged tree: `pflake8`, `pylint` against `main`'s own score
      in this venv, `pytest`, and the doctest of every method touched
- [ ] Offer the resolution as a pull request into `claude/amazing-volta-ia1t4e`,
      opened as a draft and undrafted by hand
- [ ] Re-measure #766's `pylint` clearance: `fail-under` is **8.77** since #489,
      not the 8.75 its 8.76 was read against
