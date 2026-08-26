# TODO

> Does the style review even read notebooks? let's fix that

Posted by toumix as a comment on [#626](https://github.com/discopy/discopy/pull/626).
Confirmed: `style-review.yml` diffs `-- '*.py'` only, so a PR touching only
`docs/notebooks/*.md` (a marimo notebook) never has a non-empty diff — the
review step is skipped and the correctness reviewer is called with no style
pass at all, silently.

- [x] Extend the diff in `style-review.yml` to include `docs/notebooks/*.md`
- [x] Make `review.py`'s file-content wrapping and docstrings file-type
      aware rather than assuming every changed file is Python
- [x] Update `prompt.md` to describe notebooks as a second kind of changed file
- [x] `CHANGELOG.md` entry
- [x] Smoke-test `review.py`'s `assemble()` against a real notebook diff
      locally (no test harness covers `.github/style-review` today)
