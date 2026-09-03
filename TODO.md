# TODO

> open a PR with the draft the .coderabbit.yaml from STYLE.md and all of our
> custom style reviewer workflow retired

- [x] Draft `.coderabbit.yaml`, translating each `STYLE.md` principle into
      per-path review instructions, style-only so it stays in its lane beside
      cubic (correctness).
- [x] Retire the in-house reviewer: remove `.github/style-review/`, the
      `style-review.yml` workflow, and its `.github/tests/` tests.
- [x] Strip `.github/tests/conftest.py` back to what the remaining
      `benchmark_comment` test needs.
- [x] Record the retirement in `CHANGELOG.md` (`### Removed`).
- [ ] Confirm CI is green: the `workflows` job (`actionlint`, `pflake8 .github`,
      `pytest .github/tests/*.py`) still passes with the files gone.
- [ ] USER decision: prune the now-moot style-review entries under `### Added`
      / `### Fixed` at release time, or leave them as cycle history? (flagged,
      not done here.)
- [ ] USER decision: keep the one-week bake-off idea (CodeRabbit alongside the
      old reviewer) or go straight to the swap, as done here?
