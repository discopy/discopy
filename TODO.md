# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

The 🌙 Evening turn of 2026-09-29 picked the standing `proptest` reporting proposal, nineteen
mornings on `USER_TODO.md` with no reaction, and argued for on three separate nights:

> report a cell that *was* passing and is now skipped, and one that passes because its strategy
> cannot draw the failing case. On #665 `serialisation` was green while `dumps(Scalar(1j))` raised,
> because the distribution held only `Scalar(0.5)`. **A cell that cannot fail is indistinguishable
> from a law that holds.**

Siblings: [#773](https://github.com/discopy/discopy/issues/773) — `transparency` is quantified
*over* `python.Ty` and never stated *of* it — and
[#775](https://github.com/discopy/discopy/issues/775) — `dumps` crashes on complex data and no cell
says so.

- [x] Measure the claim first: which cells of the matrix exhaust their strategy's support on `main`,
      at two budgets, rather than asserting that some do
- [x] `proptest/report.py`: the outcomes of a run, the cells that cannot fail, the cells that
      stopped being checked against a baseline, and the JSON both halves travel in — pure functions
- [x] `proptest/conftest.py`: the plugin — the two options, the fixture that counts what a cell
      drew, collection that survives `-n auto`, and the terminal summary
- [x] `proptest/test_axioms.py`: record the terms each cell actually drew
- [x] `proptest/test_report.py`: unit tests for the pure half
- [x] `proptest.yml`: carry the report between runs, so "was passing, now skipped" has a baseline
- [x] `CONTRIBUTING.md`: what the two reports mean and how to pass a baseline
- [x] `CHANGELOG.md`: an `[Unreleased]` entry
- [WIP] @01YLq8Jk-2026-09-29 01:12 Validate before pushing: `pflake8`, the matrix at `dev` and at `pr`, serially and under
      `-n auto`, and the baseline comparison against a report captured before the change
