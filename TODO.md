# TODO

CodeRabbit's first round on this pull request, five findings, every one labelled 🟠 Major and
none a nit. Quoted verbatim:

> **Restrict baseline selection to the intended branch.** This artifact search has no branch
> filter. A successful manual run on a feature branch can therefore supply the next `main` or
> pull-request baseline.

> **Select a successful run for the coverage baseline.** If property tests stop on a failure, the
> always-run upload can store a partial report. `workflow_conclusion: completed` can then select
> that report as the next baseline.

> **Report the case where no matrix cells ran.** If a run selects no marked cells, `CELLS` is empty
> and this return skips both baseline comparison and `--proptest-report` output. A baseline with
> previously passing cells therefore cannot identify that every cell is now gone.

> **Include `XPASSED` baseline cells in stopped-coverage checks.** An `XPASSED` cell checked its
> law, but this filter excludes it. If that cell is skipped, xfailed, or removed in the next run,
> `stopped` reports no loss.

> **Validate the baseline JSON shape before reading its entries.** If a baseline contains valid
> JSON such as `[]` or `null`, `.items()` raises `AttributeError`. `pytest_terminal_summary`
> catches only `OSError` and `ValueError`, so an unreadable baseline can interrupt the suite
> instead of producing the promised diagnostic.

All five reproduce, and two of them are this change failing its own stated principle: a section
that vanishes when it has nothing to say, and a baseline that crashes the suite the docstring
promises it cannot.

- [ ] `loads` raises `ValueError` on a top-level value that is not an object — reproduced on `[]`,
      `null`, `"x"` and `3`, all `AttributeError` today
- [ ] `stopped` counts an `XPASSED` baseline cell as checked, plus a test for that transition
- [ ] the terminal summary runs when a baseline or a report was asked for, even with no cell — and
      stays silent when neither was and there is nothing to report
- [ ] `proptest.yml` takes its baseline from `main` only, and only from a run that succeeded;
      verified against the pinned action's own `action.yml` rather than the finding's word
- [ ] Re-validate: `pflake8`, the matrix at `dev` and `pr`, serial and `-n auto`, the baseline
      round trip, and each of the five findings turned into a test or shown fixed
- [ ] Reply on all five threads and resolve them
