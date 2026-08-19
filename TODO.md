# TODO

> analyze this PR which replaces absolute benchmarking against a baseline by
> relative benchmarking that's only comparing a PR's base and head commits.
> Look into the previous benchmarking PR and analyze what can be removed from
> the benchmarking suite

> make the cuts, one per commt

- [x] Render the pull request comparison once, in `report.py`, and let the
      comment workflow post it instead of reimplementing it in JavaScript
- [x] Drop the HTML and CSV renderings of the scaling table
- [x] Derive the plot panels from the data instead of the hardcoded `PLOTS`
- [ ] Drop the `series` and `tensor` aliases for `repeated`
- [ ] Drop the `BENCH_FLAGS=bench:smoke` no-op
