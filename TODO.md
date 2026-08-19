# TODO

> analyze this PR which replaces absolute benchmarking against a baseline by
> relative benchmarking that's only comparing a PR's base and head commits.
> Look into the previous benchmarking PR and analyze what can be removed from
> the benchmarking suite

> make the cuts, one per commt

> keep the absolute benchmarking results, but only keep the svg, markdown and
> generated json. artifacts should be generated for the head commit
> (regardless of whether the job is running on main or on a PR). otherwise
> make all cuts you mentionned

- [x] Render the pull request comparison once, in `report.py`, and let the
      comment workflow post it instead of reimplementing it in JavaScript
- [x] Drop the HTML and CSV renderings of the scaling table
- [x] Derive the plot panels from the data instead of the hardcoded `PLOTS`
- [x] Drop the `series` and `tensor` aliases for `repeated`
- [x] Drop the `BENCH_FLAGS=bench:smoke` no-op
- [x] Render the scaling plot as SVG rather than PNG
- [x] Render the report for the head commit on every run
