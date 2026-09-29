# TODO

CodeRabbit's architecture review, three retained concerns. Quoted verbatim:

> **Medium · architecture · inferred:** A successful setup report creates a passing cell before its
> property body runs. If no settling call report follows but terminal reporting proceeds, the
> current report can describe that cell as checked without draw data.

> **Medium · architecture · observed:** When no marked cells reach the collector, terminal reporting
> returns before comparing the baseline or writing this run's report.

> **Low · reliability · observed:** A syntactically valid baseline whose JSON root is not an object
> raises an unhandled exception during terminal reporting.

All three are against `3fe6cf41`. **The second and third landed in `d9b494a0`** with the five
inline findings. The first is live on the current head and is the sharpest instance of the defect
this module exists to remove — reproduced by driving the hook with a setup phase and no call:

```
after setup alone: {'cat.Arrow.unitality': Cell(outcome='passed', distinct=None, budget=None)}
render says:
  every one of the 1 passing cell(s) drew a distinct term for each example it was given
```

A law that was never stated, reported as one checked on a distinct term per example.

- [ ] A cell reads `passed` only when a `call` phase says so: seed the collector with an outcome
      that means *started and never settled* instead of `PASSED`
- [ ] That outcome counts as no longer checking its law, so a baseline cell that becomes it is lost
      coverage, and the summary names it in its own right
- [ ] Tests: the hook driven phase by phase — setup alone, setup then call, a skip at setup — and
      the baseline transition
- [ ] Re-validate: `pflake8`, the matrix at `dev` and `pr`, serial and `-n auto`, the full suite,
      and confirm every cell still reads `passed` under `-n auto` where three phases do arrive
- [ ] Reply on the review, saying which of the three were already fixed and which this round fixes
