# AGENTS.md

Agent-facing notes for working in this repository. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor guide.

## Setup

```shell
uv sync --group dev            # tests + linting
uv sync --group dev --group all  # everything (quantum, grammar, docs)
```

## Tests & linting

```shell
uv run pflake8 discopy
uv run pylint discopy
uv run coverage run -m pytest
uv run coverage report -m --fail-under=98
```

The default `pytest` run collects `discopy` (doctests), `test/*/*.py` and the
notebooks. Long-running benchmarks are marked `slow` and live outside
`testpaths`, so they are never part of the default run.

## Benchmarks

The composition benchmark (`benchmark/test_composition.py`, arXiv:2105.09257)
uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io): each
`(case, size)` is one parametrised `benchmark.pedantic` test, the fixture owns
timing (CPU clock, GC disabled, median). Size is swept via
`@pytest.mark.parametrize` and gated by `BENCH_FLAGS` (small/medium by default,
`bench:full` for the heavy tail). Reporting is separate: `benchmark/report.py`
reads the `--benchmark-json` output and renders the table/plot (polars +
matplotlib), and optionally gates against a committed baseline.

```shell
uv run pytest benchmark/ -v --benchmark-json=benchmark-results/bench.json
uv run python benchmark/report.py benchmark-results/bench.json
```

Don't add a hand-rolled timing harness — use the `benchmark` fixture. Keep
measurement (the test) and rendering (`report.py`) separate. See
[Run the benchmarks](CONTRIBUTING.md#run-the-benchmarks).
