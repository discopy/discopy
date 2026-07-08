# Contributing

So excited to have you here! If you want any guidance whatsoever, don't hesitate to reach out on [Discord](https://discopy.org/discord)!

The first step is to clone DisCoPy and install the default development environment:

```shell
git clone https://github.com/discopy/discopy.git
cd discopy
uv sync
```

## Package infrastructure

DisCoPy uses [uv](https://docs.astral.sh/uv/).

Different dependency groups are available (switch with `uv sync --group <group-name>`):
- no group: minimal set of dependencies required to work with DisCoPy.
- `dev`: testing and linting tools.
- `quantum`: includes quantum computating dependencies
- `grammar`: natural language processing libraries
- `docs`: for generating the documentation
Since dependency groups are not standard, we also provide equivalents via optional dependencies.

## Run the tests

After cloning the repository, you should check you haven't broken anything by running the test suite.
Use `uv sync --dev` before running any part of the test suite, and `uv sync --dev --group all`
if you want to run the full test suite involving all extra dependencies.

```shell
uv sync --dev --group all
uv run pflake8 discopy
uv run pylint discopy
uv run coverage run -m pytest
uv run coverage report -m --fail-under=98
```

## Run the benchmarks

The composition benchmark (`benchmark/test_composition.py`) reproduces the scaling
experiments of arXiv:2105.09257 for both `Diagram` and `Hypergraph`. It lives
outside `testpaths`, so the normal `pytest` run never collects it — run it
explicitly. Each `(case, size)` is a declarative
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io) test — the fixture
owns timing (CPU clock, GC disabled, median of a few rounds), so there is no
hand-rolled timing code.

```shell
uv sync --group dev
# small/medium sizes (the default); add BENCH_FLAGS=bench:full for the heavy tail
uv run pytest benchmark/ -v --benchmark-json=benchmark-results/bench.json
# render the scaling table + log-log plot (polars + matplotlib)
uv run python benchmark/report.py benchmark-results/bench.json
```

`report.py` writes `results.md`, `results.csv` and `scaling.png` into
`benchmark-results/`. To gate on a regression, pass a committed baseline:

```shell
uv run python benchmark/report.py benchmark-results/bench.json \
    --baseline benchmark/baseline.json --fail-threshold 0.25
```

It joins the two runs on `(case, size)` and exits non-zero if any case's median
regressed by more than the threshold. The baseline is machine-dependent, so
generate it once on the CI runner (`workflow_dispatch` on `main`, with
`BENCH_FLAGS=bench:full`) and commit the resulting `bench.json` as
`benchmark/baseline.json`. The `benchmark` GitHub workflow runs the suite on pull
requests (smoke sizes) and on `main` / manual dispatch (full sizes), uploading the
report as an artifact.

## Build the docs

You can build the documentation locally with [sphinx](https://www.sphinx-doc.org/en/master/):
You'll need to install [pandoc](https://pandoc.org/) and [graphviz](https://graphviz.org/) as external dependencies not managed by `uv`.

```shell
uv sync --group docs
uv run sphinx-build docs docs/_build/html
```

## Build without uv

The project uses the `uv_build` PEP 517 build backend, so package builds still work from standard Python tooling.
If you do not use `uv`, create a virtual environment and install the relevant extras manually:

```shell
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pip install coverage pyproject-flake8 pylint pytest nbmake
```

Then run:

```shell
pflake8 discopy
pylint discopy
coverage run -m pytest
coverage report -m --fail-under=99
```

To build distributions without uv:

```shell
python -m pip install build
python -m build
```

## Release a version

New versions (tag with 'X.X.X') of the package are released on [PyPI](https://pypi.org/project/discopy/) using `uv publish`.
You should run the following commands from a clean clone of the repo:

```shell
git tag X.X.X
git push origin --tags
uv build
uv publish
```

Finally, [create a release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) for the newly created tag.
