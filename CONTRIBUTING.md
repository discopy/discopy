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

DisCoPy uses [CodSpeed](https://codspeed.io) via [pytest-codspeed](https://github.com/CodSpeedHQ/pytest-codspeed) to track performance over time. Benchmarks live under `test/bench/` (integration) and as `test_*_bench` functions inside existing unit test files (micro-benchmarks).

Install the bench extra and run locally:

```shell
uv sync --group dev --extra bench
uv run pytest test/bench/ --codspeed -v          # small + medium sizes
BENCH_FLAGS=bench:full uv run pytest test/bench/ --codspeed -v   # all sizes
```

Without `--codspeed` the `benchmark` fixture is a no-op passthrough — micro-benchmarks in unit test files collect and pass during normal `uv run pytest` without the bench extra installed.

**Adding a new benchmark alongside a unit test:**

```python
import pytest

@pytest.mark.benchmark(group="my-module-micro")
def test_my_thing_bench(benchmark):
    # setup (not timed)
    data = build_fixture()
    # only this lambda is instrumented
    benchmark(lambda: my_function(data))
```

**Wiring CodSpeed in GitHub:**
The `.github/workflows/codspeed.yml` workflow runs on every push to `main` and on pull requests. To upload results to the CodSpeed dashboard you need to add a `CODSPEED_TOKEN` secret to the repository:

1. Sign in at <https://codspeed.io> and create a project linked to this repository.
2. Copy the token from the CodSpeed project settings.
3. Add it as a repository secret named `CODSPEED_TOKEN` at *Settings → Secrets and variables → Actions*.

The workflow runs cleanly without the token (walltime mode collects and prints results locally) — the secret is only required for the dashboard upload.

## Build the docs

You can build the documentation locally with [sphinx](https://www.sphinx-doc.org/en/master/):
You'll need to install [pandoc](https://pandoc.org/) as an external dependency not managed by `uv`.

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
