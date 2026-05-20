# Contributing

So excited to have you here! If you want any guidance whatsoever, don't hesitate to reach out on [Discord](https://discopy.org/discord)!

## Run the tests

DisCoPy uses [uv](https://docs.astral.sh/uv/) for local development, dependency groups, locking, and builds.
The first step is to clone DisCoPy and install the default development environment:

```shell
git clone https://github.com/discopy/discopy.git
cd discopy
uv sync
```

Then you should check you haven't broken anything by running the test suite:

```shell
uv sync --group test
uv run pflake8 discopy
uv run pylint discopy
uv run coverage run -m pytest
uv run coverage report -m --fail-under=99
```

The quantum and integration dependencies are intentionally kept outside the default install.
Use `uv sync --group test` before running the full test suite.
Use `uv sync --group all` if you want the development, test, and docs groups in one environment.

## Build the docs

You can build the documentation locally with [sphinx](https://www.sphinx-doc.org/en/master/):

```shell
uv sync --group docs
uv run sphinx-build docs docs/_build/html
```

## Build without uv

The project uses the `uv_build` PEP 517 build backend, so package builds still work from standard Python tooling.
If you do not use uv, create a virtual environment and install the relevant extras manually:

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
