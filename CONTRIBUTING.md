# Contributing

So excited to have you here! If you want any guidance whatsoever, don't hesitate to reach out on [Discord](https://discopy.org/discord)!

## Run the tests

The first step is clone DisCoPy and install it locally.

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

## Build the docs

You can build the documentation locally with [sphinx](https://www.sphinx-doc.org/en/master/):

```shell
uv sync --group docs
uv run sphinx-build docs docs/_build/html
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
