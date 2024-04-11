# Contributing

So excited to have you here! If you want any guidance whatsoever, don't hesitate to reach out on [Discord](https://discopy.org/discord)!

## Run the tests

The first step is clone DisCoPy and install it locally.

```shell
git clone https://github.com/discopy/discopy.git
cd discopy
pip install .
```

Then you should check you haven't broken anything by running the test suite:

```shell
pip install ".[test]"
pycodestyle discopy
coverage run -m pytest
coverage report -m --fail-under=99
```

## Build the docs

You can build the documentation locally with [sphinx](https://www.sphinx-doc.org/en/master/):

```shell
pip install ".[docs]"
sphinx-build docs docs/_build/html
```

## Release a version

New versions (tag with 'X.X.X') of the package are released on [PyPI](https://pypi.org/project/discopy/) using `twine`.
You should run the following commands from a clean clone of the repo:

```shell
pip install twine
pip install 'build<0.10.0'
git tag X.X.X
git push origin --tags
python -m build
twine upload dist/*
```

Finally, [create a release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) for the newly created tag.
