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
pip install ".[test]" .
coverage run -m pytest --doctest-modules --pycodestyle
coverage report -m discopy/*.py discopy/*/*.py
```

## Build the docs

You can build the documentation locally with [sphinx](https://www.sphinx-doc.org/en/master/):

```shell
pip install ".[docs]" .
sphinx-build docs docs/_build/html
```
