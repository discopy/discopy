# Contributing

~~Let no one enter who does not know geometry.~~
*Let everyone enter and teach them category theory.*

Thank you for considering contributing to DisCoPy, we're so excited to have you here! If you got this far, you are already part of a new generation of engineers, scientists and mathematicians making equations and programs free of the one-dimensional cave in which they are being chained.

This is an open source project which started as part of [two PhD theses](https://docs.discopy.org/en/main/extra/papers.html#phd-theses) i.e. we come from academia and we are always enthusiastic about sharing ideas and their implementations.

Please read the [STYLE.md](STYLE.md) for guidelines which encapsulate some our coding philosophy.

## Make a first contribution

Every bit of contribution will be cherished however big or small, in particular you can:

- [Report bugs](#report-bugs)
- [Add documentation](#add-documentation)
- [Request features](#request-features)
- [Review pull requests](#review-pull-requests)

If you're unsure where to begin, we suggest you start with one of our tutorial notebooks e.g. [What is a diagram?](https://docs.discopy.org/en/main/notebooks/diagrams.html)
If you're looking for some inspiration on potential applications of string diagrams and category theory, you could try reading:

- the publications of the [Compositionality](https://compositionality.episciences.org/browse/volumes) journal
- the Applied Category Theory proceedings e.g. [dblp:eptcs429](https://dblp.org/db/series/eptcs/eptcs429.html)
- this list of papers at the intersection of [Category Theory ∩ Machine Learning](https://github.com/bgavran/Category_Theory_Machine_Learning)

If you want any guidance whatsoever, don't hesitate to reach out on [Discord](https://discopy.org/discord) or [open an issue](https://github.com/discopy/discopy/issues/new) even if it's to ask a simple question.

## Get started

DisCoPy uses [uv](https://docs.astral.sh/uv/).

The first step is to clone DisCoPy and install the default development environment:

```shell
git clone https://github.com/discopy/discopy.git
cd discopy
uv sync
```

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
uv run coverage run -m pytest
uv run coverage report -m
```

Without the extras installed, run `uv run pytest --skip-extra` to skip what needs them.

## Run the benchmarks

`benchmark/test_composition.py` reproduces the scaling experiments of
arXiv:2105.09257 for `Diagram` and `Hypergraph`, with analogous `CMap` cases;
`benchmark/test_conversion.py` covers conversions between all three. They live
outside `testpaths`, so run them explicitly. Results are keyed by suite
(`composition` or `conversion`), family (representation or conversion), case
(workload) and size `n`. Each data point is a declarative
[`pytest-benchmark`](https://pytest-benchmark.readthedocs.io) test; the fixture
owns timing (CPU clock and GC disabled) and automatically calibrates rounds
and iterations for each workload.

```shell
uv sync --group dev
# small/medium sizes (the default); add BENCH_FLAGS=bench:full for the heavy tail
uv run pytest benchmark/ -v --benchmark-json=benchmark-results/bench.json
# render the scaling tables + log-log plots (polars + matplotlib)
uv run python benchmark/report.py benchmark-results/bench.json
```

`report.py` writes `NAME-results.md` and `NAME-scaling.png` for each
`benchmark/test_NAME.py`. To compare two runs made sequentially on the same
machine, pass the base run when rendering the head:

```shell
uv run python benchmark/report.py benchmark-results/head.json \
    --base benchmark-results/base.json --fail-threshold 0.25
```

It joins the runs on `(suite, family, case, size)` and computes the raw change
`head / base - 1`. It writes `comparison.md`, listing regressions and speedups
larger than the threshold, and exits non-zero when an important regression is
present. Both runs must use the same benchmark sizes and machine; there is no
cross-machine normalisation. Only shared measurements are gated; the report
counts measurements present in only one of the two runs.

The `benchmark` GitHub workflow runs the full suite on `main` and manual
dispatches. On a pull request labelled `benchmark`, one job checks out and
benchmarks the exact base commit, then the exact head commit on the same runner.
It uploads the full report and posts or updates a pull request comment with the
important regressions and speedups.

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
python -m pip install coverage pyproject-flake8 pytest marimo
```

Then run:

```shell
pflake8 discopy
coverage run -m pytest
coverage report -m
```

To build distributions without uv:

```shell
python -m pip install build
python -m build
```

## Release a version

Before tagging, rename the `[Unreleased]` section of [CHANGELOG.md](CHANGELOG.md) to the new
version and date, and commit it.

New versions (tag with 'X.X.X') of the package are released on [PyPI](https://pypi.org/project/discopy/) using `uv publish`.
You should run the following commands from a clean clone of the repo:

```shell
git tag X.X.X
git push origin --tags
uv build
uv publish
```

Finally, [create a release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) for the newly created tag, using the
matching section of [CHANGELOG.md](CHANGELOG.md) as its description.

## Report bugs

We try our best to keep DisCoPy as close as possible to the mathematics but as any Python package it mostly likely contains bugs.
If you happen to find one, please [open an issue](https://github.com/discopy/discopy/issues/new) with your best attempt at describing what the problem is and how to reproduce it.

## Add documentation

We would be thrilled to welcome contributions in the form of examples, tests, notebooks, etc.
We are also keen to hear if you spot any part of the documentation that you suspect is broken, outdated or plain wrong.

We use the following convention so that documentation images are generated and compared against a baseline when running doctests:

```
Example
-------
>>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
>>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
>>> (f0 @ f1).draw(doctest='docs/_static/monoidal/tensor-example.svg')

.. image:: /_static/monoidal/tensor-example.svg
    :align: center
```

If the image already exists, drawing the example checks it against the
committed baseline and raises an error when they differ. To update an image,
delete its baseline so the next run regenerates it, or set
`discopy.config.OVERRIDE_DOCTEST_IMAGES = True` before running the tests, in
which case `doctest=` behaves like `path=` and just overrides the images.
Commit the regenerated images just like any other test change, otherwise
the CI won't pass. A plain `draw(path=...)` just saves the drawing,
overwriting any existing file.

## Request features

DisCoPy has the ambition to cover all of applied category theory.
If you are unsure what that can mean you could read [What is applied category theory?](https://www.appliedcategorytheory.org/what-is-applied-category-theory/) or [From quantum foundations via natural language meaning to a theory of everything](https://arxiv.org/abs/1602.07618).

If there's a particular feature needed for your application, we can probably guide you through how to implement it.
If your request is for some general abstract nonsense that can be used throughout many applications, we're also keen to hear about it.

## Review pull requests

We take our pull request reviews to the same level of rigour and courtesy as our academic peer reviews.
That is, we do our best to make sure that critical parts of the reasoning / implementation are correct but we also know there can be a next PR / paper fixing our mistakes.

## LLM guidelines

We accept contributions from large language models so long as they are explicitly indicated as such.
The [RULES.md](RULES.md) bind every agent working on a branch or pull request in this repo; they define the checkbox mutex and append-only shared-branch protocol.
Use our [AGENTS.md](AGENTS.md) in your prompts so that the model has enough context to give quality results.

LLMs have shifted the bottleneck of software development from writing code to reviewing it, please ensure that your AI assistants save more human time than they require to supervise them.
In particular, AI contributions should be small (a thousand lines is a red line not to cross lightly) and well-planned (delegate the execution not the design).

One specific guideline for PR descriptions: it's fine to have the detailed list of changes LLM-generated but the high-level description should be either a) written by a human, b) linking to a human-written prompt or c) quoting a human's prompt verbatim.
