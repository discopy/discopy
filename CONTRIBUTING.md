# Contributing

~~Let no one enter who does not know geometry.~~
*Let everyone enter and teach them category thory.*

Thank you for considering contributing to DisCoPy, we're so excited to have you here! If you got this far, you are already part of a new generation of engineers, scientists and mathematicians making equations and programs free of the one-dimensional cave in which they are being chained.

This is an open source project which started as part of [two PhD theses](https://docs.discopy.org/en/main/extra/papers.html#phd-theses) i.e. we are academics and we are always enthusiastic about collaboration, sharing and discussing ideas and their implementations.

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

If you want any guidance whatsoever, don't hesitate to reach out on [Discord](https://discopy.org/discord) (sadly not very active) or [open an issue](https://github.com/discopy/discopy/issues/new) even if it's to ask a simple question.

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
python -m pip install coverage pyproject-flake8 pytest nbmake
```

Then run:

```shell
pflake8 discopy
coverage run -m pytest
coverage report -m --fail-under=98
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

## Report bugs

We try our best to keep DisCoPy as close as possible to the mathematics but as any Python package it mostly likely contains many bugs.
If you happen to find one, please [open an issue](https://github.com/discopy/discopy/issues/new) with your best attempt at describing what the problem is and how to reproduce it.

## Add documentation

We would be thrilled to welcome contributions in the form of examples, tests, notebooks, etc.
We are also keen to hear if you spot any part of the documentation that you suspect is broken, outdated or plain wrong.

We use the following convention so that documentation images are generated automatically when running doctests:

```
Example
-------
>>> x, y, z, w = Ty('x'), Ty('y'), Ty('z'), Ty('w')
>>> f0, f1 = Box('f0', x, y), Box('f1', z, w)
>>> (f0 @ f1).draw(path='docs/_static/monoidal/tensor-example.png')

.. image:: /_static/monoidal/tensor-example.png
    :align: center
```

For now this is not done automatically so make sure you remember to push changes to these documentation images but don't push if the changes are only due to minor glitches e.g. font aliasing.

## Request features

DisCoPy has the ambition to cover all of applied category theory.
If you are unsure what that can mean you could read [What is applied category theory?](https://www.appliedcategorytheory.org/what-is-applied-category-theory/) or [From quantum foundations via natural language meaning to a theory of everything](https://arxiv.org/abs/1602.07618).

If there's a particular feature needed for your application, we can probably guide you through how to implement it.
If your request is for some general abstract nonsense that can be used throughout many applications, we're also keen to hear about it.

## Review pull requests

We take our pull request reviews to the same level of rigour and courtesy as our academic peer reviews.
That is, we do our best to make sure that critical parts of the reasoning / implementation are correct but we also know there can be a next PR / paper fixing our mistakes.

## Code style guide

- **DisCoPy is pure.** Diagram composition should never cause side-effects, only functor application does when the codomain is effectful.
- **DisCoPy is deterministic.** Even in their internal representation, data structures should not depend on sources of non-determinism (e.g. hashing).
- **DisCoPy is transparent.** `eval(repr(x)) == x` should always be true and `eval(str(x)) == x` should be true assuming the obvious variable naming convention e.g. `x, y = Ob("x"), Ob("y")` and `f = Box("f", x, y)`. This `str(x)` should be as close as possible to what a mathematician would write on the board.
- **DisCoPy has no secrets.** We avoid using private or semiprivate attributes and let the user see the internals of each data structure. We expose the interface of every subprocedure as methods that can be tested and reused.
- **DisCoPy cares about naming.** Classes and methods should have short descriptive names, when possible the names correspond to well-known mathematical definitions.
- **DisCoPy speaks for itself.** The code should be clear enough that it doesn't need comments, only documentation with links to mathematical definitions.
- **DisCoPy does not show off.** If there is a simpler way to name or explain something, don't make it more sound more complicated.
- **DisCoPy never repeats itself.** The identity and composition of diagrams are defined once in `cat`, not in every level of the hierarchy. If there's duplicate code then you're probably working at the wrong level of abstraction.
- **DisCoPy aims at never nesting.** We believe if your code goes beyond three levels deep then you're probably working at the wrong level of abstraction.

## LLM guidelines

We accept contributions from large language models so long as they are explicitly indicated as such.
We recommend using our [AGENTS.md](AGENTS.md) in your prompts so that the model has enough context to give quality results.

LLMs have shifted the bottleneck of software development from writing code to reviewing it, please ensure that your AI assistants save more human time than they require to supervise them.
In particular, AI contributions should be small (a thousand lines is a red line not to cross lightly) and well-planned (delegate the execution not the design).

One specific guideline for PR descriptions: it's fine to have the detailed list of changes LLM-generated but the high-level description should be either written by a human or quoting a human's prompt verbatim.
