# AGENTS.md

## What

DisCoPy is a Python toolkit for computing with string diagrams.

## Context

Please read the following documents (<10k tokens) before attempting any serious work on the package:

- @README.md contains a high-level description of the features along with some examples.
- @CONTRIBUTING.md contains setup instructions, our philosophy and a code style guide.
- @RULES.md contains the rules for every agent — human-run or autonomous — that writes code
  in this repo, enforced by the `no-todo-on-main` merge gate.

## Where

- [discopy](discopy/) contains the code with a lot of modules and a few submodules
- [discopy.abc](discopy/abc.py) contains abstract base classes for each level of the hierarchy
- [test](test/) is split into `syntax` and `semantics` with one file for each module or submodule
- [docs](docs/) contains notebooks and pictures generated automatically when running the tests

## How

Before writing any code, make sure that:

1) your change was first described in high-level mathematical terms
2) this description aligns with the data structures you plan to use

Before pushing anything, make sure that:

- you have reported any bugs or confusing docs that you encounter even if unrelated
- you have added docs and tests that are complete but concise as best as you can
- you have `uv run` both `pflake8 discopy` and `coverage run -m pytest` as described in @CONTRIBUTING.md
- you have respected the [code style guide](CONTRIBUTING.md#code-style-guide)
