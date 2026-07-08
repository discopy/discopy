# AGENTS.md

## What

DisCoPy is a Python toolkit for computing with string diagrams.

## Context

Please read the following documents (<10k tokens) before attempting any serious work on the package:

- @README.md contains a high-level description of the features along with some examples.
- @CONTRIBUTING.md contains setup instructions, our philosophy and a code style guide.

## Where

- [discopy](discopy/) contains the code with a lot of modules and a few submodules
- [discopy.abc](discopy/abc.py) is the main entry point to navigating the architecture
- [test](test/) is split into `syntax` and `semantics` with one file for each module or submodule
- [docs](docs/) contains notebooks and pictures generated automatically when running the tests

## How

Before pushing anything, make sure that:

- you have first described your change in terms of high-level mathematical definitions
- you made sure that this description aligns with the data structures you have used
- you have added docs and tests that are complete but concise as best as you can
- you have run `pflake8`, `pytest` and checked that `coverage` hasn't dropped
- you report any bugs or confusing docs that you encounter even if unrelatedww