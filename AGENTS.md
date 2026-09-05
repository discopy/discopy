# AGENTS.md

## What

DisCoPy is a Python toolkit for computing with string diagrams.

Read the following documents (<10k tokens) before any work on the package:

- [RULES.md](RULES.md) describes the collaboration and Git protocol for shared PR branches, follow it exactly.
- [STYLE.md](STYLE.md) describes coding guidelines that all your work should try to follow.
- [README.md](README.md) contains a high-level description of the features along with some examples.
- [CONTRIBUTING.md](CONTRIBUTING.md) contains setup instructions and our general coding philosophy.

## Where

- [discopy](discopy/) contains the code with a lot of modules and a few submodules
- [discopy.abc](discopy/abc.py) contains abstract base classes for each level of the hierarchy
- [discopy.testing](discopy/testing.py) contains the property-testing module and documents the property-first workflow: laws stated before implementation, counterexamples recorded, strategies audited. Read it when a property test fails or before adding a carrier or an axiom.
- [test](test/) is flat with one file for each module, mirroring `discopy/`; submodules (`grammar`, `quantum`, `drawing`, `python`) keep their own directory with one file per submodule, and shared test data lives in `fixtures`
- [docs](docs/) contains notebooks and pictures generated automatically when running the tests

## How

Before writing any code, make sure that:

1) your change was first described in high-level mathematical terms
2) this description aligns with the data structures you plan to use

Before pushing anything, make sure that:

- you have reported any bugs or confusing docs that you encounter even if unrelated
- you have added docs and tests that are complete but concise as best as you can
- you have added an entry to the `[Unreleased]` section of [CHANGELOG.md](CHANGELOG.md) for any user-facing change
- you have `uv run` both `pflake8 discopy` and `coverage run -m pytest` as described in [CONTRIBUTING.md](CONTRIBUTING.md)
- you have respected the [code style guide](CONTRIBUTING.md#code-style-guide)

A pull request is **ready for sign-off** when all three of these hold:

1) its `TODO.md` is deleted, every point having been `[x]` or filed as an issue
2) CI is green on the real jobs, with the target branch merged in
3) no review thread is waiting on an agent, i.e. every thread is either resolved or waiting on human feedback
