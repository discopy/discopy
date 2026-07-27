# TODO

Prompt (Alexis, [memory#21 review](https://github.com/toumix/memory/pull/21#discussion_r3654941486), verbatim):

> We can have a lighter test set up which avoids this, open a PR on discopy

The "this" is the state a sandboxed agent finds the suite in: with only the
default dependencies installed, `pytest` aborts during collection with eight
errors and runs nothing at all, so a run cannot tell a real regression from a
missing wheel.

- [x] Reproduce: `uv sync --group dev` then `uv run pytest` — 8 collection errors, 0 tests run
- [x] Ignore the two package modules that cannot be imported without their backend
- [x] Let the test modules in the same case skip themselves with `pytest.importorskip`
- [x] Skip, rather than fail, a doctest that only wants an optional backend or graphviz
- [x] Do the same for a notebook, which reports the error as text rather than an exception
- [x] `uv run pflake8 discopy conftest.py`
- [x] `uv run pytest` green on the default install: 521 passed, 70 skipped
- [x] Confirm on CI that the full install still runs everything, with nothing skipped — `test (3.14)` on `03ebf1a`: **753 passed, 1 skipped**, every quantum module still collected and covered
- [x] Rename `.claude/` to `.agents/` and repoint RULES.md rule 4 — Alexis on the review: "Also let's rename .claude to .agents to be provider-agnostic", "no same PR is fine"
- [x] Keep `conftest.py` out of coverage: its skip paths are unreachable once the extras are installed, so they were pulling the total under the 98% gate
