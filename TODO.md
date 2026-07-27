# TODO

Prompt (Alexis, [memory#21 review](https://github.com/toumix/memory/pull/21#discussion_r3654941486), verbatim):

> We can have a lighter test set up which avoids this, open a PR on discopy

The "this" is the state a sandboxed agent finds the suite in: with only the
default dependencies installed, `pytest` aborts during collection with eight
errors and runs nothing at all, so a run cannot tell a real regression from a
missing wheel.

- [x] Reproduce: `uv sync --group dev` then `uv run pytest` — 8 collection errors, 0 tests run
- [x] One rule for all of it, per review: read the error, and if it names a dependency outside
      `uv sync --dev`, skip instead of fail — at collection (a module that cannot be imported) and
      at call (a doctest or a notebook that needs a backend). No per-file list, no edits to the
      test modules.
- [x] Put the explanation in CONTRIBUTING.md where it belongs, not in the `conftest.py` docstring
- [x] `uv run pflake8 discopy conftest.py`
- [x] `uv run pytest` green on the default install: 509 passed, 80 skipped after merging `main`
- [x] Confirm on CI that the full install still runs everything, with nothing skipped — `test (3.14)` on `03ebf1a`: **753 passed, 1 skipped**, every quantum module still collected and covered
- [x] Keep `conftest.py` out of coverage: its skip paths are unreachable once the extras are installed, so they were pulling the total under the 98% gate
