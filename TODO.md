# TODO

No verbatim human prompt — self-contained bug fix found and fixed in one
session, per the discopy#513/#514 precedent (see also optyx#34).

Fixes #522: `frobenius.Diagram.unfuse`'s doctest sets `Spider.color = "red"`
and never resets it, leaking into every later doctest of the same pytest
process and baking red spiders into unrelated committed SVG baselines.

- [x] Reset `Spider.color` back to `"black"` at the end of the `unfuse`
      doctest in `discopy/frobenius.py`.
- [x] Regenerate the two baselines that had silently inherited the leaked
      red colour: `docs/_static/tensor/frobenius-example.svg` and
      `docs/_static/tensor/chain-rule.svg`. (Checked: no other baseline in
      the repo is affected — these are the only two doctests that draw a
      default-coloured Frobenius spider after `unfuse`'s in suite order.)
- [x] `uv run pflake8 discopy`: clean.
- [x] `uv run python -m pytest --skip-extra`: 625 passed, 50 skipped.
- [x] `CHANGELOG.md` entry added.
