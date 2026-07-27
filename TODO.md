# TODO

Human prompt (verbatim):

> merge main into this PR and refactor it with the new RichDisplay
> https://github.com/discopy/discopy/pull/404

Context: `RichDisplay` landed on `main` with rich-display #445, closing #425. Luca asked on
this PR for #425 to be done properly before merging, so the temporary `show` helper of 7e2ca18
goes away.

- [x] Merge `main` into the branch, resolving the conflicts (`.ipynb` deletions,
      `docs/conf.py`, `pyproject.toml`, `CONTRIBUTING.md`)
- [ ] Port main's notebook changes to the marimo `.md` notebooks: `ob`/`ar` renamed to
      `ob_map`/`ar_map`, `ar_factory` renamed to `factory`, `drawing.Equation` moved to the
      relevant module, and the semantic changes (`foliation`, `is_close`, `Equation` truthiness)
- [ ] Drop the `show` helper from every notebook, displaying diagrams through `RichDisplay`
- [ ] Run `docs/export_notebooks.py --check`, `pflake8 discopy` and the test suite
