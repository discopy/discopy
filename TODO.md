> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Tonight's head is [#780](https://github.com/discopy/discopy/issues/780), the board's ranked
candidate: `cat.Box.from_tree` rebuilds a box by keyword, so every subclass that narrows
`__init__` raises.

- [ ] reproduce all twelve failures on `main`, through the tree alone, so the result does not
      depend on the JSON encoder of #775/#782
- [ ] measure the issue's two candidate repairs before picking one
- [ ] implement the pick: a `to_tree`/`from_tree` pair per class that needs one
- [ ] every affected box round-trips, and no box in the package regresses
- [ ] tests that fail against `main`'s `gates.py`
- [ ] `pflake8`, `pylint`, `coverage run -m pytest`, property matrix
- [ ] `CHANGELOG.md` entry under `[Unreleased]`
