> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Tonight's head is [#780](https://github.com/discopy/discopy/issues/780), the board's ranked
candidate: `cat.Box.from_tree` rebuilds a box by keyword, so every subclass that narrows
`__init__` raises.

- [x] reproduce all twelve failures on `main`, through the tree alone, so the result does not
      depend on the JSON encoder of #775/#782
- [x] measure the issue's two candidate repairs before picking one — option 2's premise is false:
      `dagger` and `rotate` are not field rebuilds (`Copy`↔`Match`, `Ket`↔`Bra`, `Sqrt`→self)
- [x] implement the pick: a `to_tree`/`from_tree` pair per class that needs one
- [x] every affected box round-trips, and no box in the package regresses — 41 boxes enumerated
      from the module rather than taken from the issue's list, which found four more classes
- [x] tests that fail against `main`'s `gates.py` — 37 of 48 do
- [x] `pflake8`, `pylint` 8.73 against `main`'s 8.72 in the same venv, 995 passed / 2 skipped,
      property matrix 10 passed
- [x] `CHANGELOG.md` entry under `[Unreleased]`
- [x] the package-wide half reported on #780 rather than folded in
