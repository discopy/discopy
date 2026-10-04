# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Taking [#781](https://github.com/discopy/discopy/issues/781) — *no `tensor.Box` round-trips
through JSON, and `matrix.Matrix` has no tree at all* — the head 🐦 ranked for tonight.

- [x] Reproduce both halves of #781 on `main` at `4d960251`
- [x] Enumerate every `NamedGeneric` with a `to_tree`, rather than trust the issue's list of two
- [WIP] @evening-kthknj-2026-10-04 00:30 Resolve a subscripted factory name on the read side, so today's trees read back
- [WIP] @evening-kthknj-2026-10-04 00:30 `monoidal.Dim` serialises its integers
- [WIP] @evening-kthknj-2026-10-04 00:30 `abc.Nat` serialises, and `monoidal.Nat` stops repeating it
- [WIP] @evening-kthknj-2026-10-04 00:30 `matrix.Matrix.to_tree`/`from_tree`, which `tensor.Tensor` inherits
- [WIP] @evening-kthknj-2026-10-04 00:30 Tests that fail against `main`'s code, one per defect
- [WIP] @evening-kthknj-2026-10-04 00:30 `CHANGELOG.md` entry under `[Unreleased]`
- [WIP] @evening-kthknj-2026-10-04 00:30 `pflake8 discopy`, `pylint discopy`, `coverage run -m pytest`, and the score against `main` in the same venv
- [WIP] @evening-kthknj-2026-10-04 00:30 Open the pull request as a draft ourselves (#779), delete this file to hand it over
