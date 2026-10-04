# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Taking [#781](https://github.com/discopy/discopy/issues/781) — *no `tensor.Box` round-trips
through JSON, and `matrix.Matrix` has no tree at all* — the head 🐦 ranked for tonight.

- [ ] Reproduce both halves of #781 on `main` at `4d960251`
- [ ] Enumerate every `NamedGeneric` with a `to_tree`, rather than trust the issue's list of two
- [ ] Resolve a subscripted factory name on the read side, so today's trees read back
- [ ] `monoidal.Dim` serialises its integers
- [ ] `abc.Nat` serialises, and `monoidal.Nat` stops repeating it
- [ ] `matrix.Matrix.to_tree`/`from_tree`, which `tensor.Tensor` inherits
- [ ] Tests that fail against `main`'s code, one per defect
- [ ] `CHANGELOG.md` entry under `[Unreleased]`
- [ ] `pflake8 discopy`, `pylint discopy`, `coverage run -m pytest`, and the score against `main` in the same venv
- [ ] Open the pull request as a draft ourselves (#779), delete this file to hand it over
