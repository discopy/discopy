# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Taking [#652](https://github.com/discopy/discopy/issues/652) — *`Matrix.copy(x, n)` is wrong for
`x, n >= 2`* — the head 🐦 ranked for tonight.

- [x] Reproduce #652 on `main` at `4d960251`, over a range of `x` and `n` rather than its examples
- [x] Enumerate what else is wrong with the comonoid, rather than trust the issue's one defect
- [x] The diagonal `x -> n * x`, built without the in-place assignment `swap` uses — JAX refuses it
- [x] `copy`, `discard`, `merge` and `ones` carry the dtype of their carrier
- [x] Tests that fail against `main`'s code, one per defect, and the three laws as `Matrix` equalities
- [x] `CHANGELOG.md` entry under `[Unreleased]`
- [ ] `pflake8 discopy`, `pylint discopy`, `coverage run -m pytest`, and the score against `main` in the same venv
- [ ] Open the pull request as a draft ourselves (#779), delete this file to hand it over
