# TODO

Self-initiated fix during the nightly 🌙 Evening cycle — no human prompt behind it, this
paragraph stands in for one per `RULES.md`.

`python.additive.Function.trace` translates a looping **output** tag straight back in as an
**input** tag. The two only coincide when `len(dom) == len(cod)`, which is why every existing
test (all endo-shaped) missed it — filed as [#554](https://github.com/discopy/discopy/issues/554).

- [x] Reproduce the bug on an unequal-arity traced function and confirm the reported `IndexError`.
- [x] Fix `Function.trace`'s `inside` closure to translate the tag (`tag - len(cod) + len(dom)`)
      before feeding it back in, only when re-entering the loop (not on the first, externally
      supplied tag).
- [x] Add a regression test in `test/python/additive.py` covering `len(dom) != len(cod)`.
- [x] `CHANGELOG.md` entry under `[Unreleased]` / `Fixed`.
- [x] `uv run pflake8 discopy` clean.
- [x] `uv run pytest --skip-extra`: 626 passed, 51 skipped (same skip count as before the change).
