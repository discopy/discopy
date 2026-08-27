# TODO

USER's ruling on [#609](https://github.com/discopy/discopy/issues/609)
("Abstraction over a non-atomic variable is silently wrong in biclosed and
crashes in closed"), verbatim:

> Enforce the invariant. `Variable.__init__` rejects a non-atomic `cod`, so
> the error arrives where the variable is built rather than as a wrong type
> or an internal `finset` message. This matches the rest of the term
> machinery, which counts free variables and wires interchangeably.

- [x] Reproduce both bugs from the issue on `main`.
- [x] Make `biclosed.Variable.__init__` (which `closed.Variable` inherits
      unchanged via MRO) raise `ValueError` via `utils.assert_isatomic` when
      `cod` is not atomic.
- [x] Add tests in `test/biclosed.py` and `test/closed.py`: non-atomic `cod`
      raises, atomic `cod` still works, `Abstraction`/`eval` on atomic
      variables unaffected.
- [x] Check other constructors of `Variable` in the codebase
      (`grammar/categorial.py`, doctests) for non-atomic `cod` — none found,
      all pass atomic (`Exp`) types.
- [x] Add a `[Fixed]` entry to `CHANGELOG.md`.
- [x] `uv run pflake8 discopy` clean.
- [x] `uv run pytest --skip-extra` green (702 passed, 51 skipped).
- [x] Mutation-check the new tests (revert the fix, confirm they fail;
      restore the fix).
