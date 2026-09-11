# TODO

> `main` moved to `6008fd7`, "Add List, the free monoid on a type, and drop the
> tuple functor hack (#728) (#747)", which left #659 `dirty`: three conflicts in
> `CHANGELOG.md`, `discopy/monoidal.py` and `test/monoidal.py`.

#747 rewrote `monoidal.Functor.__call__` around `List`: `_map_colour` is
inlined, `_map_atomic` — the tuple hack — is `super().__call__`, and the `sum`
loops are `tensor`. That version subsumes this branch's, and drops two
semiprivate helpers `STYLE.md` would rather not have, so it is taken whole.

- [WIP] @session_01SojhqwnPVHxjyJHhXs4hy1-2026-09-11 16:05 Merge `main`:
      union the imports, take #747's `Functor.__call__`, keep both files' tests
- [WIP] @session_01SojhqwnPVHxjyJHhXs4hy1-2026-09-11 16:05 Reconcile
      `CHANGELOG.md`'s `[Unreleased]` with #747's entry
- [ ] `pflake8` clean, both suites green, every change in the counts explained
