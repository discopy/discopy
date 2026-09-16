# TODO

> do the same refactor for cartesian and all other categories touched in this PR
> define generators for projections in cartesian too
> make the most out of this other branch, it should greatly simplify the entire architecture of the package and remove as many hacks and boilerplate as possible

- [WIP] @session_01Ux5oPZt3xhLRBUZBAZTQ4f-2026-09-16 18:30 Sweep every module this PR touches for shells and hand assignments the `Generator` machinery now covers: `feedback`, `traced`, `closed`, `markov`, `balanced`, `symmetric`, `compact`, `cartesian`, `cartesian_feedback`.
- [WIP] @session_01Ux5oPZt3xhLRBUZBAZTQ4f-2026-09-16 18:45 Route the term factories through `Generator` the same way, so `cartesian` pulls `TermBase`, `Constant` and `Variable` and declares only its `Application`.
- [WIP] @session_01Ux5oPZt3xhLRBUZBAZTQ4f-2026-09-16 18:45 `Projection` generators in `cartesian`: `abc.CartesianCategory.projection` derived from copy and discard, the free generator declared with `@Generator`, mapped by `Functor`, with tests.
- [ ] Changelog, `pflake8`, full test suite.
