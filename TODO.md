# TODO

> introduce abc.DaggerCategory as described in https://github.com/discopy/discopy/issues/731
> leave a dagger implementation on diagrams as it currently stands, but avoid infecting `Functor`s with it so that property testing (in #658, #659) doesn't generate the associated axiom instead of marking dagger contravariance and involution as inapplicable.

- [x] Add `DaggerCategory` to `discopy.abc`: a `Category` with an abstract `dagger`, documenting involution and contravariance, listed in the module summary.
- [x] Make `cat.Arrow` a `DaggerCategory` so the diagram hierarchy inherits it, leaving `FreeCategory` (shared with types) and `Functor` untouched.
- [x] Test that arrows and diagrams are dagger categories while types and functors are not.
- [WIP] @session_0132JeN3utT2cHDd1hFL3wCJ-2026-09-09 11:04 - Add a `CHANGELOG.md` entry, run `pflake8` and the test suite.
