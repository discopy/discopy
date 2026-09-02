# TODO

> yes let's go!

Approving, on #705's thread in this session, the proposal that followed the question *"is there any
way to shrink the module further? i would have expected the parametric lenses to reduce the code
size"*: one execution instead of three, `Interaction` dropped for `CMap`, and `solver`, `cells` and
`laws` out of the library into the notebooks of step three. The execution-as-a-functor idea stays a
design question and is not taken here.

- [x] one `Execution` for every backend: the fused torch path of `CMap.forward` becomes its vectorised strategy, grouping boxes by module, and `CMap.forward` dispatches to it alone
- [x] `Interaction` is gone: what `MapNN.compile` returns is the `CMap`, with the `(generator, role)` addressing of `read` and `write` on it
- [x] `solver`, `cells` and `laws` leave the package with their tests, to return as the notebooks of step three; `model` keeps only what a `CMap` needs to run
- [x] docstrings of what moved or merged trimmed with it, nothing else
- [x] `pflake8`, `pytest --skip-extra`, the JAX tests locally; changelog
- [ ] the torch half on CI
