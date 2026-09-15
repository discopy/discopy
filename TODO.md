# TODO

> why do we need this extra planar layer? why not just have:
>
> * feedback -> symmetric (not markov)
> * traced -> feedback (not markov)
> * closed -> biclosed + symmetric (not markov)
>
> the structure of free diagrams should be as defined in the literature

- [x] Dissolve `discopy.planar`: the `Trace` bubble moves to `monoidal` as pure syntax next to `Bubble`, the trace recursion is stated once on `abc.TracedCategory` via `trace_factory`, and the `monoidal.Functor` maps a `Trace` whenever its codomain has one, so `balanced` keeps its own `Trace` and `pivotal` keeps deriving its trace from cups and caps with no planar base.
- [x] `closed.Diagram` inherits from `symmetric.Diagram` and `biclosed.Diagram` rather than `markov.Diagram`, keeping the copy supply its non-linear lambda terms use; `abc.ClosedCategory` extends `BiclosedCategory` and `SymmetricCategory` accordingly.
- [x] Re-point the users: `interaction` reads its trace off the abc, `cmap` doctests and tests move from `planar` to `traced` or `balanced`, `test/planar.py` redistributes, docs updated.
- [x] Changelog, `pflake8`, full test suite.
