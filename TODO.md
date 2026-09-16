# TODO

> remove the delay method, we only need d as an abstract property
> also remove the copy/discard supply on feedback, if necessary make a new cartesian_feedback module. for now keep it simple, but while you implement it, conceive a way to avoid this combinatorial explosion of structure by finding a convenient way to combine structure without repeating too much code

- [ ] Replace `delay(n_steps)` by the abstract property `d` on `abc.DelayedMonoid` and `abc.FeedbackCategory`, trivial on `abc.TracedCategory` and `monoidal.Ty`, with the `time_step` arithmetic of `feedback` going through constructors.
- [ ] Remove the copy supply from `feedback` and `traced`; a new `cartesian_feedback` module hosts the meet, the stream examples that copy, and their tests; `test_simplify` moves to `frobenius`.
- [ ] Route the cross-generator references through factories — `Copy.dagger` via `merge_factory` like `Constant.__call__` via `application_factory` — so a meet module is one-line subclasses and factory assignments, no re-overrides.
- [ ] Changelog, `pflake8`, full test suite.
