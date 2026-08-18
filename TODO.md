# `trace(0)` should be a no-op, not an empty boundary

> good catch this is a big indeed let's fix it, n=0 should indeed be a no op (vanishing axiom)

— USER on [#578](https://github.com/discopy/discopy/issues/578), 2026-08-17

- [ ] guard `n == 0` in every `trace` that slices with `[:-n]`: `hypergraph`,
      `python.additive`, `python.multiplicative`
- [ ] `rigid.Diagram.curry(0, left=True)` curries the whole domain, same slice
- [ ] state the vanishing axiom on `abc.TracedCategory.trace`
- [ ] test the vanishing axiom against every implementation of `trace`
- [ ] CHANGELOG entry, `pflake8 discopy` and `coverage run -m pytest`
