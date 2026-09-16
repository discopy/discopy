# TODO

> actually, keep the delay for morphism, and forget about this @union thing, its hacky. just embrace the boilerplate without hacks
> minimize the diff with main, keep documentation minimal and not too verbose
> also, CartesianCategory is MarkovCategory + determinism (though determinism is only definable once we have property testing), then define CartesianClosedCategory, which should also inherit from ClosedCategory.
> the cartesian module should be cartesian closed though

- [WIP] @session_01Ux5oPZt3xhLRBUZBAZTQ4f-2026-09-16 16:30 Keep `delay(n_steps)` on morphisms and concrete objects, `d` stays the one abstract property of `DelayedMonoid`: restore `abc`, `feedback`, `traced`, `stream`, `para`, the tests and the README to their pre-conversion text.
- [ ] Drop the factory indirection added for the meet — `markov` daggers, `head_factory`/`tail_factory` — and write the overrides out in `cartesian` and `cartesian_feedback`.
- [ ] `abc.CartesianClosedCategory(CartesianCategory, ClosedCategory)`; `cartesian.Diagram` is cartesian closed and `python.Function` declares it.
- [ ] Trim the documentation: `cartesian_feedback` docstrings and the changelog entries, minimal and not verbose.
- [ ] `pflake8`, full test suite.
