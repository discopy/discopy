# TODO

> actually, keep the delay for morphism, and forget about this @union thing, its hacky. just embrace the boilerplate without hacks
> minimize the diff with main, keep documentation minimal and not too verbose
> also, CartesianCategory is MarkovCategory + determinism (though determinism is only definable once we have property testing), then define CartesianClosedCategory, which should also inherit from ClosedCategory.
> the cartesian module should be cartesian closed though

- [x] Keep `delay(n_steps)` on morphisms and concrete objects, `d` stays the one abstract property of `DelayedMonoid`: restore `abc`, `feedback`, `traced`, `stream`, `para`, the tests and the README to their pre-conversion text.
- [x] Drop the factory indirection added for the meet — `markov` daggers, `head_factory`/`tail_factory` — and write the overrides out in `cartesian` and `cartesian_feedback`.
- [x] `abc.CartesianClosedCategory(CartesianCategory, ClosedCategory)`; `cartesian.Diagram` is cartesian closed and `python.Function` declares it.
- [x] Trim the documentation: `cartesian_feedback` docstrings and the changelog entries, minimal and not verbose.
- [x] `pflake8`, full test suite.
