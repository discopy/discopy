# TODO

> reverted, keep factory as is, but reintroduce @generator to avoid duplication between all the def x_factory methods

- [WIP] @session_01S8HY2Chfnczvn5kcoTY9wE-2026-09-16 `utils.Generator` back in place of `utils.cached_classproperty`: each `x_factory` is a method returning its class, decorated with `@Generator()`, or `@Generator("permutation_factory")` for a swap and `@Generator("copy_factory")` for a discard, the building of the subclass written once; `utils.factory` unchanged; tests, README and CHANGELOG follow; `pflake8`, the full suite and the property tests green.
