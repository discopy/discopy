> on the algorithm itself: I'm not sure whether de bruijn indices computed on the fly are as efficient as the substitution accumulators? we want to avoid taking two terms and building two new terms when we compare them

— toumix, 2026-09-17 20:58, on the review thread of #767; answered there with the measurement, binder depths three to eight times faster than the substitutions and neither building a term, and the choice between the two. Then:

> ok you're clearly better at Python than I am let's go

— toumix, 2026-09-18 08:22.

- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-18 08:45 `alpha_eq_under` back to binder depths: a `dict[Variable, int]` per term in place of the `Substitution`s, extended and restored the same way, no fresh names so no `free`; the categorial overrides, the docstrings, the tests, the changelog and the description follow
- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-18 08:45 the `Sampler` defers each node through a closure rather than `functools.partial`, whose calls nest on the C stack and overflow it on a deep term where Python frames do not
