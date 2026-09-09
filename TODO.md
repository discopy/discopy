# TODO

Review round on PR #658 by toumix, 2026-09-09 08:13 to 08:16 UTC, on `discopy/axioms.py`:

> That's not true for the axioms that are quantified over the objects of the category rather than the morphisms

> "the underlying set of a model" that's the issue: a category is defined by two sets not one, in that sense it's not a universal algebra and "carrier" just sounds like the wrong abstraction

> This sounds like slop, "categorical law" isn't a technical concept, "carrier" isn't a categorical concept

> maybe better to just call this `category`

- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-09 08:30 Rename `Axiom.carrier` to `Axiom.category` and say "category" wherever the docs, the tests, the matrix and the changelog said "carrier": an axiom is stated of a category and quantified over its objects, its arrows or its terms
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-09 08:30 Reply on the three threads, resolve them, refresh the PR body and tell #659 about `proptest/categories.py`
