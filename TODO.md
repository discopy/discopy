# TODO

Review round of 2026-09-15 by toumix on https://github.com/discopy/discopy/pull/400:

> call these Predicate and Quantifier instead

(`test/grammar/abstract.py`, `ET, NP = e >> t, (e >> t) >> t`)

> this breaks the repr transparency, it's usually called "Position" denoted little o
> ```suggestion
> Position = Ty("o")
> ```

(`discopy/grammar/abstract.py`, `star = Ty("*")`)

> Let's use capitalised words for types
>
> ```suggestion
> String = Position >> Position
> ```

(`discopy/grammar/abstract.py`, `string = star >> star`)

> This is composition of lambda terms so it should be a method of `closed.TermBase`

(`discopy/grammar/abstract.py`, `def concat`)

> which paper? add a link to a reference

(`discopy/grammar/abstract.py`, the `Lexicon` docstring)

> this method looks like slop, recursing over function types should be done with a functor not a recursive call

(`discopy/grammar/abstract.py`, `def saturate`)

> this is a more general check it should go to biclosed.Functor and there should not be a need for overriding `__call__` here

(`discopy/grammar/abstract.py`, `Lexicon.__call__`)

- [x] Rename `ET`, `NP` to `Predicate`, `Quantifier` in the Montague test and the module example.
- [x] `Position = Ty("o")` and `String = Position >> Position` in place of `star` and `string`.
- [x] Composition of terms of function types as `closed.TermBase.then`, `concat` removed.
- [x] Link the paper from the `Lexicon` docstring.
- [x] Move the typing check of a mapped term to `biclosed.Functor`, drop `Lexicon.__call__`.
- [ ] `saturate` and `slots` through a functor rather than a recursion on the categorial type.


## Approved implementation, 2026-09-15

> ok great findings now implement your propositions and push to the PR, make sure to clean previous slop rather than add on top of it

- [ ] Refactor binding-aware maps, lexical boundaries, normalization and alpha-equivalence; replace slash saturation by a derivation interpretation; add grammar examples and validate the full change.
