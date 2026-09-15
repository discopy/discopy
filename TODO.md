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

- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-15 16:35 Rename `ET`, `NP` to `Predicate`, `Quantifier` in the Montague test and the module example.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-15 16:35 `Position = Ty("o")` and `String = Position >> Position` in place of `star` and `string`.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-15 16:35 Composition of terms of function types as `closed.TermBase.then`, `concat` removed.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-15 16:35 Link the paper from the `Lexicon` docstring.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-15 16:35 Move the typing check of a mapped term to `biclosed.Functor`, drop `Lexicon.__call__`.
- [ ] `saturate` and `slots` through a functor rather than a recursion on the categorial type.
