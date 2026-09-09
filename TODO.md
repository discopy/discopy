# TODO

> FreeMonoid didn't exist in the previous release so the rename doesn't make sense

> Ok we had a chat with @daydream6728 and in fact we need to keep FreeMonoid because it's actually a free *coloured* monoid.
>
> What we need is a `List` type which is really a `Monoid` i.e. with the unit type as colours.
>
> This means we have three levels: Ty has arbitrary colours and boxes, List has a single colour and arbitrary boxes, Nat has a single colour and a single box (hence by Eckmann Hilton it's also commutative)

> another realisation: Ty is the only subclass of FreeMonoid so we can just remove FreeMonoid altogether and make Ty a subclass of cat.Ob, cat.FreeCategory and abc.Monoid directly

toumix on https://github.com/discopy/discopy/pull/747#pullrequestreview-5153471709

- [WIP] @session_01Uhnqw8SbstUbervfVzscnT-2026-09-09 11:55 `Ty(cat.Ob, cat.FreeCategory, abc.ColouredMonoid)` with `generator_factory = Wire`: the free coloured monoid is `Ty` itself, no class in between, no `FreeMonoid` and no deprecated alias since it never shipped
- [WIP] @session_01Uhnqw8SbstUbervfVzscnT-2026-09-09 11:55 `List[X](abc.Monoid, NamedGeneric['generator_factory'])`: a tuple of atoms with concatenation as `tensor`, the unit as its only colour, `len`, slicing, iteration, `**`, `cast`; `python.Function.ob = List[type]`
- [WIP] @session_01Uhnqw8SbstUbervfVzscnT-2026-09-09 11:55 `CHANGELOG.md` says `List` is added and `python.Function.ob` changed, not that anything is renamed; tests, `pflake8`, the full suite, the docs and the notebooks
