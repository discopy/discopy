# TODO

> rename FreeMonoid to List and remove the functor hack where we used addition of tuples for python.Function https://github.com/discopy/discopy/issues/728

Design (from [#728](https://github.com/discopy/discopy/issues/728) and
[toumix's guidance on #727](https://github.com/discopy/discopy/pull/727#issuecomment-5574811520)):
`monoidal.FreeMonoid` is already "a sequence of atoms with tensor as concatenation";
make it `List`, a `NamedGeneric["generator_factory"]`, so `List[type]` is the free
monoid on Python's `type`. Then `python.Function.ob = List[type]` supports `@`, so the
`monoidal.Functor` object-map can fold with `@` uniformly instead of `+` on tuples.

- [ ] Rename `monoidal.FreeMonoid` to `List`, a `NamedGeneric["generator_factory"]`
      free monoid; keep `FreeMonoid` as a deprecated alias.
- [ ] Give `List` a tuple-friendly interface (`+`, `*`, indexing, `==`) so it is a
      drop-in for the `tuple[type, ...]` it replaces at the `python.Function` call sites.
- [ ] Point `python.function`/`multiplicative`/`additive` `Function.ob` at `List[type]`.
- [ ] Remove the tuple hack in `monoidal.Functor`/`cat.Functor` object mapping and fold
      the `Ty` branch with `@`.
- [ ] `uv run pflake8 discopy` and `uv run coverage run -m pytest` green.
- [ ] CHANGELOG entry, tests, then delete this file.
