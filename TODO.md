# TODO

> Add let statements and not-strictly-associative products to DisCoPy terms https://github.com/discopy/discopy/issues/370
> use the term for the CatGPT benchmark as example https://github.com/discopy/discopy/issues/458

- [ ] Add a `Product` type constructor to `discopy.closed` with `Ty.product`, `Ty.__mul__`, `Ty.is_product`, `Ty.factors` and strictification in `Functor`.
- [ ] Add `Pack` and `Unpack` boxes for the canonical isomorphism between a product type and the tensor of its factors.
- [ ] Add `Tuple`, `Projection` and `Let` terms with `eval` into any closed markov category, plus the `let` introspection helper of issue #370.
- [ ] Extend `Substitution` to the new terms and fix the bugs found in it: constants substituted to `None`, the `left` flag dropped on applications, infinite recursion on abstractions.
- [ ] Replace the non-deterministic `list(set(...))` in `Application.__check_dom__` with a deterministic deduplication.
- [ ] Use a term for the CatGPT transformer block of issue #458 as the documentation example, with a drawing test.
- [ ] Run `pflake8` and the full test suite, commit any regenerated documentation images.
