# TODO

USER's 🚀 on the body of
[#568](https://github.com/discopy/discopy/issues/568), which reads verbatim:

> A layer is a coloured **semigroup**: it has an associative product and no unit.
> `Ty` is the genuine monoid — and there `then` *is* `tensor`, which is not the case
> for `Layer`.
>
> The honest fix is a `ColouredSemigroup` base carrying the product, with
> `ColouredMonoid` extending it with `unit`. `Layer` would inherit the former.

- [x] Split `abc.ColouredMonoid` into `ColouredSemigroup` (the product, `then`,
      `whisker`, `@` and its mirror) and `ColouredMonoid` (`unit` and `id`).
- [x] Keep `whisker` in the semigroup, and do **not** inherit `Category`
      there: a category has an identity on every object and a semigroup has
      nothing to send them to, so inheriting it would make `Category.id`
      abstract on a class that has no unit (daydream6728's review).
- [x] `monoidal.Layer` inherits `ColouredSemigroup`, so `Layer.unit()` is gone
      rather than raising.
- [x] Say in `Layer`'s docstring that `Layer.id()` adjoins a unit which never
      appears inside a `Diagram`.
- [x] `test_Layer_has_no_unit`, checking that `Ty.unit()` still works and that
      `Layer.id()` is still neutral for `tensor`. A second test pinning that the
      semigroup is not a `Category` was added and then removed on review: it
      asserted `issubclass`, i.e. Python rather than DisCoPy (daydream6728).
- [x] `CHANGELOG.md`: the new entry, and #438's line corrected — it claimed
      `Layer` is a `ColouredMonoid`.
- [x] `uv run pflake8 discopy` and `uv run pytest --skip-extra`.
