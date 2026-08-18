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
- [x] Keep `id` and `whisker` in the semigroup: `Category.id` stays abstract
      there, so a subclass says what whiskering a colour gives without
      claiming an empty tensor exists.
- [x] `monoidal.Layer` inherits `ColouredSemigroup`, so `Layer.unit()` is gone
      rather than raising.
- [x] Say in `Layer`'s docstring that `Layer.id()` adjoins a unit which never
      appears inside a `Diagram`.
- [x] `test_Layer_has_no_unit`, checking that `Ty.unit()` still works and that
      `Layer.id()` is still neutral for `tensor`.
- [x] `CHANGELOG.md`: the new entry, and #438's line corrected — it claimed
      `Layer` is a `ColouredMonoid`.
- [x] `uv run pflake8 discopy` and `uv run pytest --skip-extra`.
