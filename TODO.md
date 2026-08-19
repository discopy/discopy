# TODO

USER on [#590](https://github.com/discopy/discopy/pull/590), choosing between
the three ways out of the `ColouredSemigroup` design, verbatim:

> 2. but we would need either to make the type signature more general or just
> stop respecting it for layers in particular

with option 2 being their own earlier suggestion, also verbatim:

> another option would be to relax the type signature for the unit so that it
> can return something outside the class, i.e. `Layer.unit(colour)` could
> return `Ty.unit(colour)`

- [x] `ColouredMonoid.unit(colour)` returns the identity on a colour, typed
      `C0 | C1` — the general signature rather than an unrespected one.
- [x] `ColouredMonoid.id` becomes the primitive, `unit` delegates to it, so a
      class that already has coloured identities (`Ty`) needs no override.
- [x] `Layer.unit(colour)` returns the empty type, which `tensor` accepts on
      either side. `Layer.unit()` no longer raises, which is what #568 reported.
- [x] `ColouredSemigroup` is gone: `Layer` is a `ColouredMonoid` again.
- [x] `test_Layer_unit` replaces `test_Layer_has_no_unit`, covering the
      coloured units on both sides and the wrong colour raising.
- [x] `CHANGELOG.md`: the semigroup entry dropped, #438's line put back, and a
      `Changed` entry for the widened signature.
- [x] `uv run pflake8 discopy` and `uv run pytest --skip-extra`.

USER again on #590, once the empty-layer bug came out of @daydream6728's
question, verbatim:

> let's make this PR fix it instead of adding a Semigroup abstract class:
> Layer.id should raise and there should never be empty plumbing created when
> whispering layers

- [x] `Layer.id` raises: there is no identity layer.
- [x] `Layer.whisker` leaves a type as a type, and `tensor` and `__rmatmul__`
      merge it into the boundary, so whiskering builds no empty plumbing.
- [x] `test_Layer_has_no_identity` covers both, and that whiskering leaves no
      empty type behind.
- [x] `Layer(Ty(), normalise=False)` still builds the boxless layer. Reported
      on the PR and ruled by USER, verbatim:

      > Yes it's fine if the user explicitly wants to be stupid we should let
      > them

      So `Diagram.__init__` is left alone and #599 closes with this PR.
