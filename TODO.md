# TODO

> this should go to the docs not the changelog! also we should be 100% sure that none of our internal methods produce this junk

— @toumix on [#599](https://github.com/discopy/discopy/issues/599), 2026-08-19

Document the boxless-`Layer` contract raised in #599 in the docs, not the
changelog, having first verified no internal method produces one.

- [x] Audit every internal method for boxless-`Layer` production inside a
  `Diagram`: none found. Only `Layer.id` builds a boxless layer, a transient
  in `Layer.tensor` that `Layer.normalise` merges into the box it whiskers;
  every other `normalise=False` site maps one-to-one over an existing layer
  (`subs`, `dagger`, `lambdify`, `rigid.rotate`, `feedback.delay`), preserving
  its boxes. Confirmed empirically across `monoidal`, `symmetric`, `rigid`,
  `braided`, `markov`, `compact`, `frobenius`.
- [x] Document the contract on `Layer.id` (docstring only, no changelog entry).
- [x] Report the audit on #599 (comment 5349620334); PR opened as #601.
