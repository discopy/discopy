# TODO

Prompt, from USER on [#560](https://github.com/discopy/discopy/issues/560), verbatim:

> B

i.e. option **B** of the two offered on that issue:

> **B** — `biclosed.py:279,463`, `curry` default `False` → `True`. matches `abc` and
> `rigid`, i.e. three of four modules agree and `biclosed` was the odd one. touches
> `biclosed` and everything currying through it.

## Points

- [x] Establish what `left` actually selects, since B as written flips `curry` only
- [x] Flip the `biclosed` defaults to `left=True`: `Diagram.curry`, `Diagram.ev`,
      `Diagram.uncurry`, `CMap.curry`, `CMap.uncurry`
- [x] Keep the right/`Under` cases in the doctests by passing `left=False` explicitly
- [x] Regenerate the drawing baselines the flip changes
- [x] `pflake8 discopy` clean and the suite green
- [x] `CHANGELOG.md` entry — this changes a public default
- [x] Report the review cost and the closed-lane collisions on the PR

## What B turned out to mean

B as written flips `curry` only, which would have left `biclosed` itself
contradicting `ev` — the exact bug #560 is about, moved one module along. `left`
is not a free convention: it selects `Over` (`<<`) over `Under` (`>>`), so `curry`,
`ev` and `uncurry` have to move together or the round-trip stops composing. All
five defaults are flipped here, which is what "matches `abc` and `rigid`" means.

`discopy.closed` is unaffected in expressivity because `<<` collapses onto `>>`
there — both sides round-trip, `left` only picks which argument is curried.

## Three call sites relied on the old default and are now explicit

Pinned to `left=False` rather than left to follow the default, since each is the
right/`Under` case and each has a sibling that already passes `left=True`:

- `closed.Abstraction.eval` — **load-bearing**: with the flipped default,
  `\x. \f. f(x)` evaluated to `(X >> Y) >> (X >> Y)` instead of
  `X >> ((X >> Y) >> Y)`, i.e. the wrong type and the wrong denotation. Caught by
  the `closed.Ty` doctest.
- `grammar.categorial.Diagram.bc` — backward composition, the mirror of `fc`
  which already passed `left=True`.
- the `CMap.curry` right-hand doctest, named `biclosed-curry-right`.

## Filed, not fixed here

- [#562](https://github.com/discopy/discopy/issues/562) — the two branches of
  `closed.Abstraction.eval` curry on opposite sides, which is why the default was
  load-bearing above.
