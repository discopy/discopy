# TODO

Prompt, from USER on [#560](https://github.com/discopy/discopy/issues/560), verbatim:

> B

i.e. option **B** of the two offered on that issue:

> **B** — `biclosed.py:279,463`, `curry` default `False` → `True`. matches `abc` and
> `rigid`, i.e. three of four modules agree and `biclosed` was the odd one. touches
> `biclosed` and everything currying through it.

## Points

- [ ] Establish what `left` actually selects, since B as written flips `curry` only
- [ ] Flip the `biclosed` defaults to `left=True`: `Diagram.curry`, `Diagram.ev`,
      `Diagram.uncurry`, `CMap.curry`, `CMap.uncurry`
- [ ] Keep the right/`Under` cases in the doctests by passing `left=False` explicitly
- [ ] Regenerate the drawing baselines the flip changes
- [ ] `pflake8 discopy` clean and the suite green
- [ ] `CHANGELOG.md` entry — this changes a public default
- [ ] Report the review cost and the closed-lane collisions on the PR
