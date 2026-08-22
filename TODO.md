# TODO

Prompt, from USER on [#520](https://github.com/discopy/discopy/issues/520), verbatim:

> @toumix-agents fix this one please

on the issue *Broken bubble drawing*:

> Bubble drawing is broken when inside and outside don't have the same number of
> wires

## Points

- [x] Reproduce both examples and find where the bubble is lost
- [x] Ask USER about the first example, which this does not change — answered,
      it was never broken
- [x] Second review round: revert the `Drawing.bubble` half, restore
      `bubble-drawing.svg`, move the test to a doctest
- [x] Third review round: keep `monoidal.Bubble`'s logic, fix the zero width
      issue in the drawing instead
- [x] Pin the picture with a doctest in `discopy/drawing`
- [x] `pflake8 discopy` clean and the suite green
- [x] `CHANGELOG.md` entry

## The bug

`monoidal.Bubble` draws a bubble as a square whenever its inside and outside
have different numbers of wires. `Drawing.bubble` then gave *every* square its
sides with `frame_boundary`, which the backend draws with **zero width**, and
hid the horizontal boundary along with them. Nothing was left to see: the
picture in the issue is an inner box and four dangling stubs.

Zero-width sides exist for `Equation` slots, where the colours of the regions
either side draw the edge in their place — which is why a *coloured* square
looks fine and only an uncoloured one looks broken. So the flag belongs to
`slot` and `frame`, which have those colours, not to every square bubble, which
has none. `Drawing.bubble` takes `frame_sides` for it and the two of them pass
it; `monoidal.Bubble` is untouched.

USER, verbatim, across three rounds:

> there shouldn't be any diff here

> this logic should stay here only the drawing should be fixed!

> it should do the opposite: keep the logic fix the zero width issue

## What each round removed

The first version fixed `monoidal.Bubble` instead, and also dropped
`wires_can_go_straight` in `Drawing.bubble`. Both are gone: `monoidal.py` is
untouched, and `bubble-drawing.svg` is byte-identical to `main`.

## Baselines

- `bubble-drawing.svg` — **unchanged**, as asked.
- `coloured-bubble.svg` — gains the boundary. USER on this round: *"ha good now
  the only diff is the boundary width, before it was drawn as a bubble with all
  the wires merged into a spider"*.
- `bubble-example.svg` — same, an explicit `draw_as_square=True` bubble.
- `bubble-uneven-wires.svg` — new, the regression doctest.

Every `Equation` and frame baseline is untouched, since slots keep their
zero-width sides; `test_bubble_boundary_is_visible` still passes unmodified.

## Closed by this PR

[#569](https://github.com/discopy/discopy/issues/569) — the zero-width frame
sides were the cause, so fixing them here closes it rather than leaving it open.
