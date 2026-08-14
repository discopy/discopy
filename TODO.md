# TODO

Prompt, from USER on [#520](https://github.com/discopy/discopy/issues/520), verbatim:

> @toumix-agents fix this one please

on the issue *Broken bubble drawing*:

> Bubble drawing is broken when inside and outside don't have the same number of
> wires

## Points

- [x] Reproduce both examples and find where the bubble is lost
- [x] Stop `monoidal.Bubble` forcing `draw_as_square` when the wire counts differ
- [x] Pin the picture with a doctest in `discopy/drawing`, per USER's review
- [x] `pflake8 discopy` clean and the suite green
- [x] `CHANGELOG.md` entry
- [x] Ask USER about the first example, which this does not change — answered,
      it was never broken
- [x] Second review round: revert the `Drawing.bubble` half, restore
      `bubble-drawing.svg`, move the test to a doctest

## The bug

`monoidal.Bubble.__init__` read:

```python
self.draw_as_square = draw_as_square or not can_draw_as_bubble
```

so a bubble whose inside and outside have different numbers of wires was
**forced** into a square that `draw_as_square=False` could not turn off. That
mode draws the frame sides with zero width, so an uncoloured bubble came out
with no visible outline at all — the picture in the issue. It is now decided by
`draw_as_square` alone, which is what the parameter is for.

`can_draw_as_bubble` was also computed twice, the first result immediately
overwritten, and `draw_as_frame`'s second disjunct was unreachable because
`draw_as_square` had just been forced `True` on exactly that branch. Both gone.

## What the second review round removed

USER, on the two baselines: *"there shouldn't be any diff here"* and *"same here
this shouldn't have to change it looked correct before"*.

He was right about `Drawing.bubble`. The first version of this branch also
dropped its `wires_can_go_straight` fallback, which changed
`bubble-drawing.svg` from a rectangle to an ellipse and was not needed to fix
#520 — `monoidal.Bubble` alone does it. That hunk is reverted and
`bubble-drawing.svg` is byte-identical to `main` again.

`coloured-bubble.svg` **does** still move, and cannot not: it is a
`monoidal.Box.bubble` with three outer wires against two inner ones, i.e. the
exact path being fixed. The ten colour regions are unchanged and
`test_bubble_regions_are_distinct` still passes; what appears is the black
outline around the inner frame, which is the boundary the issue is about. Raised
on the thread for confirmation.

The regression test moved out of `test/drawing/drawing.py` into a gallery
doctest, per USER: *"this shouldn't be in test it should just be a doctest in
discopy/drawing"*. It pins the picture rather than the flags, which is the
right check here — nothing in #520 ever raised.

## Filed, not fixed here

- [#569](https://github.com/discopy/discopy/issues/569) — `draw_as_square=True`
  on mismatched wire counts still draws no outline. Out of scope: this PR stops
  that mode being forced on people, it does not repair the mode itself.

## Scope, settled by USER

The issue's **first** example, `Box('f', x, y ** 3).bubble()`, has equal wire
counts inside and out, so it took the bubble path before this change and takes
it after: it is not affected. USER confirmed on the PR, verbatim:

> Yes sorry if it wasn't clear: first example works good (same number of wires in
> and out) second examples was the broken one.

So the second example was the whole bug, and nothing further is owed here. No
issue is opened for the box width.
