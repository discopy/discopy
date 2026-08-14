# TODO

Prompt, from USER on [#520](https://github.com/discopy/discopy/issues/520), verbatim:

> @toumix-agents fix this one please

on the issue *Broken bubble drawing*:

> Bubble drawing is broken when inside and outside don't have the same number of
> wires

## Points

- [x] Reproduce both examples and find where the bubble is lost
- [x] Stop `monoidal.Bubble` forcing `draw_as_square` when the wire counts differ
- [x] Stop `Drawing.bubble` falling back to a frame opening for the same reason
- [x] Regenerate the two drawing baselines this corrects
- [x] Regression test pinning that the bubble stays a bubble
- [x] `pflake8 discopy` clean and the suite green
- [x] `CHANGELOG.md` entry
- [x] Ask USER about the first example, which this does not change — answered,
      it was never broken

## The bug

`monoidal.Bubble.__init__` read:

```python
self.draw_as_square = draw_as_square or not can_draw_as_bubble
```

so a bubble whose inside and outside have different numbers of wires was
**forced** into a square — `draw_as_square=False` could not turn it off, and
`f.bubble(dom=x ** 3, cod=y)` silently stopped being a bubble. `Drawing.bubble`
did the same thing one level down, swapping the bubble opening for a frame
opening whenever `(len(dom), len(cod)) != (len(arg_dom), len(arg_cod))`.

Both are now decided by `draw_as_square` alone, which is what the parameter is
for. The wires bend through the opening and closing, which `Drawing.bubble`
already supported: the two `if len(dom) == len(arg_dom)` blocks below it only
*straighten* the wires when the counts happen to match, per side.

`can_draw_as_bubble` was also computed twice, the first result immediately
overwritten, and `draw_as_frame`'s second disjunct was unreachable because
`draw_as_square` had just been forced `True` on exactly that branch. Both gone.

## Scope, settled by USER

The issue's **first** example, `Box('f', x, y ** 3).bubble()`, has equal wire
counts inside and out, so it took the bubble path before this change and takes
it after: it is not affected. USER confirmed on the PR, verbatim:

> Yes sorry if it wasn't clear: first example works good (same number of wires in
> and out) second examples was the broken one.

So the second example was the whole bug, and nothing further is owed here. No
issue is opened for the box width.
