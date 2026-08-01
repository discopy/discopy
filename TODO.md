# TODO

> On discopy/discopy: file an issue titled "draw_discard shadows the layer index: KeyError drawing a discard on more than one wire", then open a PR fixing it.
> Repro: `from discopy import monoidal; x = monoidal.Ty('x'); monoidal.Box('discard', x ** 2, monoidal.Ty(), draw_as_discards=True).draw()` raises `KeyError: Node('box_dom', i=1, j=2, x=x)`. One wire works; two or more always fail.
> Cause: in `discopy/drawing/backend.py`, `Backend.draw_discard` does `box, j = node.box, node.j` and then `for j in range(3):`, rebinding the layer index. The inner loop leaves `j == 2`, so the next wire looks up a node that isn't in `positions`.
> Fix — rename the inner variable:
>
> ```
> -            for j in range(3):
> ```
>
> `-                source = (left + .1 * j, height - .1 * j)`
> `-                target = (right - .1 * j, height - .1 * j)`
> `+            for k in range(3):`
> `+                source = (left + .1 * k, height - .1 * k)`
> `+                target = (right - .1 * k, height - .1 * k)`
> Verified: with this patch the repro draws at n = 1, 2, 3, 5, and `at_time(3).draw()` works again in optyx ([rel-int/optyx#15](https://github.com/rel-int/optyx/issues/15)). Link the issue and PR back to that PR.

- [WIP] @07455c9b-2026-08-01 09:00 File the issue on `discopy/discopy` (done: [#513](https://github.com/discopy/discopy/issues/513))
- [WIP] @07455c9b-2026-08-01 09:00 Rename the inner loop variable in `Backend.draw_discard`
- [WIP] @07455c9b-2026-08-01 09:00 Add a regression test drawing a discard on more than one wire
- [WIP] @07455c9b-2026-08-01 09:00 Add a `CHANGELOG.md` entry under `[Unreleased]` / `Fixed`
- [WIP] @07455c9b-2026-08-01 09:00 Run `pflake8 discopy` and the test suite
- [WIP] @07455c9b-2026-08-01 09:00 Open the PR, linking it and the issue back to rel-int/optyx#15
