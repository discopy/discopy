# TODO

> seed the todo

— Alexis, bridge session 2026-07-23, on his 25 review comments of 2026-07-22 (09:18–09:52Z) on
[the ribbon-colors PR (#360)](https://github.com/discopy/discopy/pull/360). Each point below is
seeded from those threads (file:line as reviewed); the comment text on GitHub is the instructing
prompt for its point. Frozen against head `8f4413b` (daydream6728's 2026-07-22 merge of main).

- [WIP] @fable-e6oj9d-2026-07-23 06:27 Introduce `config.COLOUR_DRAWING_ATTRIBUTES` and move the drawing attributes into it:
  `_ribbon` (backend.py:184), `_ribbon_color` (backend.py:190), and `cat.Ob`'s
  `min_right_margin`/`ribbon` class attrs (cat.py:133).
- [WIP] @fable-e6oj9d-2026-07-23 06:27 Drop the ribbon special case in the backend (`draw_ribbons`, backend.py:221) — draw as a
  coloured region instead.
- [WIP] @fable-e6oj9d-2026-07-23 06:27 Make `Ribbon` a subclass of `Colour` (balanced.py:45; `monoidal.Colour` exists unused).
- [WIP] @fable-e6oj9d-2026-07-23 06:27 Move the colour map out of the library into the tests as `auto_colour_ribbons` (UK
  spelling; balanced.py:106) and move the `RIBBON_COLORS` palette from config.py to the tests.
- [WIP] @fable-e6oj9d-2026-07-23 06:27 No auto random colours: default gray with dark-gray back (balanced.py:175 — replace the
  red/green/blue/yellow cycle in `to_braided(color="auto")`).
- [WIP] @fable-e6oj9d-2026-07-23 06:27 Evict ribbon/dual-rail content from `cat.py` (the `ribbon` attr, `_with_ribbon`, the
  ribbon docstring; cat.py:136) — it does not belong at that level of the hierarchy.
- [WIP] @fable-e6oj9d-2026-07-23 06:27 `darken` as a one-liner without `channels` (config.py).
- [WIP] @fable-e6oj9d-2026-07-23 06:27 Remove the `l`/`r` drawing-attr wrapper in pivotal.py:70 ("not needed") and the private
  `_with_ribbon` calls on rigid adjoints (rigid.py:206, 212).
- [WIP] @fable-e6oj9d-2026-07-23 06:27 Private-method cleanup in backend.py: no private undocumented methods (`_strand`,
  `_fill_strand_band`, `_half_circle_beziers`); dedupe `_half_circle` vs `_half_circle_beziers`
  and rename to `draw_half_circle`.
- [WIP] @fable-e6oj9d-2026-07-23 06:27 Finish splitting cup/cap into two short methods (`draw_dual_rail_cup` still handles
  "(or cap)" in one method).
- [WIP] @fable-e6oj9d-2026-07-23 06:27 Replace the old `_is_cup_or_cap` logic with `draw_as_cup`/`draw_as_cap` attributes (the
  attrs were never added even though `_is_cup_or_cap` is gone).
- [WIP] @fable-e6oj9d-2026-07-23 06:27 Rewrite the named functor helper as an anonymous `ar_map` function (the ribbon.py thread;
  the code now lives in `balanced.DualRail`).
- [WIP] @fable-e6oj9d-2026-07-23 06:27 CI: fix the 4 `test (3.14)` failures in test/drawing/drawing.py (`test_draw_sentence`,
  `test_draw_long_box_name`, `test_draw_wire_auto_margin`, `test_draw_long_latex_name`) — this
  PR's byte-exact SVG baseline comparison is not machine-independent for text-bearing diagrams
  (font-metric drift in text transforms/clip rects); regenerate those baselines on CI or
  normalize/tolerate text placement before comparing.
- [ ] BLOCKED (needs Alexis's look): five threads too ambiguous post-rebase to work — the
  twist-Bezier placement (`_bezier_subcurve` still module-level), the "draws two cups?" naming
  ask, two "not needed"/comment-alignment threads whose targets moved in the merge, and the
  cat.py screenshot "no" (`_with_drawing_attrs` was merely renamed `_with_ribbon`). Reply on
  the threads to unblock.
