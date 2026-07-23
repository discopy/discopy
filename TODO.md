# TODO

> seed the todo

— Alexis, bridge session 2026-07-23, on his 25 review comments of 2026-07-22 (09:18–09:52Z) on
[the ribbon-colors PR (#360)](https://github.com/discopy/discopy/pull/360). Each point below is
seeded from those threads (file:line as reviewed); the comment text on GitHub is the instructing
prompt for its point. Frozen against head `8f4413b` (daydream6728's 2026-07-22 merge of main).

- [x] Introduce `config.COLOUR_DRAWING_ATTRIBUTES` and move the drawing attributes into it:
  `_ribbon` (backend.py:184), `_ribbon_color` (backend.py:190), and `cat.Ob`'s
  `min_right_margin`/`ribbon` class attrs (cat.py:133). Done: the dict holds
  `min_right_margin` and an anonymous `ribbon` function; `Ty.to_drawing` applies it like the
  BOX and WIRE dicts; the backend helpers and the `cat.Ob` class attrs are gone.
- [x] Drop the ribbon special case in the backend (`draw_ribbons`, backend.py:221) — draw as a
  coloured region instead. Done: a ribbon is now the `monoidal.Colour` region between its two
  rails (the rails' facing sides are the shared `Ribbon`), so the straight rails are filled by
  the ordinary `draw_regions` and `draw_ribbons` is deleted. Region names in `config.COLORS`
  now resolve to their hexcodes, as box colours always did (TikZ still draws no regions).
- [x] Make `Ribbon` a subclass of `Colour` (balanced.py:45). Done: a frozen dataclass adding
  `width`, so the colour region also carries how far apart the rails are drawn
  (`Ty.wire_offsets`), replacing `set_rail_margins` and the negative-margin trick.
- [x] Move the colour map out of the library into the tests as `auto_colour_ribbons` (UK
  spelling; balanced.py:106) and move the `RIBBON_COLORS` palette from config.py to the tests.
- [x] No auto random colours: default gray with dark-gray back (balanced.py:175 — the
  red/green/blue/yellow cycle is gone, `to_braided`/`to_ribbons` take `colour="gray"`).
- [x] Evict ribbon/dual-rail content from `cat.py` (the `ribbon` attr, `_with_ribbon`, the
  ribbon docstring; cat.py:136) — `cat.Ob` is back to its pre-drawing state.
- [x] `darken` as a one-liner without `channels` (config.py).
- [x] Remove the `l`/`r` drawing-attr wrapper in pivotal.py:70 ("not needed") and the private
  `_with_ribbon` calls on rigid adjoints (rigid.py:206, 212). Done: `rigid.Ob.r` already swaps
  `dom` and `cod`, which is exactly what keeps the two rails one region under the adjoint.
- [x] Private-method cleanup in backend.py: `braid_strand`, `fill_strand_band`, `fill_fold`
  and `half_circle_beziers` are now public and documented; `_half_circle` and
  `_half_circle_beziers` are deduped into `half_circle_beziers` + `draw_half_circle`.
- [x] Finish splitting cup/cap into two short methods: `draw_dual_rail_cup` and
  `draw_dual_rail_cap` with symmetric bodies, dispatched by `draw_as_dual_rail_cup`/`_cap`.
- [x] Replace the old `_is_cup_or_cap` logic with `draw_as_cup`/`draw_as_cap` attributes,
  initialised in `rigid.Cup`/`Cap` and folded into `draw_as_wires` in config.
- [x] Rewrite the named functor helper as an anonymous `ar_map` function — already satisfied
  post-rebase: `balanced.DualRail` passes `ar_map=lambda f: f.name`.
- [x] CI: fix the `test (3.14)` failures in test/drawing/drawing.py — root cause was
  `utils.text_width` measuring glyph outlines to 3 decimals, so sub-pixel font-metric drift
  across environments shifted whole figures. It now rounds up to the next 1/16 inch, which
  the drift cannot cross (this sandbox disagreed with the committed baselines exactly like
  CI's 3.14 and now reproduces `alice-loves-bob.svg` byte-for-byte). To be confirmed on CI;
  Python 3.14 could not be installed in this sandbox.
- [x] BLOCKED threads reassessed: the region redesign swept them up — the twist fills live in
  `draw_dual_rail_twist`, the "draws two cups?" method is split and documented, the cat.py
  "no" and the pivotal "not needed" wrappers are deleted with `_with_ribbon` itself. Alexis to
  double-check when resolving the threads on the PR.

> no no no it’s not correct anymore! the two braids of a twist should be the same orientation
> so they don’t cancel each other out
> there still is a margin issue with these nested cups they shouldn’t touch the boundary of
> the image

— Alexis, in-session 2026-07-23. Both traced to daydream6728's merge of main (`16d078c`),
whose conflict resolution silently dropped a slice of the drawing code:

- [x] Twist braids of the same orientation: restored the same-handedness crossing logic in
  `draw_dual_rail_twist` (the NW–SE strand goes under at both crossings, verified
  programmatically), lost in the merge.
- [x] Margin around dual rail folds: restored `Drawing.frame_dual_rail`,
  `config.RIBBON_FOLD_DEPTH` and the fold-depth capping (`Backend.fold_depths`) that
  flattens wide cups into ellipses, all lost in the same merge.
- [x] Also lost there: the ribbon-aware wire spacing in `reposition_box_dom`/`reposition_box_cod`
  (`then`'s "recover legacy behaviour" step), which spread nested rails back to unit gaps and
  made the inner fold fat. They now use the one `Ty.wire_offsets`.
