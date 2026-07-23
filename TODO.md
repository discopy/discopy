# TODO

fix https://github.com/discopy/discopy/issues/462

Issue #462 "Use SVG for reproducible testing": extend the SVG-for-reproducibility
work (#435/#436, #457) from documentation static images to the drawing tests, so
that `draw_and_compare` checks byte-for-byte SVG equality instead of tolerance-based
raster (PNG) comparison, removing flakiness from antialiasing across environments.

- [x] Rewrite `draw_and_compare` to render SVG and compare bytes exactly (like `tikz_and_compare`), dropping the `tol` raster tolerance machinery.
- [x] Switch every `@draw_and_compare(...)` filename from `.png` to `.svg` and drop obsolete `tol=`/tolerance comments.
- [x] Regenerate the committed baselines under `test/drawing/imgs/` as SVG, removing the PNGs.
- [x] Update the `test/drawing/imgs/*.png` references in `README.md` to `.svg`.
- [x] Run `pflake8 discopy` and `coverage run -m pytest test/drawing`.
