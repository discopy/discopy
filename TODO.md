# TODO

> fix issue https://github.com/discopy/discopy/issues/453
> there is already a PR for this at https://github.com/discopy/discopy/pull/497, but i think we need to start over with a cleaner solution which I suggested at some point in my review: instead of using outlines to make diagrams readable on both light and dark backgrounds, we should use media queries in SVG to decide colours
> start over from a fresh branch off main and implement this adaptative rendering using media queries. a single SVG file should be readable at all times by using light/dark theme settings.
> push this branch but don't create another PR just yet.

- [WIP @session_01X78P89HnF1rQfdrF4FZimK-2026-09-07 12:30] Save Matplotlib SVGs on a transparent canvas: transparent figure and
  boundary polygon, while raster formats keep their white background since
  they cannot adapt to the page behind them.
- [WIP @session_01X78P89HnF1rQfdrF4FZimK-2026-09-07 12:30] Inject a `prefers-color-scheme: dark` style block into every saved SVG,
  file or in-memory buffer, turning the tagged black elements white on a dark
  page.
- [WIP @session_01X78P89HnF1rQfdrF4FZimK-2026-09-07 12:30] Tag the elements drawn on the neutral canvas: wires, braids, wire
  labels, spiders and their labels, control dots. Elements over coloured
  regions or on box interiors keep their static colours.
- [WIP @session_01X78P89HnF1rQfdrF4FZimK-2026-09-07 12:30] Draw white spiders (e.g. equation symbols) unfilled, so they leave no
  white patch on a non-white page.
- [WIP @session_01X78P89HnF1rQfdrF4FZimK-2026-09-07 12:30] Regenerate the SVG baselines and keep the raster and TikZ baselines
  unchanged.
- [WIP @session_01X78P89HnF1rQfdrF4FZimK-2026-09-07 12:30] Test the style injection and the tagging, add a changelog entry, run
  `pflake8` and the full test suite.
