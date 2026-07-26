# TODO — Address the final review on PR #470

User prompt, verbatim:

> last round of comments [discopy/discopy#470](https://github.com/discopy/discopy/pull/470)

## Checklist

- [x] @codex-2026-07-24 21:32 Address all four unresolved review threads: rename the save helper,
      replace regex SVG filtering with structured XML normalization, restore
      the feedback image name, document baseline deletion, and verify fully.
- [x] @claude-2026-07-24 16:42 Address toumix's follow-up comments on the
      previous fix: unjustified `x0, y0, z0` rename and duplicate `x, y, m`
      assignment in feedback.py, and the unused `tol` parameter on
      `Hypergraph.draw`; merge main into the branch.
- [x] @claude-2026-07-24 17:22 Make `utils.text_width` system-independent: it measures glyph extents
      with `TextPath` so baselines drawn on different freetype versions
      drift, e.g. `long-box-name.svg` (218.88pt vs 218.952pt).
- [x] @claude-2026-07-24 17:39 Count LaTeX commands as single symbols in
      `text_width` instead of summing their source characters, and compare
      SVG coordinates up to a small tolerance for rounding errors.
- [x] @claude-2026-07-24 18:16 Keep main's full mathtext rendering for
      `$...$` labels in `text_width`, the metric table only applies to
      plain text.
- [x] @claude-2026-07-24 18:25 Drop the metric table and restore main's
      original `text_width`, the SVG tolerance absorbs the cross-system
      drift so comparisons stop raising.
- [x] @claude-2026-07-24 20:20 Fix the CI coverage gate: cover the raster
      and byte comparison branches of `compare_drawing` that no committed
      baseline exercises.
- [x] @claude-2026-07-25 08:45 Review 4778986121 by toumix-agents proposed
      moving the baseline comparison out of `draw(path=...)`. First approval
      check was VOID (not authored by Alexis); now approved by Alexis's PR
      comment (2026-07-25), frozen verbatim:
      > I agree this would be annoying for users just running scripts.
      >
      > Let's add a new argument doctest argument to draw and update all the
      > "path=" in the doctests with "doctest=" so that the diff stays small.
      > The path argument recovers its previous behaviour.

- [x] @claude-2026-07-26 08:30 Address Alexis's review 4781256804: drop the
      `replace` argument (only `OVERRIDE_DOCTEST_IMAGES`, renamed, with
      missing baselines saved), retire the docs-static bot job, remove
      `normalize_svg` in favour of telling matplotlib not to emit volatile
      metadata, use `tol` for GIF comparison, rename the config key to
      `plt_tol` and drop its comments, keep absolute URLs for README images,
      and fix the contributing guidelines (images are committed like tests).

## Mathematical description

Drawing comparison is equality after a canonical projection of an image:
volatile platform-specific SVG representation is discarded structurally while
element order and text content are preserved. A save operation either creates
this baseline, explicitly replaces it, or compares the new canonical image
against it.
