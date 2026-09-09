# TODO

> fix this issue in a new PR: https://github.com/discopy/discopy/issues/751
> we first had https://github.com/discopy/discopy/pull/497 then we closed it in favor of https://github.com/discopy/discopy/pull/725 which got recently merged, however the latter is missing what the former PR did right, replacing the default region colour (as implemented in https://github.com/discopy/discopy/pull/354) from white to transparent.

- [WIP] @session_01Psss5f4oAymHHt8grgLjJn-2026-09-09 12:00 Make `monoidal.Colour` transparent by default: `config.TRANSPARENT`,
      `monoidal.transparent` in place of `monoidal.white`, `rigid.Wire`.
- [WIP] @session_01Psss5f4oAymHHt8grgLjJn-2026-09-09 12:10 Remove the "if white then transparent" special cases in
      `drawing.backend` (region cells, legend, neutral canvas, wire labels,
      spiders) and in `Drawing.add`, so an explicit white gets painted.
- [ ] Update the tests that assert white is the neutral background and add
      regressions for an explicitly white region and spider.
- [ ] Add a `CHANGELOG.md` entry, run `pflake8 discopy` and the full suite.
