why is this one not transparent? (docs/_static/balanced/ribbon_twist.svg)
actually the wire labels are a deal breaker: if they become invisible on dark background then we cannot ship this (docs/_static/balanced/twist.svg)
@toumix-agents let's go back for another round!
[#497 review comments by toumix, 2026-08-14]

- [x] wire labels readable on dark for real: inject a `prefers-color-scheme: dark` style block into saved SVGs that turns the tagged black strokes and labels white in dark-mode browsers, keeping the white halos as the static fallback
- [x] answer why ribbon_twist is not transparent: the white-erase fallback of #521 — explain on the thread, revisit once regions are filled between their boundaries
- [x] regenerate all baselines, run lint and the full test suite, merge main in (a4a45b9)
