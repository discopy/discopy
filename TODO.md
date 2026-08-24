# TODO

Round 4 (USER, verbatim, 2026-08-24): "just enable reasoning! I got
complaints that the style reviewer was dumb"

Rounds 2-3 fought to *disable* reasoning by default and only fell back to
enabling it when the gateway 400'd demanding it — which for
`stealth/ox-alpha` meant every single request paid for a failed attempt
first, and the reviews people actually saw ran with reasoning off, hence
"dumb". The direction is to flip the default: always let the model reason.

- [WIP] @claude-issue611-2026-08-24 16:52 `ask()` always requests with
  reasoning enabled (no `reasoning` field sent at all, so every model
  uses its own default), dropping the `disable_reasoning` parameter and
  the retry-on-mandatory-reasoning path entirely — always use the 32,768
  `max_tokens` budget rounds 2-3 added for the reasoning-enabled case,
  since that's now the only case
- [ ] confirm on this PR's own `style-review` run that findings look
  substantive, not empty
- [ ] update the `CHANGELOG.md` entry for #611 to match the final
  behavior instead of the abandoned disable-by-default/retry design
