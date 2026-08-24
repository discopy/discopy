# TODO

Round 4 (USER, verbatim, 2026-08-24): "just enable reasoning! I got
complaints that the style reviewer was dumb"

Rounds 2-3 fought to *disable* reasoning by default and only fell back to
enabling it when the gateway 400'd demanding it — which for
`stealth/ox-alpha` meant every single request paid for a failed attempt
first, and the reviews people actually saw ran with reasoning off, hence
"dumb". The direction is to flip the default: always let the model reason.

- [x] `ask()` always requests with reasoning enabled (no `reasoning`
  field sent at all, so every model uses its own default), dropping the
  `disable_reasoning` parameter and the retry-on-mandatory-reasoning path
  entirely — always use the 32,768 `max_tokens` budget rounds 2-3 added
  for the reasoning-enabled case, since that's now the only case
Follow-up (USER, verbatim, 2026-08-24): "nah it's fine we'll just switch
back to GLM for now" — the retry-then-429-rate-limit loop against
`stealth/ox-alpha`'s free shared pool wasn't worth chasing further.
`STYLE_REVIEW_MODEL` is reverted to `z-ai/glm-4.6` (its value before
today's switch, read from a pre-switch job log since GitHub doesn't keep
variable history). `CODEBASE_REVIEW_MODEL` is left untouched — it belongs
to an unrelated, unmerged `codebase-review.yml` on another branch, out of
scope here. `ask()` still always requests with reasoning enabled per round
4: that direction wasn't specific to `stealth/ox-alpha`, and GLM should
only benefit from it too.

- [ ] confirm on this PR's own `style-review` run — now against
  `z-ai/glm-4.6` — that findings look substantive, not empty
- [x] update the `CHANGELOG.md` entry for #611 to match the final
  behavior instead of the abandoned disable-by-default/retry design
