# TODO

Prompt (USER, verbatim, 2026-08-24): "seems like switching from z glm to
stealth/ox-alpha on openrouter didn't work out of the box
https://openrouter.ai/stealth/ox-alpha?view=api help me solve this issue
https://github.com/discopy/discopy/issues/611"

Issue [#611](https://github.com/discopy/discopy/issues/611): the style-review
job 400s on `stealth/ox-alpha`, and `review.py` swallows the gateway's error
body so the cause can't be read from the log. Two things are wrong per the
issue and its own investigation comments:

1. `ask()` lets `urllib.error.HTTPError` propagate without reading it.
2. `assemble()` budgets raw file text against `BUDGET`, but `numbered()`,
   the per-file headers, `prompt.md` and `STYLE.md` are added on top,
   uncounted, so the assembled prompt can exceed `BUDGET` (425,159 chars on
   #546 against a 400,000 budget).

- [x] catch `HTTPError` in `ask()`, print the response body
  (`error.read()`) before re-raising, so the next 400 is diagnosable from
  the job log
- [x] budget the actually-assembled blocks (numbered text, per-file
  headers, `prompt.md`, `STYLE.md`) against `BUDGET`, not the raw file text
- [x] add a `CHANGELOG.md` entry under `Fixed`
