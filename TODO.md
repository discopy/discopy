# TODO

> yeah let's go to the one-shot version, we could also feed all the files that the diffs depend on?

The review round on the harness cost: two agentic runs re-billed ~35k tokens
per turn with no cache discount on the route, ~1.1M tokens for two reviews.


  `.github/style-review/review.py`: assemble prompt.md, `STYLE.md`, the
  package-local files the changed files import (context), every changed
  file whole with line numbers and the diff, within a size budget that
  reports what it drops; one chat completion, findings to the same JSON

  `.github/workflows/style-review.yml`: call review.py, drop the harness
  install; `BASE_URL` + `/v1/chat/completions` is the one endpoint

  `.github/style-review/prompt.md`: rewrite for inline material instead of
  tool use
- [x] `CHANGELOG.md`:
  the entry says one request, with the imported files as context
- [ ] deleting this TODO fires `ready_for_review`: read the one-shot review
  of this PR's own Python files that it triggers
