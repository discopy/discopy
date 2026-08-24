# TODO

> yeah let's go to the one-shot version, we could also feed all the files that the diffs depend on?

The review round on the harness cost: two agentic runs re-billed ~35k tokens
per turn with no cache discount on the route, ~1.1M tokens for two reviews.

- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 08:15
  `.github/style-review/review.py`: assemble prompt.md, `STYLE.md`, the
  package-local files the changed files import (context), every changed
  file whole with line numbers and the diff, within a size budget that
  reports what it drops; one chat completion, findings to the same JSON
- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 08:15
  `.github/workflows/style-review.yml`: call review.py, drop the harness
  install; `BASE_URL` + `/v1/chat/completions` is the one endpoint
- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 08:15
  `.github/style-review/prompt.md`: rewrite for inline material instead of
  tool use
- [WIP] @session_01JFJANWnm5ZdrfujFQmgrff-2026-08-24 08:15 `CHANGELOG.md`:
  the entry says one request, with the imported files as context
- [ ] deleting this TODO fires `ready_for_review`: read the one-shot review
  of this PR's own Python files that it triggers
