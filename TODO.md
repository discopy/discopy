# TODO.md

USER, live session of 2026-08-29, verbatim:

> one thing that should help is writing the combinators as lambda terms rather than point free diagrams, we have a PR open for that already

> no it's this one https://github.com/discopy/discopy/pull/489
> from_callable works fine for first-order functions / kappa calculus
> Term works fine for higher-order functions but doesn't have product types yet

> yep add to the queue Evening will deal with it

The queue item is this branch's re-merge: #687 (hybrid LLM-GoNI) picks
these terms as the diagram language the LLM writes, and #443's last box
waits here too.

- [WIP] @evening-2026-09-01T00:30Z Merge `origin/main` into this branch (append-only, never rebase)
  and resolve the conflicts: `closed.py` moved under #532/#560 —
  `curry`/`ev`/`uncurry` now default `left=True`, `CMap` is
  parameterised as `NamedGeneric["category"]`.
- [ ] Rerun `uv run pflake8 discopy` and `uv run coverage run -m pytest`
  on the merged tree and fix what the merge broke, regenerating any
  invalidated baselines (`docs/_static/closed/catgpt-block.svg`).
- [ ] Re-read the open review threads against the merged state and
  answer or fix each, then delete this `TODO.md` to hand the pull
  request back to review.
