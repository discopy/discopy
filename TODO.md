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

- [x] Merge `origin/main` into this branch (append-only, never rebase)
  and resolve the conflicts: `closed.py` moved under #532/#560 —
  `curry`/`ev`/`uncurry` now default `left=True`, `CMap` is
  parameterised as `NamedGeneric["category"]`. Merge commit `cae76aca`;
  `closed.py`'s conflict was `Pack`/`Unpack` (this branch) alongside
  `Permutation`/`Swap` (main) — kept both. `CHANGELOG.md`'s conflict was
  two independent `[Unreleased]` bullets — kept both, using main's
  updated wording for the one entry both sides touched.
- [x] Rerun `uv run pflake8 discopy` and `uv run coverage run -m pytest`
  on the merged tree and fix what the merge broke, regenerating any
  invalidated baselines (`docs/_static/closed/catgpt-block.svg`).
  Clean: 721 passed, 51 skipped (missing quantum/grammar extras, torch
  download is blocked in this sandbox), 0 failed; `catgpt-block.svg`'s
  doctest still matches the committed baseline, no regeneration needed.
  Fixed one merge-fallout warning: `Product` was left subclassing the
  deprecated `biclosed.Ob` alias instead of `biclosed.Wire` (#566's
  rename, landed on `main` after this branch forked).
- [ ] Re-read the open review threads against the merged state and
  answer or fix each, then delete this `TODO.md` to hand the pull
  request back to review.
