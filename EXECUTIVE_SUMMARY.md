# Evening report — 2026-07-20

First round of the evening prompt. This file is meant for review, not for
merging into `main`. Everything below was verified against the repository
at `add8d65` (current `main`) and the live GitHub state tonight.

## TL;DR

The library is healthy (CI green, flake8 clean, only two TODOs in the whole
source tree) but the project has three pressure points:

1. **Two visual regressions shipped to `main` this week** (#426, #427),
   both invisible to CI because generated images are never compared —
   exactly the blind spot that #418/#419 are meant to close.
2. **A release is overdue**: `main` is 45 commits ahead of 1.2.2
   (published 2025-12-19), including speedups, 2-category syntax and
   drawing features that users on PyPI don't have.
3. **The PR queue is the bottleneck**: 32 open PRs (13 ready for review,
   19 drafts), with external contributors waiting since January.

## Verified findings

### P0 — regressions on `main`

- **#426 Bubble drawing broken** by the `hypergraph_factory` refactor
  (#379): top and bottom frame boundaries missing in
  `frame-opening.png`, `frame-closing.png`, `bubble-drawing.png`.
- **#427 Feedback docstring drawing broken** by the `Ty.name` cleanup
  (#421): the functor at `discopy/feedback.py:67` maps `x.name` into
  stream types and the lazy-name change altered what delayed objects
  report. The broken `slide-unroll.png` only surfaced via #419's CI-drawn
  images.

Both are small, well-localised, and good candidates for the next evening
round. Neither is caught by the test suite — which is the strongest
possible argument for merging #419 (generate `docs/_static` in CI) soon.

### P0 — the README's own example is silently broken (#395)

Reproduced tonight on `main`:

```python
Recipe.swap_factory = CookingSwap   # what the README tells users to do
type(Recipe.swap(x, y))             # -> Swap, not CookingSwap
```

`swap_factory` is read nowhere; all swap machinery goes through
`braid_factory`. Any user following the front-page cooking example gets a
silently ignored attribute, and custom categories built this way fail
`assert_isinstance` when hypergraph round-trips introduce swaps. The fix
is a design decision (alias `swap_factory` to `braid_factory` in symmetric
modules vs. renaming) — small either way once you pick.

### P1 — CI blind spot for documentation images

#418/#419/#383 all circle the same problem: doctests draw images, then
throw them away, so drawing regressions ship invisibly (see P0 above).
#419 is ready for review. It needs a repo-settings decision from you
(workflow write permission or a PAT so CI can commit images back).

### P1 — release 1.3.0

45 commits since 1.2.2 including: `Ty` construction speedup (#420),
2-category syntax and drawing (#354, #355), colour legends (#357), long
wire label margins (#365), hypergraph hash fix (#387), pickle
compatibility fix (#417). Suggested order: fix #426/#427 first so the
release doesn't ship known-broken drawings, then tag.

### P2 — PR queue (32 open, oldest ready PR from January)

Ready for review, external contributors first:

| PR | Author | What |
|----|--------|------|
| #305 | colltoaction | Fixpoint recursion fix — **ready since January** |
| #389 | 0x0f0f0f | Move benchmark to Modal, smaller full sweep |
| #406 | salvatomm | Vectorization of neural + GRUs |
| #402 | giodefelice | einsum through CMap as the only tensor evaluation |
| #404 | discopy-bot | Jupyter → marimo notebooks (#318) |
| #403 | discopy-bot | Test directory refactor (#381) |
| #415, #419, #397, #347 | toumix | Your own — cat.Equation, CI images, README hello-world, property-based testing |

Drafts #295 (colltoaction, Feb 2025) and #308/#325 (IsidorManning) have
been open long enough that a close-or-commit decision would be kind.

### P2 — strategic threads needing a design call before execution

Per the LLM guidelines ("delegate the execution not the design"), these
are blocked on you, not on implementation effort:

- **Tensor refactor** (#410 + #402): make CMap/einsum the single
  evaluation path. #402 is ready; it decides the architecture.
- **Equation into `cat`** (#413 + #415): replaces the
  `hypergraph_equality` context manager with `Functor.quotient` — an
  API-visible change.
- **`functor_factory`** (#380) and **CMap/Hypergraph alignment** (#391):
  both reshape how custom categories plug into hypergraphs.
- **Marimo migration** (#318/#404/#320) is blocked on #425
  (`_repr_mimebundle_` rich display hooks) — #425 itself is a
  well-scoped, autonomous-friendly task.

### Issue backlog

62 open issues; roughly a third predate 2023 (#25, #26, #27, #37,
#54–#57, #92, #121–#124, #150–#169...). A labelled triage sweep
(good-first-issue / stale / superseded) would make the tracker honest
again. I can draft close-or-keep recommendations in a future round if
wanted.

## Proposed queue for the next evening rounds

In order, each small enough to review comfortably:

1. Fix #427 (feedback docstring `Ty.name` regression) — likely a
   few-line fix plus regenerated image.
2. Fix #426 (bubble frame boundaries) — localised in `drawing/`.
3. Implement #395 (`swap_factory` honoured, README example actually
   exercised in tests — #386 already drafts README testing).
4. Implement #425 (`_repr_mimebundle_` on `Diagram`/`Drawing`) to
   unblock the marimo migration.
5. Issue-tracker triage report (no code, just recommendations).

Say which of these to green-light (or reorder) and the next round will
pick them up.

## Review of the new routine prompts

Read in full: `Alexis/.agents/{RULES,ROUTINE,EVENING,BIRDSONG,DAYLIGHT}.md`
on `claude/evening-discopy-planning-4etphi` (commits `37a8171`, `523c952`).
The approval design is strong — the INTEGRITY check (void on post-reaction
edits), the 7-day EXPIRY, the reaction-only-from-`toumix` rule and the
anti-forgery convention for Slack tags together close most of the obvious
injection paths. Findings from this first live round:

1. **Config placeholders are unfilled**: `ALEXIS_SLACK_ID = "U________"`
   and `SLACK_CHANNEL = "#________"`. Evening's READ MEMORY and POST
   steps cannot run until these are set, and no Slack connector is
   attached to this environment.
2. **Session scope doesn't match `REPOS`**: `rel-int/wiki` (the
   CONTROL_REPO) is not in this session's GitHub scope — only
   `discopy/discopy` is. The routine environment needs both sources, or
   the prompts folder should live in a repo the sessions can reach.
3. **The TODO.md merge gate doesn't exist yet**: RULES.md #1 relies on
   "a merge gate blocks any PR that still contains it", but neither
   `build.yml` nor `benchmark.yml` checks for TODO.md. Until that
   workflow exists the sign-off mechanism is honour-system.
4. **No Slack fallback for Evening**: EVENING.md is report-only and
   posts to Slack. Tonight (no Slack, no channel configured) I fell back
   to committing this file to a `claude/` branch — consistent with
   ROUTINE.md's push rules, but worth writing down as the intended
   fallback so future rounds behave predictably.
5. **No stale-claim recovery for the mutex**: a crashed session leaves a
   point `[WIP] @<SessionID>` forever, and no rule allows reclaiming it.
   Suggest: a `[WIP]` older than ~24h may be reset to `[ ]` with a note
   in the Slack summary.
6. **Birdsong→Daylight timing is tight**: Birdsong (09:50) pushes TODO.md
   guidance and Daylight (10:00) starts claiming points ten minutes
   later. Birdsong's full GitHub scan can easily run past 10:00; its
   "retry once on non-ff, else skip" makes this safe but means guidance
   may silently miss the morning run. A 09:30 start (or Daylight at
   10:15) would give slack.
7. **Minor rule tension**: RULES.md #2 (never write PR descriptions,
   drafts with empty body) is stricter than CONTRIBUTING.md's LLM
   guideline (LLM-generated change-lists allowed under a human-written
   header). Fine if intentional — RULES.md wins for routine agents —
   but a one-line cross-reference in CONTRIBUTING.md would avoid
   confusion for other contributors' agents.
8. **Trigger creation is still manual**: nothing schedules the 02:00 /
   09:50 / 10:00 / 14:00 / 18:00 runs yet, and in this autonomous
   session the trigger-management tools require an interactive approval
   that cannot be granted, so I could not verify or create schedules.

## Steps only you can do

- [ ] **Pick the `swap_factory` fix**: alias vs. rename (#395).
- [ ] **Review/merge #419** and grant the workflow permission (repo
      Settings → Actions → "Read and write permissions", or a PAT
      secret) so CI can commit regenerated `docs/_static` images.
- [ ] **Review external PRs**: #305 (7 months old), #389, #406, #402.
      #402 doubles as the design decision for the tensor refactor (#410).
- [ ] **Decide the API for #413** (Equation/`Functor.quotient`) before
      anyone executes it.
- [ ] **Close-or-commit** on stale drafts #295, #308, #325.
- [ ] **Release 1.3.0** after #426/#427 land: `git tag`, `uv build`,
      `uv publish` (PyPI credentials), GitHub release notes — none of
      this can be done from an agent session.
- [ ] **Modal account/secrets** if you want #389's benchmark migration.
- [ ] Optional: green-light / reorder the proposed evening-round queue
      above.

To activate the routine pipeline (see the prompts review above):

- [ ] Fill in `ALEXIS_SLACK_ID` and `SLACK_CHANNEL` in `ROUTINE.md` and
      attach a Slack connector to the routine environment.
- [ ] Add `rel-int/wiki` (CONTROL_REPO) to the environment's sources.
- [ ] Add the TODO.md merge-gate workflow that RULES.md #1 assumes.
- [ ] Create the schedules (02:00, 09:50, 10:00, 14:00, 18:00 UTC?) —
      and decide the timezone, which the prompts don't state.
- [ ] Decide the Evening fallback when Slack is unreachable (suggested:
      commit the report to a `claude/` branch, as done tonight).

## Environment notes for future rounds

- `uv sync --dev`, `pflake8` (clean) and the core pytest suite all run in
  the sandbox; every local test failure I saw tonight traced back to a
  missing dependency, not to code.
- `uv sync --group all` fails here: the torch wheel download is blocked
  by the network proxy, which aborts the whole sync, leaving `sympy`,
  `nltk`, `pyzx`, `jax`, `pytket` and `pennylane` uninstalled too.
  Installing `sympy`/`nltk` manually turns most of the suite green; the
  graphviz `dot` binary is also absent, which fails three `cmap` drawing
  doctests. CI (green on `add8d65`) remains authoritative for
  `discopy/quantum` and `discopy/tensor`.
- A session-start hook that runs `uv sync --dev && uv pip install sympy
  nltk` (and installs graphviz if the base image allows) would let future
  evening rounds verify their own work.
