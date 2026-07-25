# TODO — Review follow-up on non-self-dual hypergraphs

Alexis's live directive, verbatim:

> Go through the discopy PRs you own and follow the agents/EVENING.md prompt i.e. go through the reviews and implement them

## Checklist

- [x] @evening-2026-07-25T09:00Z-2026-07-25 09:16Z Address the review on
  [PR #393](https://github.com/discopy/discopy/pull/393): preserve nondefault
  `dom` and `cod` in `Ob.unwind`, test it, document `Ty.unwind` as atomic-only,
  and clean the unused test alias and spacing.
- [ ] Merge current `main` — blocked: `discopy/hypergraph.py` overlaps the PR's
  `same_side` cup/cap fix with main's `ports` performance refactor; merge
  aborted per Evening.

## Mathematical description

Unwinding an adjoint object removes its winding number while preserving its
chosen left and right adjoint steps. Unwinding a type is defined only for one
atomic object at a time.
