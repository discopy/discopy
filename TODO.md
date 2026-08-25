# Review round: cubic-dev-ai, 2026-08-24

Quoted verbatim from the two unresolved review threads on #442.

## discopy/closed.py:583 (P2)

> When callers pass a negative budget, `Strategy.spend()` keeps contracting
> beta redexes because the budget never equals zero. Stop when the finite
> budget is non-positive, or reject negative budgets before reduction.

- [WIP] @session_018brmB6AEz1RbKcyXjU3dXU-2026-08-25 09:00 Reproduce with a
      negative `budget`, fix `Strategy.spend` to stop on any non-positive
      finite budget.

## test/closed.py:19 (P3)

> Unitype's __eq__ makes Unitype() (and thus Ty(Unitype()) == U) equal to
> Exp(U, U) (thus U == U >> U), but __hash__ returns hash("U") for every
> type, which differs from the hash of Ty(Exp(U,U)). Two objects that
> compare equal now hash differently, which breaks lookups whenever these
> types are used as dict/set keys (e.g. in a Functor ob_map or diagram
> dict). Base __hash__ on the same structural fields __eq__ uses, so that
> types equal under __eq__ are guaranteed the same hash.

- [ ] Reproduce (a `Functor.ob_map` keyed on `Unitype()` misses a lookup by
      a structurally-equal `Exp(U, U)`), then fix `Unitype.__hash__` to
      match what `Exp.__hash__` would give the same `(base, exponent)`.
