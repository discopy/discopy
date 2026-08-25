# Review round: cubic-dev-ai, 2026-08-24

Quoted verbatim from the two unresolved review threads on #442.

## discopy/closed.py:583 (P2)

> When callers pass a negative budget, `Strategy.spend()` keeps contracting
> beta redexes because the budget never equals zero. Stop when the finite
> budget is non-positive, or reject negative budgets before reduction.

- [x] Reproduce with a negative `budget`, fix `Strategy.spend` to stop on
      any non-positive finite budget.

## test/closed.py:19 (P3)

> Unitype's __eq__ makes Unitype() (and thus Ty(Unitype()) == U) equal to
> Exp(U, U) (thus U == U >> U), but __hash__ returns hash("U") for every
> type, which differs from the hash of Ty(Exp(U,U)). Two objects that
> compare equal now hash differently, which breaks lookups whenever these
> types are used as dict/set keys (e.g. in a Functor ob_map or diagram
> dict). Base __hash__ on the same structural fields __eq__ uses, so that
> types equal under __eq__ are guaranteed the same hash.

- [x] Reproduce (a `Functor.ob_map` keyed on `Unitype()` misses a lookup by
      a structurally-equal `Exp(U, U)`), then fix `Unitype.__hash__` to
      match what `Exp.__hash__` would give the same `(base, exponent)`.
      cubic's own suggested one-liner (`hash((self.base, self.exponent))`)
      does not actually fix it: `Exp.__hash__` is `hash(repr(self))`, a
      different formula, so it still mismatches. Delegated to a plain
      `Exp`'s own hash instead.
