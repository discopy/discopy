# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice covers `discopy.python.finset.Function`/`Permutation`, stacked
directly on `split/3-cmap-hypergraph-strategy` (no tensor dependency).
See the earlier branches' BUGS.md for the rest of the property suite's
history.

## Type atoms that are not wires

- `Function`'s `ob` was the bare `int`, which the strategy/axiom
  machinery cannot treat as a wire-like generator; changed to
  `discopy.testing.Natural`.

## Open, declared and recorded in the matrix

- `finset.Function.swap` returns the inverse permutation (#606): correct
  only where both halves have equal length, a joint constraint
  per-argument generation cannot state. `braid_naturality`, `hexagon_left`
  and `hexagon_right` are declared broken and recorded in the
  counterexample ledger.
