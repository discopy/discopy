# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice covers `discopy.python.multiplicative.Function`, stacked
directly on `split/3-cmap-hypergraph-strategy`. See the earlier branches'
BUGS.md for the rest of the property suite's history.

No new bugs were surfaced enrolling this carrier: `Function` is compared
extensionally (probing both sides on canonical arguments, recursively
through curried types), and its strategy generates functions that select
an argument or return a small constant, which was enough to keep every
inherited `ClosedCategory` axiom green.
