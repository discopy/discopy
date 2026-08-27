# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice covers `discopy.grammar.categorial.Functor` and
`discopy.grammar.pregroup.Functor`, stacked directly on
`split/3-cmap-hypergraph-strategy`. See the earlier branches' BUGS.md for
the rest of the property suite's history.

No bugs were found and neither functor is enrolled in the property
matrix — grammar diagrams have no generic strategy of their own, being
built from a lexicon rather than generated. Each functor was simply
missing its `@factory` decorator (the same static-binding bug as
`python.additive.Function` on an earlier branch: without it, `ar`
resolves to the base `biclosed.Functor`/`frobenius.Functor` instead of
the grammar-level subclass), fixed here as a one-line change to each
file with no test or matrix impact.
