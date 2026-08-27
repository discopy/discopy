# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice covers `discopy.python.additive.Function`, stacked directly on
`split/3-cmap-hypergraph-strategy`. See the earlier branches' BUGS.md for
the rest of the property suite's history.

## Static bindings where a factory should dispatch

- `python.additive.Function` missed its `@factory` decorator, so its `ar`
  resolved to `function.Function`, the base class of all python
  functions, instead of itself — every generic operation building an
  `additive.Function` from a functor or a diagram silently produced the
  wrong class. Fixed by adding `@factory`.
