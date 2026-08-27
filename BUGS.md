# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice covers `discopy.tensor.Tensor`/`Diagram`, stacked on
`split/4-matrix` since `Tensor` subclasses `Matrix`. See the earlier
branches' BUGS.md for the axiom infrastructure, monoidal strategy,
cmap/hypergraph and matrix slices.

## Pickling that loses or demands state

- `NamedGeneric.__setstate__` (in `discopy.abc`) was defined on the class
  its subscripts never inherit from, so a subscripted instance —
  `Matrix[int]`, `Tensor[...]`, `Hypergraph[...]`, `CMap[...]` — unpickled
  as its bare origin class. Fixed by moving the restore into the
  dynamically-built subscript class itself. This lands here rather than
  with the axiom infrastructure because the only call site depending on
  the previous (buggy) signature is `discopy.tensor.Box.__setstate__`,
  which called `NamedGeneric.__setstate__(self, state)` explicitly; both
  sides of the fix are in this commit.

## Open, declared and recorded in the matrix

- A `Tensor` with more than `config.NUMPY_THRESHOLD` entries elides its
  repr as a literal ellipsis, breaking transparency
  (`eval(repr(x)) == x`).
