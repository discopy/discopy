# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice covers `discopy.hopf.Intertwiner`, the ribbon category of
representations of a finite-dimensional Hopf algebra. It stacks on
`split/4-tensor` (not directly on `split/3-cmap-hypergraph-strategy`
like the other stage-4 branches) because `Intertwiner` subclasses
`tensor.Diagram` and declares the `HypergraphCategory` axioms
inapplicable, so it needs both `discopy.tensor` and
`HypergraphCategory`'s axioms to exist first — see the deferral note on
`split/2-monoidal-strategy`'s BUGS.md. See the earlier branches' BUGS.md
for the rest of the property suite's history.

## An object discipline torn between modules and dimensions

`hopf.Representation` is a `Dim` carrying an action, and the code mixes
the two freely: generic diagram operations slice modules down to bare
dimensions by design, while the module structure is needed wherever an
action is read.

- Fixed: the ribbon classmethods `Intertwiner.braid`, `twist`, `cups`
  and `caps` returned plain dimension boundaries, dropping the module
  structure their callers read the action from.
- Open: `Intertwiner` is not its own factory — its `ar` resolves to the
  plain tensor category, and making it one cascades into every generic
  operation that builds dimension-boundaried composites — so the
  arrow-quantified laws are declared inapplicable.
- Open: the hypergraph functor rebuilds a representation-typed cup or
  cap whose adjoint is its dimension reversal, not the dual module, so
  `normal_form` and `foliation` cannot be checked up to hypergraph.
- Open: a class subscripted by an algebra instance has no importable
  factory name, so its trees cannot be decoded.

## Open, declared and recorded in the matrix

- Reidemeister 1 fails semantically on a composite module of
  `Rep(D(Z/2))`, recorded in the ledger on `V @ V`: the swap is the
  braiding, and the pivotal correction of cups and caps fires on a
  *structural* comparison of the pivotal element with the unit —
  semantically equal but structurally distinct composites — flakily,
  since the rebuilt dual actions compare structurally unstably.
