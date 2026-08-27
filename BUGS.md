# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice adds the bugs found while building the cmap/hypergraph strategy
and enrolling `Hypergraph`/`CMap` at every monoidal-derived level that has
one. See the previous branches' BUGS.md for the axiom-infrastructure and
monoidal-strategy slices.

## Equality sensitive to representation noise

- `Hypergraph.to_graph` keyed spider nodes by the boundary's object rather
  than the spider's own type, creating a phantom attributeless node
  whenever a boundary wire reads an adjoint of its spider type — `hash`
  crashed with `KeyError: 'box'`. Fixed by keying on `spider_types`.

## Open, declared and recorded in the matrix

- `CMap.to_diagram` and `Hypergraph.to_diagram` need swaps to decode a
  trace, cup or cap at `traced`, `balanced` and `pivotal`, and
  `Hypergraph.cups`/`caps` accept only the right-adjoint orientation, so
  `to_hypergraph` is partial on rigid's left-handed cups and caps.

## Naming carried over from the previous branch

`NamedGeneric`'s subscript naming (in `discopy.abc`) now reads a
subscript's own `factory_name` instead of its bare `__name__`, so
`Hypergraph[frobenius.Diagram]` reprs and hashes with its full dotted
name instead of the collapsed `Hypergraph[Diagram]` the previous branch
still expects in `test/hypergraph.py::test_Hypergraph_repr` — hence this
one hunk of `discopy.abc` lands here rather than with the axiom
infrastructure, alongside the `cmap`/`hypergraph` modules whose own tests
already expect it.
