# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice covers only the base axiom infrastructure (`discopy.cat` and
`discopy.abc`'s `Category`/`ColouredMonoid` levels). Bugs found once
later stages enrol more carriers are recorded on their own branch.

## Type atoms that are not wires

- `utils.is_tuple` only recognised the bare `tuple[type, ...]` alias, so
  a tuple-subclass `ob` (the python carriers' `Types`, added in a later
  branch) would make `Functor._map_atomic` (in `discopy/monoidal.py`,
  which reads it) iterate a bare type. Fixed in `discopy/utils.py`.

## Open, declared and recorded in the matrix

- `cat.Functor.unitality` fails: `MappingOrCallable.then` composes by
  iterating the keys of the left-hand map, and the identity functor
  enumerates none, so `id >> f` forgets everything `f` does instead of
  being `f`. Composition is unital only on the left. Declared broken
  (`Category.unitality.failing(...)` on `Functor`) rather than fixed,
  since it is a property of how `Functor` composition is defined, not a
  small implementation slip; recorded in the counterexample ledger.

## Deferred to a later branch

`NamedGeneric.__setstate__` (in `discopy.abc`) is defined on the class its
subscripts never inherit from, so a subscripted instance — `Matrix[int]`,
`Tensor[...]`, `Hypergraph[...]`, `CMap[...]` — unpickles as its bare
origin class. The only call site that depends on the current (buggy)
signature is `discopy.tensor.Box.__setstate__`, which calls
`NamedGeneric.__setstate__(self, state)` explicitly; fixing the base class
without also updating that call site breaks pickling for every tensor-family
box. The fix (relocating `__setstate__` into the dynamically-built subclass)
therefore lands together with `discopy/tensor.py` on `split/4-tensor`,
where both sides of the change are made in the same commit.
