# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice covers `discopy.matrix.Matrix`, the first concrete semantic
carrier enrolled. See the earlier branches' BUGS.md for the axiom
infrastructure, monoidal strategy and cmap/hypergraph slices.

## Static bindings where a factory should dispatch

- `Matrix.braid = swap` bound the integer-typed `swap` directly rather
  than reading it off the subclass, so `Tensor.braid` (defined on the
  branch enrolling `discopy.tensor`) would silently keep using
  `Matrix.swap` on `int`s instead of `Tensor.swap` on `Dim`s. Fixed by
  `braid = classproperty(lambda cls: cls.swap)`.

## Open, declared and recorded in the matrix

- `Matrix.copy(x, n)` is wrong for `x, n >= 2` (#606): recorded in the
  counterexample ledger. `copy_cocommutativity`, `copy_counitality` and
  `copy_monoidal_coherence` are declared broken
  (`MarkovCategory.copy_*.failing(...)`) on the full carrier, and
  `Axiom.weaken` is used for the first time — `copy_cocommutativity_small`
  and `copy_counitality_small` restate the same two laws quantified over
  `Small[C0]` (objects of length at most one), where the bug does not
  reach: the property matrix shows one expected failure and one green
  cell per law instead of a single blanket expected failure. The
  monoidal-coherence law reaches dimension two even from atomic
  arguments, so it has no analogous small-object restatement.
