# BUGS.md

Every bug the property suite has found so far, grouped by root cause:
most failures were one design flaw repeating across classes, so each group
names the flaw once and lists where it struck. Fixed means fixed on this
branch; open bugs carry their declaration in the matrix.

This slice adds the bugs found while building the recursive monoidal
strategy and enrolling every monoidal-derived category (braided, traced,
balanced, symmetric, biclosed, rigid, pivotal, ribbon, compact, markov,
closed, feedback, frobenius) in the matrix. `hopf` is deferred (see
below). See the base branch's BUGS.md for the axiom-infrastructure slice.

## Serialisation inherited with the wrong signature

A structural box inherited `__repr__`, `to_tree` or `from_tree` from `Box`
or `Bubble`, whose `(name, dom, cod)` keys its own `__init__` rejects — so
`eval(repr(x))`, `dumps`/`loads` or both crashed on any diagram containing
one. Fixed by giving each class the serialisation its constructor reads.

- `traced.Trace` (repr printed `str` of its argument; no tree at all).
- `feedback.Feedback` (no tree; the memory was not stored).
- `balanced.Twist`, `braided.Braid` (the tree lost `is_dagger`).
- `markov.Copy`, `Merge`, `Discard` (no tree).
- `frobenius.Spider` (no tree).
- `biclosed.Eval`, `Coeval`, `Curry` and their `closed` subclasses
  (repr and tree).

  (`hopf.Representation`/`quantum.*` cases of the same bug land on the
  branches that enrol those carriers, since they only apply once tensor
  and quantum work is in place.)

## Static bindings where a factory should dispatch

A class attribute captured a concrete sibling instead of reading the
subclass's factory, so every override downstream was silently skipped.
Fixed by dispatching through `cls`.

- `Copy.dagger` and `Merge.dagger` built bare `markov` classes, so the
  dagger of a `closed.Copy` was a `markov.Merge` that closed diagrams
  reject — and `closed` had no `Merge` class at all.
- `Diagram.to_staircases` ran the bare `monoidal.Functor`, rebuilding any
  `Trace` as a `monoidal.Bubble` the level rejects, crashing `foliation`
  on every traced diagram.

## Pickling that loses or demands state

- `markov.Copy.__new__` required an argument the pickle protocol's bare
  `__new__(cls)` call cannot pass, so `Copy` and `Discard` never
  unpickled.

## Partial operations that crashed instead of degrading

- `foliation` crashed where `to_hypergraph` is partial — traced diagrams
  (via `to_staircases` above) and boundary-disconnected pivotal diagrams,
  whose rejection is by design; it now falls back to merging layers.
- `Feedback.dagger` crashed with a `TypeError` from generic bubble
  reconstruction; it now raises a clean `AxiomError`, the delay being
  irreversible.

## Open, declared and recorded in the matrix

- `feedback.Diagram.feedback` unrolls its memory in the wrong order
  (#606), falsified even on homogeneous memory.
- An uncoloured `monoidal.Wire` reprs as the `cat.Ob` that `Ty` coerces,
  which its type-strict equality rejects.

## Deferred to a later branch

- `discopy/hopf.py`'s `Intertwiner` subclasses `tensor.Diagram` and
  declares `frobenius`/`speciality`/`spider_fusion` (a
  `HypergraphCategory` axiom) inapplicable, so despite being a
  ribbon/pivotal-family carrier it cannot be enrolled until both
  `discopy.tensor` (a later branch) and `HypergraphCategory`'s axioms
  (the next branch) exist. Its bugs — the module/dimension discipline in
  `Representation`/`Intertwiner`, and the Reidemeister-1 counterexample on
  a composite `Rep(D(Z/2))` module — are recorded on `split/4-hopf`, which
  stacks on `split/4-tensor`.
- `NamedGeneric.__setstate__` (in `discopy.abc`) is defined on the class
  its subscripts never inherit from, so a subscripted instance —
  `Matrix[int]`, `Tensor[...]`, `Hypergraph[...]`, `CMap[...]` — unpickles
  as its bare origin class. The only call site that depends on the
  current (buggy) signature is `discopy.tensor.Box.__setstate__`, which
  calls `NamedGeneric.__setstate__(self, state)` explicitly; fixing the
  base class without also updating that call site breaks pickling for
  every tensor-family box. The fix (relocating `__setstate__` into the
  dynamically-built subclass) therefore lands together with
  `discopy/tensor.py` on `split/4-tensor`, where both sides of the change
  are made in the same commit.
