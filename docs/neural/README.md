# `discopy.neural`

> `discopy.neural` trains neural interpretations of DisCoPy diagrams.
> `MapNN` compiles diagram structure and shared learnable generator maps
> into a global interaction, while a solver specifies how that interaction
> is executed.

## The workflow

A dataset of `(diagram, inputs, target)` samples, a `MapNN` interpreting
them, a solver, and then an ordinary PyTorch training loop.

```python
import torch
from discopy.frobenius import Ty
from discopy.neural import Dim, Iterate, MapNN, Mode, Orbit, Signature, Site, Sym

message, state, clue = Ty("message"), Ty("state"), Ty("clue")
cell = Signature((Orbit(message, 3, Sym.PERM),
                  Orbit(state, traced=True), Orbit(clue, traced=True)))

model = MapNN(
    ob={message: Dim(24), state: Dim(96), clue: Dim(24)},
    ar={"cell": Site(cell, {message: 24, state: 96, clue: 24},
                     {state: Mode.STATE, clue: Mode.INPUT}, hidden=192)},
    solver=Iterate(rounds=16))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for diagram, x, target in loader:
    final = model(diagram, {("cell", clue): encoder(x)})
    loss = criterion(readout(model.read(diagram, final, ("cell", state))),
                     target)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

Four things and no fifth:

* **`ob`** sends each atomic *role* — a name for what a wire carries, not
  how wide it is — to the `Dim` it carries. `Dim(0)` **erases** a role: its
  ports and the wires on them vanish, which is how one diagram serves two
  models.
* **`ar`** gives one shared learnable module per generator *name*. Every
  site of that name in every diagram is the same module, so the model size
  is independent of the diagram size and a dataset of differently shaped
  diagrams trains one set of weights.
* **the solver** says how the compiled diagram is run: `Iterate`,
  `FixedPoint`, `Recursion` or `ACT`.
* **the diagram** is an ordinary DisCoPy diagram (or combinatorial map) in
  any compact or hypergraph source category. `from_incidence` and
  `from_relation` draw one out of a family's combinatorics when it is more
  natural to *generate* the wiring than to compose it.

`MapNN` is a `torch.nn.Module`: `.parameters()`, `.to(device)`,
`.state_dict()` and every optimizer work as usual. Training loops are
deliberately **not** in the library.

Diagrams whose shapes differ batch as their monoidal product:

```python
from discopy.neural import Batch

batch = Batch([small, large, small], pad=True)   # a diagram, as far as MapNN
state = model(batch, {("cell", clue): encoded})
pieces = batch.split(model.read(batch, state, ("cell", state)), ("cell", state))
```

Sites of the same degree sharing a module still cost one batched call
between them, so a batch of mixed shapes is one map and a handful of calls,
not one call per member.

## The semantics

A diagram `D` in a source category `C` is interpreted by a monoidal functor
`F_θ`. Each atomic role goes to the `Dim` it carries; each generator
`f : X → Y` goes **not** to a feed-forward map `X → Y` but to a *parametric
interaction map* on its boundary,

    Φ_f : P_f ⊗ ∂f → ∂f,        ∂f = X* ⊗ Y,

reading one incoming message on every port and emitting one outgoing
message on every port. That is exactly what a `Network`'s torch module
computes: `R**w → R**w` for `w` the sum of the domain and codomain widths.
`discopy.neural.map` writes the two down and keeps them apart — `ParamMap`
composes by substitution, `InteractionMap` **refuses** to, because gluing
two interactions is wiring plus iteration, not substitution.

Wiring the boundaries together compiles the diagram to a global transition

    T_{D,θ} = σ_D ∘ Φ_θ  :  S_D → S_D

on the state object `S_D = ⊕_p R**w_p`, one summand per port — the
execution formula of the geometry of interaction. That compiled object is
an `Interaction`, and `Interaction.advance(state, n)` is the one
implementation of `T^n`. When the initial vector `i` is re-injected the
round is `T(s) = σ_D(Φ_θ(s)) + i`, an affine dependence.

Swaps, cups, caps and traces are wiring in the target category, so a functor
preserves them strictly and for free — they leave no box behind, only a
different involution. What survives as a box is a generator whose legs carry
a symmetry, and *that* is a promise about a torch module:
`discopy.neural.laws` names the group and `check_equivariant` measures the
residual. Permutation equivariance is **not** Frobenius structure, and
`fusion_residual` says by how much.

Four notions that are easy to conflate are kept apart in
[ARCHITECTURE.md](ARCHITECTURE.md): the categorical **trace**, the
persistent **state channel** (delayed feedback), **finite iteration**, and a
**fixed point**. Nothing in the category makes `T` contract;
`FixedPoint` is a solver that *looks* for a fixed point and
`Interaction.residual` is the number that says whether it found one.

## Layout

    discopy/neural/
      model.py       MapNN, the central abstraction
      map.py         the interpretation, the compiled Interaction, and the
                     formal ParamMap / InteractionMap specifications
      solver.py      Iterate, FixedPoint, Recursion, Refresh, ACT, HaltHead
      batch.py       batching over heterogeneous diagrams
      cells.py       Site, Relation, Gate, Cyclic -- concrete interpretations
      signature.py   the port layout of one generator, and wiring builders
      laws.py        equivariance laws, check_equivariant, fusion_residual
      core.py        the compact closed category: Dim, Network, CMap

    docs/neural/
      examples/sudoku/   three solvers, and everything to reproduce them
      golden/            the frozen pre-refactor fingerprints + recorder
      ARCHITECTURE.md    the layers, and which of them is which
      NOTES.md           what was left alone, and why

`import discopy.neural` does not import `torch`: diagrams, signatures, laws
and the whole compilation layer work without it; the torch-dependent names
are imported on first use.

## The example

[`examples/sudoku`](examples/sudoku/) is the workflow at full size: three
architectures from the literature as three choices of diagram, widths and
solver, compared under one protocol, plus the searched recipes and the
adaptive-computation-time study. See its
[README](examples/sudoku/README.md) for the results.

## What guards what

| claim | guarded by |
|---|---|
| the recorded models compute what they always did | `test/neural/test_equivalence.py` against `golden/`, bitwise |
| the fused forward equals the one-call-per-box oracle | `test_general.py`, `test_equivalence.py::test_forward_reference` |
| the compilation layer is torch-free | `test_formal.py::test_compiling_a_diagram_does_not_import_torch` |
| `interaction_spec` reads a `Network` without touching it | `test_formal.py::test_interaction_spec_changes_nothing` |
| an `InteractionMap` does not compose by substitution | `test_formal.py::test_interaction_maps_do_not_compose` |
| `T(s) = σ(Φ(s)) + i` | `test_formal.py::test_reinjection_is_an_affine_shift` |
| `T^(a+b) = T^b ∘ T^a`, bitwise | `test_formal.py::test_iteration_is_resumption` |
| a batch is the monoidal product, member for member | `test_general.py::test_a_batch_is_the_product_of_its_members` |
| the detach boundary of a segment is where it says | `test_general.py`, `test_sudoku_smoke.py` |
| the whole training machinery runs end to end | `test_sudoku_smoke.py`, in a few seconds |
| a trained model really solves sudoku | `test_sudoku_act_e2e.py`, `pytest -m neural_e2e` |
