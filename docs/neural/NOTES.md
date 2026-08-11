# Notes from the refactor

`discopy.neural` was rebuilt around `MapNN`: a diagram plus shared
learnable generator maps compiled into a global interaction, and a solver
that says how to run it. The rule was that **no number moves**, which is
checked bitwise by `test/neural/test_equivalence.py` against the
fingerprints frozen in `golden/` — structure, parameters, logits, gradients
and 20-step loss trajectories, in float32 and float64, for all four
recorded models.

This file records what was noticed along the way and deliberately *not*
fixed, so that it is written down somewhere rather than smuggled into a
change whose job was to preserve behaviour.

## What did change, and why it is not a number

**The modules moved.** The generators now live under the `MapNN` that
shares them (`map.ar.cell.*`) and the answer refresh under the solver that
runs it (`map.solver.refresh.*`). A pre-refactor checkpoint therefore needs
its keys renamed; `examples/sudoku/model.py` carries the map, as `rename`,
`translate` and `load_checkpoint`, and `test_parameters` asserts that the
translation is strict and total and that the weights are bitwise the
golden's, taken in the golden's own order.

**`Skeleton`, `Interpretation`, `Wiring`, `Router`, `Schedule` and the
engines are gone.** They were five names for what is now two: a diagram and
a `MapNN`. The port families a solver reads used to come from a declared
`Signature` per box; they are now read off the *wiring* — a port is a head
unless it is wired to an earlier port of the same box, which is exactly the
second copy of a traced leg — so a diagram no longer has to carry its
signatures to be compiled. `Signature` survives where it earns its keep:
laying out one generator's ports for a cell, for a wiring builder and for
`check_equivariant`.

**`InteractionMap` no longer pretends to compose.** It used to record the
boundary bookkeeping of `f >> g` while its own docstring explained that two
interaction maps glued along a shared object do *not* compose by
substitution. It now raises. The tensor is kept, because `Phi_theta` really
is the parallel application of every local interaction.

**`Transition` and `Iteration` dissolved into `Interaction`.** They were
specifications of the object that now exists: the compiled interaction
carries `local`, `routing`, `state` and `is_involution`, and the resumption
law `T^(a+b) = T^b . T^a` is asserted on the forward pass rather than on a
dataclass that could not fail it.

## Left alone, on purpose

**The clue loop of models A and B is written twice and read once.** A cell
emits its clue (or zeros) on *both* ends of the clue loop, and reads only
one of them. The same holds for the answer loop of model C. One of the two
emissions is dead in every round; the wire is there because a trace is a
*pair* of ports, but only one direction is ever consumed. Removing the
redundant write would change the flat state and hence the arithmetic, so it
stays.

**`inject=True` re-adds the initial vector to every port, not just the ones
it was written on.** `CMap.forward` adds the whole `init` vector to the
incoming messages each round. `init` is zero everywhere except the clue
ports, so the effect is the intended one — but the addition is performed
over the entire flat tensor each round, which is `total`-many adds where
`sites * width` would do.

**Model A pays for an answer role it does not have.** Models A and C share a
diagram, and A's interpretation erases the answer role. The erasure is
complete — no ports, no wires, no width — but the cell module still carries
`Mode.CARRY` in its `mode` mapping for a role of width zero. Harmless, and
it is what makes one signature serve both.

**`train_epoch` reports the loss per supervised checkpoint for both
supervision schemes, but the two schemes take a different number of
optimizer steps per batch.** The single-run models take one, the recursion
takes `steps`. This is a residual asymmetry of the protocol, documented in
`examples/sudoku/train.py` rather than hidden.

**`evaluate` decodes with the clues written back over the predictions**, so
a model is never scored on a cell it was given. That is the right rule for
fill-in-the-blanks, but it means cell accuracy is not comparable across
benchmarks with different numbers of givens.

**`Interaction.write` returns a copy.** `index_copy` (not `index_copy_`) is
used so the state stays a graph output that autograd can differentiate
through; every `initial` and every refresh therefore allocates a fresh flat
tensor. Deliberate, and the reason the segmented loop can detach cleanly.

**`forward_reference` is quadratic in the number of boxes.** It is the
reference oracle and is only ever run on small maps.

**The noise study's plotting is not carried over.** `evaluate.py --noise`
reproduces the depth-by-noise grid and writes the same JSON and `npz`
artifacts, with the provenance of the run beside the numbers; the ~230
lines of matplotlib that turned those artifacts into the committed figures
under `figures/` were dropped rather than ported.

## Things that are not bugs, but will surprise

**A golden is a fingerprint of one interpreter — run the gate in the locked
environment.** On torch 2.2 the float64 forward and backward and every loss
trajectory differ from `golden/` in the last bits, for all four models,
while float32 forward and backward match bitwise. That is not something to
re-record or to give a tolerance: the version *is* pinned. `uv.lock`
resolves `torch==2.13.0+cpu`, which is what the goldens were recorded under
and what CI installs with `uv sync --locked`. A run that shows those
failures is a run against the wrong interpreter. The locked torch is the
**CPU** build, so it runs the tests but cannot run the GPU studies; those
want a CUDA build, which is not the environment any golden is recorded
against.

**Thread count is part of what "bitwise" means.** A multi-threaded CPU
reduction splits its sum across threads and adds the partial sums back in a
different order. The goldens are recorded, and the tests run, with
`torch.set_num_threads(1)` — which at these widths is also several times
*faster*, since handing a cell to a thread pool costs far more than its
arithmetic.

**The GPU is not reproducible run to run.** Two `train.py goi --seed 0
--quick` runs on the same GPU with the same code differ in the fourth
decimal of the first epoch's loss. Any before/after comparison of training
has to be done on the CPU; that is what the equivalence tests do.

**A batch of mixed shapes is not bitwise a batch of one shape.** Running two
maps as their monoidal product evaluates their shared sites in one batched
module call rather than two, and a matmul over six rows is not the same
kernel as one over two. The values agree to a few units in the last place.
This is capability (`discopy/neural/batch.py`), not a change to any existing
run, and it is the same rounding freedom `CMap.compile` documents.

## The one deliberate structural change to the wiring

The clique cell used to keep the two states of its `LSTMCell` as one wide
port that it sliced in half. It now keeps them as two named roles, `hidden`
and `memory`, on one traced loop, so it reads them by name.

The refinement was laid out to be a *renaming and nothing else*: the two
roles occupy exactly the bytes the one wide port did, in the same order, so
the routing permutation `src`, the box-order permutation `perm`, the group
metas and the total width are unchanged, and every logit, every gradient
and the whole loss trajectory are bitwise identical. What does change is the
bookkeeping *around* those bytes — the `edges` involution and `port_widths`
gain two entries per cell, and `layout`/`inverse` relabel accordingly, since
there are now two ports where there was one. `test_ports_refined` asserts
that merging each `(hidden, memory)` pair back recovers the golden port
widths exactly.

Model B's map therefore reports 1053 wires where it used to report 972: the
81 extra are the second half of each cell's state loop, which was always
there as half of a wide wire.

## Not attempted

**Closing the loops with `CMap.trace` rather than with an explicit
self-wire.** A loop *is* a trace, and `CMap.from_box(g).trace() ==
from_wiring(CMap, (g,), [((0, 0), (0, 1))])` — asserted as a doctest in
`discopy/neural/signature.py`. Building the 108-box wirings that way would
mean constructing the open map's involution by hand first and then splicing
it, which is strictly more index arithmetic than wiring the loops directly
from `Signature.loops()`. The equation is documented and checked; the
construction stays direct.

**A `neural.frobenius` / `neural.symmetric` hierarchy.** The target category
is compact closed either way — swaps, cups, caps and traces are all wiring
in it — so mirroring the source hierarchy would duplicate the whole module
per category and buy nothing. The source category is a *parameter* of
`Signature.box` and of the wiring builders instead, and its
`require_planar` / `require_acyclic` / `require_oriented` /
`require_connected` flags do the guarding.

**Out of scope, and now stale.** `docs/optuna/*.py` and
`docs/notebooks/neural-cells-lecture.ipynb` import the removed
`sudoku`/`core` packages. The optuna scripts were *already* stale before
this refactor — they call `zoo.RRNSolver` and `zoo.TRMSolver`, names that
were removed earlier — so they need a pass of their own; the brief was
`discopy/neural` and `docs/neural`.
