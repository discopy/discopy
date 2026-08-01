# Notes from the refactor

`discopy.neural` became a category-generic engine and the study kept only
what is a study. The rule for that refactor was that **no number moves**,
which is checked bitwise by `test/neural/test_equivalence.py` against the
fingerprints frozen in `golden/`.

This file records what was noticed along the way and deliberately *not*
touched, so that it is written down somewhere rather than fixed in a
commit whose job was to change nothing.

## Left alone, on purpose

**The clue loop of models A and B is written twice and read once.** A cell
emits its clue (or zeros) on *both* ends of the clue loop, and reads only
one of them. The same holds for the answer loop of model C. One of the two
emissions is dead in every round; the wire is there because a trace is a
*pair* of ports, but only one direction is ever consumed. Removing the
redundant write would change the flat state and hence the arithmetic, so it
stays.

**`inject=True` re-adds the clue to every port, not just the clue ports.**
`CMap.forward` adds the whole `init` vector to the incoming messages each
round. `init` is zero everywhere except the clue ports, so the effect is the
intended one — but the addition is performed over the entire flat tensor
each round, which is `total`-many adds where `len(clue_ports) * width` would
do.

**Model A pays for an answer role it does not have.** Models A and C share a
skeleton, and A's interpretation erases the answer role. The erasure is
complete — no ports, no wires, no width — but the cell module still carries
`Mode.CARRY` in its `mode` mapping for a role of width zero. Harmless, and
it is what makes one signature serve both.

**`train_epoch` reports the loss per supervised checkpoint for both
supervision schemes, but the two schemes take a different number of
optimizer steps per batch.** The single-run solvers take one, the recursion
solver takes `n_sup`. This is documented in `core/train.py` as a residual
asymmetry of the protocol, and it is still there.

**`evaluate` and `evaluate_act` decode with the clues written back over the
predictions**, so a model is never scored on a cell it was given. That is
the right rule for fill-in-the-blanks, but it means cell accuracy is not
comparable across benchmarks with different numbers of givens.

**`Router.write` returns a copy.** `index_copy` (not `index_copy_`) is used
so the state stays a graph output that autograd can differentiate through;
every `initial` and every answer refresh therefore allocates a fresh flat
tensor. Deliberate, and the reason the segmented loop can detach cleanly.

**`forward_reference` is quadratic in the number of boxes.** It is the
reference oracle and is only ever run on small maps; `test_forward_reference`
runs it on two puzzles and two rounds for that reason.

## Things that are not bugs, but will surprise

**A golden is a fingerprint of one interpreter.** torch 2.13 and torch 2.2
disagree in the last bits of `LSTMCell`, so the clique model's loss
trajectory differs between them while every other model's is identical. The
fingerprints in `golden/` were recorded under torch 2.13 on the CPU.

**Thread count is part of what "bitwise" means.** A multi-threaded CPU
reduction splits its sum across threads and adds the partial sums back in a
different order. The goldens are recorded, and the tests run, with
`torch.set_num_threads(1)` — which at these widths is also several times
*faster*, since handing a cell to a thread pool costs far more than its
arithmetic.

**The GPU is not reproducible run to run.** Two `train_a_goi.py --seed 0
--quick` runs on the same GPU with the same code differ in the fourth
decimal of the first epoch's loss. Any before/after comparison of training
has to be done on the CPU; that is what the equivalence tests do.

**A batch of mixed shapes is not bitwise a batch of one shape.** Running two
maps as their monoidal product evaluates their shared sites in one batched
module call rather than two, and a matmul over six rows is not the same
kernel as one over two. The values agree to a few units in the last place.
This is new capability (`discopy/neural/batch.py`), not a change to any
existing run, and it is the same rounding freedom `CMap.compile` documents.

## The one deliberate structural change

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

**Closing the loops with `CMap.trace` rather than with an explicit self-wire.**
A loop *is* a trace, and `CMap.from_box(g).trace() == CMap.from_wiring((g,),
[((0, 0), (0, 1))])` — asserted as a doctest in
`discopy/neural/skeleton.py`. Building the 108-box skeletons that way would
mean constructing the open map's involution by hand first and then splicing
it, which is strictly more index arithmetic than wiring the loops directly
from `Signature.loops()`. The equation is documented and checked; the
construction stays direct.

**A `neural.frobenius` / `neural.symmetric` hierarchy.** The target category
is compact closed either way — swaps, cups, caps and traces are all wiring
in it — so mirroring the source hierarchy would duplicate the whole module
per category and buy nothing. The source category is a *parameter* of
`Signature.box` and of the skeleton builders instead, and its
`require_planar` / `require_acyclic` / `require_oriented` /
`require_connected` flags do the guarding.
