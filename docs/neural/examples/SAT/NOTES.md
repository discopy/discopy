# What this example ran into, and what it measured

`README.md` says what the study is. This file says what building it cost,
because the SAT setting stresses `discopy.neural` in a way the sudoku study
never did, and the difference is worth writing down before Part 2 spends GPU
hours on it.

## One diagram per sample is a different regime

In `examples/sudoku` every 9×9 puzzle reads the *same* diagram. `MapNN.compile`
runs once per process and the whole cost of the formalism is a rounding error
next to the arithmetic.

Here every formula **is** a diagram. A batch of 16 random instances with
`n ∈ [10, 100]` is a map with about 5 800 boxes and 34 000 ports, and it is a
new one every batch. That is exactly the regime `MapNN` advertises — *"a
dataset of `(diagram, inputs, target)` samples, the diagrams may all differ"* —
and it is the regime nothing had exercised at size.

## Measured, on one H100, `disc` env (torch 2.13, CUDA 13)

A training pool of 8 batches × 16 instances, `n` uniform in `[10, 100]`,
`α ∈ {3.0, 3.5, 4.0, 4.26}`, 46 807 boxes in total:

| phase | seconds | note |
|---|---|---|
| drawing the diagrams (`from_incidence`) | 20.0 | 0.4 ms/box |
| interpreting them (`MapNN.compile`) | 370.3 | 7.9 ms/box |
| first epoch | 109 | includes `CMap._routing`, built lazily |
| every later epoch | 6 | 0.75 s per optimizer step |

So **494 s of one-off cost buys 6 s of training per epoch**. That ratio, not
the arithmetic, is what bounds this study.

Two more facts that shape every design decision here:

* **A training step is launch-bound and its cost barely depends on the batch
  size.** 32 rounds forward and backward take 0.55 s at 2 904 boxes, 0.49 s at
  11 616 and 0.53 s at 25 536: one round issues one batched call per
  *(module, port-width)* group, about 20 of them, whatever the group holds. So
  GPU time is nearly free per instance and compilation is the whole currency.
* **`torch.compile` does not pay.** `MapNN.compile_rounds` wraps the round step
  of *one* `CMap` in a fresh closure, so a fresh diagram per batch means one
  dynamo compilation per batch. It is left off, deliberately.

## Where the time goes, and the one-word fix

Profiling `MapNN.compile` on a single 1 452-box map (`interpret`, 9.25 s total):

```
   ncalls  tottime  cumtime  function
     4356    0.183    7.257  discopy/neural/core.py:258(box_ports)
     4356    2.905    7.071  discopy/cmap.py:325(_box_port_indices)
```

`CMap._box_port_indices` is a plain `@property` that rebuilds a tuple of
`n_boxes` tuples on **every** access, and `interpret` — like `CMap._routing`,
and `batch._sites` — reads it once per box. That is `O(boxes × ports)` where
the work is `O(ports)`. `CMap.ports` has the same shape of problem one level
down: it builds its list with `sum([...], [])`, a quadratic list concatenation
(1.45 s of a 4.9 s `from_incidence` at 5 808 boxes).

Making `_box_port_indices` a `cached_property` and flattening `ports` into a
comprehension — measured by monkey-patching the two, no repository change —
turns the compile path **linear**:

| batch | boxes | draw | compile | `_routing` |
|---|---|---|---|---|
| 8 × n=50 | 2 904 | 0.41 s | 0.78 s | 0.17 s |
| 32 × n=50 | 11 616 | 1.62 s | 3.14 s | 0.50 s |
| 64 × n=55 | 25 536 | 3.83 s | 6.91 s | 0.98 s |

i.e. ~150 µs/box drawing, ~270 µs/box compiling, flat in the batch size —
against 7.9 ms/box today. **About 20× on compilation**, and the 109 s first
epoch becomes about 7 s.

Both are in the core library (`discopy/cmap.py`), not in `discopy.neural`,
both are behaviour-preserving on an immutable `CMap`, and neither is needed by
any example that compiles one diagram. They are reported here rather than
applied; the numbers above are what an approval would buy.

## What the example does about it meanwhile

`train.Pool` is the whole answer: compile a set of batches, reuse them for
several epochs, then throw the pool away and compile a fresh one. `Budget`
exposes `pool_batches`, `epochs` and `pools`, so a run trades instance
diversity against wall clock explicitly rather than pretending the cost is not
there. `train_model` records `seconds_drawing`, `seconds_compiling` and
`seconds_stepping` in every checkpoint, so no run can quietly misreport which
of the three it spent its time in.

Two smaller consequences:

* **Batches are small-ish and there is no bucketing.** With the quadratic in
  place, `k` batches of `N/k` boxes cost `O(N²/k)`, so *more, smaller* batches
  compile faster — the opposite of the usual advice, and it inverts again once
  the fix lands. Bucketing by size buys nothing either way: a batch is the
  disjoint union of its members' factor graphs, so nothing is padded and an
  instance costs exactly what it is.
* **`discopy.neural.Batch` is not used.** It folds `@` over the members, and
  `CMap.tensor` relabels every port of the accumulator each time: 2.5 s for 8
  instances of `n=40`, against 0.1 s for the same disjoint union laid out in
  one pass by `model.incidence`. The two maps differ only in the order of
  their boxes — `Batch` interleaves each member's literals with its clauses,
  the one-pass build puts all literal nodes first — and `model.check_layout`
  asserts the ordering the readout depends on against the wiring itself.

## Things that surprised us, kept for Part 2

* **`Dim(0)` does the flag-switching.** The stateful and stateless clause are
  one wiring: `CSTATE → Dim(0)` erases the clause loop and its ports, exactly
  as model A of the sudoku study erases the answer trace. So `--stateful` is
  an *interpretation*, not a second diagram, and the two share
  `model.factor_graph` verbatim.
* **A literal node can have degree 1.** A literal occurring in no clause still
  belongs to its `flip` relation, so no orbit is ever empty and `Site` never
  sees a zero arity — but `Sym.PERM` over one leg has no generators, so those
  sites carry no equivariance law at all. `evaluate.equivariance` therefore
  tests the cells at a chosen degree and reports which.
* **Literal degrees vary a lot** — Poisson with mean `kα/2 ≈ 6.4` at the
  threshold — so the number of `(module, port-width)` groups per batch is
  about 20 rather than 3. That is the reportable throughput cost `project.md`
  asks Part 2 to measure for the clique, and the factor graph already pays a
  smaller version of it.
