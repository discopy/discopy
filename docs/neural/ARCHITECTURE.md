# The architecture, layer by layer

This document says what each layer of `discopy.neural` *is*, in the language
of category theory where that language applies and in the language of
numerical analysis where it does not. It is the map to read before changing
anything: the boundaries below are the ones a later refactor should move,
and the ones this one made explicit without moving.

    source category / diagram
            |                       Skeleton: combinatorics, no widths, no torch
            v
    parametric interaction interpretation
            |                       Interpretation / Functor: roles -> Dim,
            v                       names -> torch modules
    CMap wiring and GoI execution
            |                       one round = sigma . Phi
            v
    global transition  T_theta : S -> S
            |
            v
    Schedule / Engine / RecursionEngine / ACTEngine
            |                       how many rounds, and which are differentiated
            v
    task-specific sudoku training
                                    losses, optimizer, clipping, EMA, ACT refill

Three of these layers are **categorical** (syntax, interpretation, wiring),
one is **dynamical** (the transition and its iteration), one is **Torch
realization** (the modules and the forward pass), and the last two are
**policy** (execution, differentiation, optimization). Confusing them is the
failure mode this document exists to prevent.

## 1. Source category and diagram

A model starts as a diagram `D` in some source category `C`. For the sudoku
solvers `C` is `discopy.frobenius`, and the diagram is not drawn by hand: it
is generated from the game's combinatorics by
`discopy.neural.skeleton.from_incidence` (bipartite cell/unit factor graph)
or `from_relation` (pairwise peer clique). `from_diagram` reads a skeleton
off any diagram, so a source category's own combinators can draw the wiring
instead.

What survives from `C` and what does not is the whole point. Swaps, cups,
caps and traces are **wiring** in the target category, so a functor into
`discopy.neural` preserves them strictly and for free — they leave no box
behind, only a different involution. What cannot be preserved for free is a
box whose legs carry a symmetry: a constraint over nine variables, a spider,
a planar node. Those stay boxes, and their equations become properties of a
torch module, measured rather than assumed (§7).

## 2. Skeleton: the combinatorics, on their own

A `Skeleton` is a **closed** `CMap` whose boxes carry no data and whose
atomic types name the *role* a port plays rather than the width it will
carry, together with the `Signature` of each box name. It is pure syntax:
degrees, involution and loop positions can be built and checked on a machine
with no torch at all.

A `Signature` is the single source of truth for the port layout of a box.
`Signature.cod` builds the abstract type, `Signature.loops` gives the traced
pairs the skeleton wires, `Signature.slices` gives the flat offsets the
module reads and writes. Because all three are derived from one declaration,
the type of a box and the cursor arithmetic of its module cannot drift
apart.

## 3. Parametric interaction interpretation

An `Interpretation` sends each role to the `Dim` it carries and each box
name to the torch module computing it; `interpret` applies it port by port.
Since `Dim(0)` is the monoidal unit, an interpretation can **erase** a role:
its ports and the wires on them vanish. That is how one skeleton serves two
models — model A sends the answer role to `Dim(0)`, model C to `Dim(48)`.

### Why a `Network` is not a layer

This is the claim the formal layer exists to make precise.

An ordinary parametric map `(P, f) : X -> Y` is a map `f : P (x) X -> Y`:
parameters and an input in, an output out. That is a feed-forward layer.

A `Network` is not one. Its module maps `R**width -> R**width` for `width`
the sum of the domain *and* codomain dimensions: it reads one incoming
message on **every** port and emits one outgoing message on **every** port.
Writing the boundary of a box as

    d(f) = X* (x) Y

a `Network` is a **parametric interaction map**

    Phi : P (x) d(f) -> d(f).

`discopy.neural.parametric` is where the two are written down and kept
apart: `ParamMap` for the first, `InteractionMap` for the second, with
`interaction_spec(network)` reading the second off a `Network` — read-only,
owning nothing, taking no part in a forward pass.

Two consequences worth stating.

* **A cell answers its inputs.** `cells.Site` broadcasts a fresh belief to
  every leg of its message orbit and re-emits its traced roles; nothing in
  an `X -> Y` signature can say that, which is why the local semantics is
  an interaction and not a layer.
* **`X*` is `X`, but not in the same order.** Every atomic `Dim` is
  self-dual, so `X*` is represented by the same data as `X`. A composite
  type is a different matter: `Dim(2, 3).r == Dim(3, 2)` reverses, whereas
  the module reads its domain ports in domain order — which is exactly what
  `CMap.box_ports` restores when it un-reverses the clockwise storage. So
  `InteractionMap.boundary` is `dom @ cod`: the object `X* (x) Y` up to the
  symmetry putting the duals back in order, in the port order the executable
  layout uses. The dagger of a network reuses its module for the same
  reason: the same weights, read in the new port order.

## 4. CMap wiring and GoI execution

An interpreted skeleton is a closed `CMap`: a finite family of boxes and a
fixpoint-free involution `sigma` on their ports. `CMap.forward` runs
synchronous message passing — every box interacts with the messages on its
own ports, then the wires carry each emission to the other end — which is
the execution formula of the geometry of interaction.

It is vectorized: all messages live in one flat tensor, one round of routing
is a single permutation of its last axis, and every box sharing a module and
a port signature is evaluated in one batched call. On a closed map the
messages are held in *box order* (`CMap._fused_routing`) so that each module
reads a contiguous view; they are permuted back to port order only where the
caller sees them, element for element. `CMap.forward_reference` is the
one-call-per-box oracle the fast path is tested against.

## 5. The global transition

Write `Phi_theta` for the parallel application of every local interaction and
`sigma_D` for the routing permutation. One round is

    T_{D,theta} = sigma_D . Phi_theta  :  S -> S

on the state object `S = (+)_p R**w_p`, one summand per port. When the
initial message vector `i` is re-injected (`inject=True`) the round is

    T_{D,theta,i}(s) = sigma_D(Phi_theta(s)) + i,

an **affine** dependence on `i`: the vector is added back to the *whole*
state after routing, every round. On an open map the boundary input ports
are not owned by a box and re-emit the input `x` instead, so the round reads
`T(s) = sigma_D(Phi_theta(s) (+) x) + i`.

`discopy.neural.dynamics` records this as a `Transition`, and `from_map`
builds one from an interpreted `CMap` without touching a tensor. It is a
specification: `CMap.forward` remains the only implementation.

### Finite iteration, delayed state, trace, fixed point

Four notions, routinely conflated, kept apart here.

* **Finite iteration.** Running `n` rounds computes `T^n(s_0)` and nothing
  more. `Iteration` records it. What holds unconditionally is resumption,
  `T^(a+b) = T^b . T^a`, which is why a segmented outer loop can stop and
  carry on — and it holds for *one* transition, so a run resumed from its
  own carried state only resumes when `inject` is off. That is why the
  segmented schedules pass `inject=False` and carry their inputs on a state
  channel instead, while an `inject=True` schedule runs exactly one
  `advance` per forward pass.
* **A persistent state channel.** A self-wired pair of ports is private
  memory: what a box writes on one end it reads on the other one round
  later. Operationally that is *delayed feedback*, in the sense of
  `discopy.feedback`, not an equation to solve.
* **A categorical trace.** Structurally, that same self-wired pair *is* the
  trace of the compact target — `CMap.from_box(g).trace()` equals the
  explicitly self-wired map, asserted as a doctest in
  `discopy/neural/skeleton.py` — and a functor preserves it strictly,
  because it is wiring. The structural statement is exact; it is a statement
  about the *diagram*, not about what finitely many rounds converge to.
* **A fixed point.** A third thing, and nothing in this package computes
  one. No fixed-point solver is used anywhere. If some `T` happens to be a
  contraction then `T^n` converges, but contractivity is an analytic
  property of learned weights, to be measured — never something the category
  supplies.

## 6. Schedule, Engine, RecursionEngine, ACTEngine

The layers above are semantics. This one is **policy**, and it deliberately
mixes two policies that are conceptually distinct:

* **execution policy** — how many rounds, cycles and steps are evaluated;
* **differentiation policy** — which of those evaluations are in the
  autograd graph.

`Schedule(rounds, cycles, steps, inject, supervise)` names both, and
`Engine.run` executes them with three nested loops:

    for step in range(steps):            # detached from one another
        for cycle in range(cycles):      # only the last one differentiated
            state = rounds of message passing

Exactly:

* `advance(state, rounds)` is the adapter for `T^rounds`: it calls
  `CMap.forward(init=state, n_rounds=rounds, inject=schedule.inject,
  return_flat=True)`, so the state in and the state out are flat port-order
  vectors of `Router.total` numbers.
* each cycle runs under `torch.set_grad_enabled(grad and cycle == cycles - 1
  and torch.is_grad_enabled())`, so with `cycles > 1` the first `cycles - 1`
  cycles are **undifferentiated** — that is what bounds activation memory to
  one segment.
* `refresh` runs at the end of *every* cycle, inside the same grad context.
  For a plain `Engine` it is the identity; for a `RecursionEngine` it is one
  update of the answer trace from the latent state.
* between supervision steps the state is `detach()`ed when
  `supervise == "step"`.

So a **cycle** denotes

    C_theta = R_theta . T_theta^rounds

with `R_theta` the refresh, and a **supervision step** denotes `C_theta^cycles`.
Denotationally a step is `C^cycles`; *differentiably* only the last `C` is in
the graph. With `cycles > 1` no gradient reaches the state the step started
from — nor, therefore, the encoder that built it or the learned initial
answer `y0`. That is not an accident: it is what lets the ACT trainer build
refilled states under `no_grad`.

Two supervision schemes are in use:

| | schedule | denotation | what is differentiated |
|---|---|---|---|
| single run (models A, B) | `Schedule(r, 1, 1, inject=True, supervise="round")` | `T^r` | all `r` rounds, one backward graph, a loss on every round |
| segmented recursion (model C) | `Schedule(r, c, s, inject=False, supervise="step")` | `(R . T^r)^(c*s)` | the last cycle of each of `s` detached segments |

`RecursionEngine` adds the answer trace: a loop the sites read but never
write, refreshed between cycles, which the readout decodes instead of the
latent state. `ACTEngine` adds a halt head on top; its inherited `step` and
`forward` are untouched, so fixed-compute evaluation of an ACT model is
exactly fixed-compute evaluation of the recursion it was built from.

## 7. Laws: what a learned cell actually promises

`discopy.neural.laws` reads a signature's symmetry as a group action
`rho_X : G -> Aut(X)` and the promise a module makes as equivariance,

    F . rho_X(g) = rho_Y(g) . F.

`Sym.NONE`, `Sym.CYCLIC` and `Sym.PERM` become `Action`s with the
corresponding generators; `Signature.generators` is the same group acting on
ports, and `check_equivariant` is the executable diagnostic that measures
the residual.

Being honest about strength is the point, and `Strictness` names it:

* **strict** — holds by construction. Swaps, cups, caps and traces are
  wiring, preserved exactly and for free.
* **lax** — holds up to the reordering of a floating-point reduction. That
  is what a pooled cell gives: `Relation` and `Site` are permutation
  equivariant because they pool symmetrically, and the residual
  `check_equivariant` reports is rounding error.
* **approximate** — neither; only measured or regularized.

And the negative result, stated rather than glossed: **permutation
equivariance is not Frobenius structure.** A learned `Relation` commutes
with permutations of its members and still does not fuse.
`cells.fusion_residual` measures how far it is from the fusion law and the
answer is not zero. Satisfying the symmetry a signature declares says
nothing about the other equations of whatever algebraic theory one had in
mind.

## 8. Task-specific training

Nothing above is sudoku's. What this folder adds is a *study*: `core/` holds
the harness (`train_epoch`, `evaluate`, the solver templates and their
RNG-load-bearing parts order, the registry, the recipes), and `sudoku/`
brings only what is irreducibly sudoku — the grid combinatorics, the roles
its wires carry, an encoder, a decoder, the benchmarks, the recorded
configurations. See [README.md](README.md) for the models and the results.

The ACT trainer is the one place where policy gets genuinely intricate, so
its order is worth writing out. Per iteration of `ACTTrainer._run_raw`:

1. `act_step` — `cycles - 1` cycles under `no_grad`, then one differentiated
   cycle, then `answer = router.read(state, answer_heads)`,
   `logits = decoder(answer)`, and `halt = q_head.read(answer.detach() if
   halt_detach else answer)`. So `halt_detach` decides whether the head's
   loss trains the head alone or also the trunk;
2. `loss = cross_entropy(logits, targets) + halt_weight * halt_loss`, where
   the halt loss is a binary cross-entropy of the head against *correctness*
   computed under `no_grad`;
3. `zero_grad(set_to_none=True)`, `backward()`,
   `clip_grad_norm_(parameters, grad_clip)`, `optimizer.step()`,
   `scheduler.step()`, then the EMA update — in that order;
4. under `no_grad`: a slot halts when its halt logit clears the threshold or
   it hits `n_sup`; halted slots are refilled from the `ExampleStream`
   *within the same batch*, their state rebuilt by `initial` and their step
   counter reset. The refill is computed on the device with a cumulative sum
   over the halt mask, so the loop stays free of host round-trips.

Evaluation comes in three flavours, and they are not interchangeable:
fixed-compute (`Engine.forward`), adaptive with early stopping
(`evaluate_act`), and best-of-k selected by the halt logit
(`evaluate_selected`) — the halt head used as a learned verifier. Stopping
early changes the predictions, which is the point, so both fixed and
adaptive numbers are worth reporting.

## What guards what

| claim | guarded by |
|---|---|
| the recorded models compute what they always did | `test/neural/test_equivalence.py` against `golden/`, bitwise |
| the fused forward equals the one-call-per-box oracle | `test_forward_reference`, `test_general.py` |
| the formal layer is torch-free | `test_formal.py::test_formal_layer_does_not_import_torch` |
| `interaction_spec` reads a `Network` without touching it | `test_formal.py::test_interaction_spec_changes_nothing` |
| `from_map` agrees with the map's width and routing | `test_formal.py::test_transition_agrees_with_the_map` |
| `T(s) = sigma(Phi(s)) + i` | `test_formal.py::test_reinjection_is_an_affine_shift` |
| `T^(a+b) = T^b . T^a`, bitwise | `test_formal.py::test_iteration_is_resumption` |
| the actions agree with `Sym` | `test_formal.py::test_actions_agree_with_sym` |
| the detach boundary of a segment is where it says | `test_sudoku_smoke.py::test_only_the_last_cycle_is_differentiated` |
| `halt_detach` cuts the trunk and nothing else | `test_sudoku_smoke.py::test_halt_detach_cuts_the_trunk` |
| the whole training machinery runs end to end | `test_sudoku_smoke.py`, in a few seconds |
