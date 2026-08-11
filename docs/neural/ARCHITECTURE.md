# The architecture, layer by layer

[README.md](README.md) explains the workflow. This document says what each
layer *is*, in the language of category theory where that language applies
and in the language of numerical analysis where it does not. It is the map
to read before changing anything.

    source category / diagram
            |                    an ordinary DisCoPy diagram or CMap whose
            v                    atomic types name roles
    interpretation  F_theta
            |                    roles -> Dim, generator names -> modules
            v                    (MapNN.ob, MapNN.ar)
    compiled global interaction
            |                    T_{D,theta} = sigma_D . Phi_theta : S_D -> S_D
            v
    solver
            |                    how many rounds, and which are differentiated
            v
    task: encoder, readout, loss, optimizer
                                 outside the library, by design

Two of these layers are **categorical** (the diagram and its
interpretation), one is **dynamical** (the transition and its iteration),
one is **policy** (execution and differentiation), and the last is the
user's. Confusing them is the failure mode this document exists to
prevent.

## 1. The diagram

A model starts as a diagram `D` in some source category `C`, whose atomic
types are *roles*: names for what a wire carries, not for how wide it is.
`C` is `discopy.frobenius` for the sudoku example. The diagram need not be
drawn by hand — `discopy.neural.signature.from_incidence` (a bipartite
incidence graph) and `from_relation` (the graph of a binary relation) draw
one out of a family's combinatorics — but it may be, and `MapNN.compile`
reads a diagram through `CMap.from_diagram` either way.

What survives from `C` and what does not is the whole point. Swaps, cups,
caps and traces are **wiring** in the target category, so a functor into
`discopy.neural` preserves them strictly and for free — they leave no box
behind, only a different involution. What cannot be preserved for free is a
box whose legs carry a symmetry: a constraint over nine variables, a spider,
a planar node. Those stay boxes, and their equations become properties of a
torch module, measured rather than assumed (§5).

## 2. The interpretation

`MapNN(ob, ar)` is the functor `F_theta`. `ob` sends each atomic role to the
`Dim` it carries; `ar` sends each generator *name* to the torch module
computing it, shared by every site of that name. `map.interpret` applies it
port by port. Since `Dim(0)` is the monoidal unit, an interpretation can
**erase** a role: its ports and the wires on them vanish. That is how one
diagram serves two models — model A of the example sends the answer role to
`Dim(0)`, model C to `Dim(48)`.

### Why a `Network` is not a layer

This is the claim the formal specifications exist to make precise.

An ordinary parametric map `(P, f) : X -> Y` is a map `f : P (x) X -> Y`:
parameters and an input in, an output out. That is a feed-forward layer, and
those are the morphisms of `Para`; `ParamMap` records them and they compose
by substitution.

A `Network` is not one. Its module maps `R**width -> R**width` for `width`
the sum of the domain *and* codomain dimensions: it reads one incoming
message on **every** port and emits one outgoing message on **every** port.
Writing the boundary of a box as

    d(f) = X* (x) Y

a `Network` is a **parametric interaction map**

    Phi_f : P_f (x) d(f) -> d(f).

`InteractionMap` records these, `interaction_spec(network)` reads one off a
`Network` — read-only, owning nothing, taking no part in a forward pass —
and `InteractionMap.__rshift__` **raises**. That refusal is load-bearing:
two interaction maps glued along a shared object do not compose by
substitution, they talk to each other along the wires, which is symmetric
feedback — the trace of the two boxes over the shared boundary — and what
computes it is a finite number of rounds. Their *tensor* is kept, because
`Phi_theta` is exactly the parallel application of every local interaction.

Two consequences worth stating.

* **A cell answers its inputs.** `cells.Site` broadcasts a fresh belief to
  every leg of its message orbit and re-emits its traced roles; nothing in
  an `X -> Y` signature can say that.
* **`X*` is `X`, but not in the same order.** Every atomic `Dim` is
  self-dual, so `X*` is represented by the same data as `X`. A composite
  type is a different matter: `Dim(2, 3).r == Dim(3, 2)` reverses, whereas
  the module reads its domain ports in domain order — which is exactly what
  `core.box_ports` restores when it un-reverses the clockwise storage. So
  `InteractionMap.boundary` is `dom @ cod`. The dagger of a network reuses
  its module for the same reason: the same weights, read in the new port
  order.

## 3. The compiled interaction

`MapNN.compile(diagram)` returns an `Interaction`: a closed `CMap` — a
finite family of boxes and a fixpoint-free involution `sigma` on their ports
— together with the port index of every `(generator name, role)` family.
`CMap.forward` runs synchronous message passing, which is the execution
formula of the geometry of interaction; `Interaction.advance(state, n)` is
the adapter that runs `n` rounds from a flat state back to a flat state.

Write `Phi_theta` for the parallel application of every local interaction
and `sigma_D` for the routing permutation. One round is

    T_{D,theta} = sigma_D . Phi_theta  :  S -> S

on the state object `S = (+)_p R**w_p`, one summand per port. When the
initial message vector `i` is re-injected (`inject=True`) the round is

    T_{D,theta,i}(s) = sigma_D(Phi_theta(s)) + i,

an **affine** dependence on `i`: the vector is added back to the *whole*
state after routing, every round.

The forward pass is vectorized: all messages live in one flat tensor, one
round of routing is a single permutation of its last axis, and every box
sharing a module and a port signature is evaluated in one batched call. On a
closed map the messages are held in *box order* (`CMap._fused_routing`) so
that each module reads a contiguous view; they are permuted back to port
order only where the caller sees them, element for element.
`CMap.forward_reference` is the one-call-per-box oracle the fast path is
tested against.

A state is addressed by family, never by offset: `Interaction.read(state,
key)` gives `(rows, sites, width)` reading one port per traced pair, and
`Interaction.write(state, key, values)` writes one value per site to *every*
copy of its trace. A port is a **head** — one a module reads a value off —
unless it is wired to an earlier port of the same box, which is exactly the
second copy of a traced leg; the wiring says so, no declaration is
consulted.

### Trace, delayed state, finite iteration, fixed point

Four notions, routinely conflated, kept apart here.

* **A categorical trace.** A self-wired pair of ports *is* the trace of the
  compact target — `CMap.from_box(g).trace()` equals the explicitly
  self-wired map, asserted as a doctest in `discopy/neural/signature.py` —
  and a functor preserves it strictly, because it is wiring. The structural
  statement is exact; it is a statement about the *diagram*, not about what
  finitely many rounds converge to.
* **A persistent state channel.** That same pair is private memory: what a
  box writes on one end it reads on the other one round later.
  Operationally that is *delayed feedback*, in the sense of
  `discopy.feedback`, not an equation to solve.
* **Finite iteration.** Running `n` rounds computes `T^n(s_0)` and nothing
  more. What holds unconditionally is resumption, `T^(a+b) = T^b . T^a`,
  which is why a segmented solver can stop and carry on — and it holds for
  *one* transition, so a run resumed from its own carried state only
  resumes when `inject` is off. That is why the segmented solvers pass
  `inject=False` and carry their inputs on a state channel instead, while
  an `inject=True` solver runs exactly one `advance` per forward pass.
* **A fixed point.** A fourth thing. `FixedPoint` is the solver that looks
  for one and `Interaction.residual` is `||T(s) - s||_inf`, the number that
  says whether it found one. If some `T` happens to be a contraction then
  `T^n` converges, but contractivity is an analytic property of learned
  weights, to be measured — never something the category supplies.

## 4. The solver

The layers above are semantics. This one is **policy**, and it deliberately
names two policies that are conceptually distinct:

* **execution policy** — how many rounds, cycles and steps are evaluated;
* **differentiation policy** — which of those evaluations are in the
  autograd graph.

Every solver runs from a flat state to a list of *checkpoints*, the states a
loss may look at; the caller decodes them.

| solver | denotation | what is differentiated |
|---|---|---|
| `Iterate(r, inject=True)` | `T^r` | all `r` rounds, one backward graph; with `deep`, a checkpoint per round |
| `FixedPoint(r, tol, backward="last")` | `T^k` for the first `k <= r` with residual under `tol` | one round from the detached limit — the Jacobian-free one-step gradient |
| `FixedPoint(..., backward="full")` | same | every round actually run |
| `Recursion(r, c, s)` | `(R . T^r)^(c*s)` | the last cycle of each of `s` detached segments |
| `ACT(...)` | the same, plus a halt head | the same |

`Recursion.step` is the unit a training loop takes an optimizer step on:

    for cycle in range(cycles):        # only the last differentiated
        state = interaction.advance(state, rounds)
        state = refresh(state)

so a **cycle** denotes `C_theta = R_theta . T_theta^rounds` with `R_theta`
the `Refresh` — one update of a trace the generators read but never write —
and a **supervision step** denotes `C_theta^cycles`. Denotationally a step
is `C^cycles`; *differentiably* only the last `C` is in the graph, which is
what bounds activation memory to one segment. With `cycles > 1` no gradient
reaches the state the step started from, nor the encoder that built it —
which is what lets an ACT training loop build refilled states under
`no_grad`.

`ACT.step` returns `(state, answer, halt)`: the trace the step ended on and
the halt output read off it. The *halting rule* — when to stop, what to do
with a finished example, how to select between rollouts — is a training or
evaluation policy and lives in the caller, which for the example is
`examples/sudoku/train.py` and `evaluate.py`.

## 5. Laws: what a learned cell actually promises

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
`laws.fusion_residual` measures how far it is from the fusion law and the
answer is not zero. Satisfying the symmetry a signature declares says
nothing about the other equations of whatever algebraic theory one had in
mind.

## 6. Where the task lives

Nothing above is sudoku's, and nothing above is training's. The library
produces states; the caller encodes inputs into the initial state, decodes
checkpoints into predictions, computes a loss and steps an optimizer. The
example carries the four artefacts a task actually brings — combinatorics,
signatures, encoder, readout — plus its protocol; see
[`examples/sudoku`](examples/sudoku/).
