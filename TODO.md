# TODO — Phase 1: formalize `discopy.neural`, do not rewrite the numerics

## The prompt, verbatim

> You are refactoring the `discopy.neural` and `docs/neural` subpackages of the branch currently provided to you.
>
> This is a semantics-first, behavior-preserving intermediate refactor. It is NOT the final architectural rewrite.
>
> Your job is to make the existing implementation formally coherent with a categorical account of parametric neural architectures while preserving the current implementation as the unique numerical source of truth.
>
> The existing implementation is valuable. Do not replace it merely because a cleaner abstraction is imaginable. In particular, the current combinatorial-map execution, Router/state layout, module sharing, schedule semantics, TRM recursion, segmented autodifferentiation, ACT machinery, batching, and Sudoku training behavior are load-bearing.
>
> Before changing code, read in full:
>
> - every file under `discopy/neural`;
>
> - every file under `docs/neural`;
>
> - every test that imports, constructs, executes, serializes, trains, differentiates, compiles, batches, or otherwise depends on `discopy.neural`;
>
> - all Sudoku/TRM/ACT examples, training scripts, configuration objects, factories, notebooks or documentation code that exercise these APIs.
>
> Do not infer behavior from names. Trace actual call graphs and tensor transformations.
>
> 1. PRIMARY GOAL
>
> Introduce a formal semantic layer that makes the following decomposition explicit:
>
>     DisCoPy categorical syntax
>
>             ↓
>
>     parametric local interaction semantics
>
>             ↓
>
>     CMap / Geometry-of-Interaction global transition
>
>             ↓
>
>     execution + differentiation policy
>
>             ↓
>
>     task/training machinery
>
> The existing executable implementation must remain the numerical source of truth during this phase.
>
> The intended mathematical reading is:
>
> A source diagram D lives in some DisCoPy category C.
>
> Each generating box f : X → Y is interpreted as a parametric local interaction map on its boundary, not naively as an ordinary feed-forward map X → Y.
>
> Write the boundary as
>
>     ∂f = X* ⊗ Y.
>
> A local neural interaction has the form
>
>     Φ_f : P_f ⊗ ∂f → ∂f,
>
> where P_f is the parameter object.
>
> For the current self-dual finite-dimensional representation, X* is represented using the same `Dim` data as X, so the executable Torch module acts on the flattened collection of incident port messages.
>
> For a complete interpreted combinatorial map, let σ_D denote the routing permutation/involution induced by the wiring and let Φ_θ denote the parallel application of all local interaction maps. One synchronous round is conceptually
>
>     T_{D,θ} = σ_D ∘ Φ_θ.
>
> Where the current implementation reinjects an initialization/input vector i, describe the actual transition faithfully, e.g.
>
>     T_{D,θ,i}(s) = σ_D(Φ_θ(s)) + i
>
> if and only if that is exactly what the current code computes.
>
> Do not change arithmetic to make it fit notation. Change the notation if necessary to fit the implementation.
>
> 2. NON-NEGOTIABLE BEHAVIORAL INVARIANTS
>
> During Phase 1, preserve all existing behavior unless you discover an actual pre-existing bug and can prove it independently. Do not silently “fix” behavior while refactoring.
>
> Preserve:
>
> - tensor shapes at every public and internal numerical boundary;
>
> - dtype behavior;
>
> - device behavior;
>
> - state flattening and unflattening;
>
> - port ordering;
>
> - Router offsets and slices;
>
> - CMap routing permutations;
>
> - fused/optimized routing representations;
>
> - grouping of boxes/cells;
>
> - module identity and parameter sharing;
>
> - module construction order;
>
> - parameter registration order;
>
> - parameter initialization under a fixed random seed;
>
> - `state_dict` key names and serialization behavior where currently tested or relied upon;
>
> - train/eval propagation;
>
> - `.to(...)` behavior;
>
> - eager behavior;
>
> - `torch.compile` behavior and compilation boundaries;
>
> - batching semantics;
>
> - bind/unbind semantics;
>
> - `Dim(0)` and erased-port behavior;
>
> - dagger/dual behavior;
>
> - `Signature` slicing/layout semantics;
>
> - all symmetry/equivariance checks;
>
> - all injection semantics;
>
> - all round/cycle/step semantics;
>
> - all `detach`, `no_grad`, and autograd boundaries;
>
> - deep-supervision behavior;
>
> - TRM refresh placement and exact state mutation;
>
> - ACT halting inputs, detach options, loss construction, slot refill, optimizer order, gradient clipping, scheduler order and EMA order;
>
> - test-time compute modifications;
>
> - all existing Sudoku construction and training entry points.
>
> Do not move registered `nn.Module` objects under new parent modules in this phase. Doing so can change state-dict paths and registration order.
>
> Do not reorder dictionaries/factories/module creation where random initialization or parameter ordering can depend on order.
>
> Do not replace a finite iterative computation by a fixed-point solve.
>
> Do not replace delayed/self-wired state with a different notion of categorical trace.
>
> Do not add “safety checks” that skip invalid states without first determining why those states exist. Fix root causes only.
>
> 3. REQUIRED PRE-REFACTOR ANALYSIS
>
> Before editing, produce an internal dependency map.
>
> For every class/function in `discopy.neural`, determine:
>
> - whether it is pure categorical/combinatorial structure;
>
> - whether it owns Torch parameters;
>
> - whether it changes tensor layout;
>
> - whether it participates in forward execution;
>
> - whether it participates in gradient construction;
>
> - whether it participates in training policy;
>
> - whether it participates in serialization;
>
> - whether it is task-specific;
>
> - who calls it.
>
> Explicitly trace at least these paths, using the actual repository names and implementation:
>
> - source diagram/Skeleton → Interpretation/Functor → Wiring/CMap;
>
> - `Network` construction and module ownership;
>
> - one `CMap.forward` round;
>
> - multi-round `CMap.forward`;
>
> - `return_rounds`;
>
> - `inject=True` and `inject=False`;
>
> - `Engine.advance`;
>
> - standard fixed-depth/deep-supervised training;
>
> - `RecursionEngine` refresh;
>
> - TRM cycles and steps;
>
> - exact locations where gradients are disabled/detached;
>
> - ACT `act_step`;
>
> - adaptive ACT evaluation;
>
> - ACT training with slot refill;
>
> - structural batching;
>
> - eager versus compiled execution;
>
> - checkpoint/state-dict save/load;
>
> - Sudoku model construction.
>
> For each tensor transformation, record the shape symbolically. Never assume a reshape/view is harmless; verify element order.
>
> 4. ADD THE FORMAL LAYER — DO NOT REIMPLEMENT EXECUTION
>
> Add a small Torch-free semantic layer. Prefer minimal files and minimal dependencies.
>
> Recommended new files:
>
>     discopy/neural/parametric.py
>
>     discopy/neural/dynamics.py
>
>     discopy/neural/laws.py
>
> The exact names may change if repository conventions strongly suggest better names, but preserve the conceptual separation.
>
> 4.1 `parametric.py`
>
> This file must not import Torch.
>
> Define formal/specification-level concepts for parametric maps.
>
> At minimum distinguish:
>
> A. Ordinary parametric map
>
>     (P, f) : X → Y
>
>     f : P ⊗ X → Y
>
> B. Parametric interaction map
>
>     (P, Φ) : X → Y
>
>     Φ : P ⊗ (X* ⊗ Y) → (X* ⊗ Y)
>
> The second concept is the one that most faithfully describes the current executable `Network`.
>
> These objects are semantic specifications, not a second numerical backend.
>
> Do NOT implement a duplicate `forward()` path that numerically competes with `Network` or `CMap`.
>
> Represent enough metadata to state:
>
> - domain;
>
> - codomain;
>
> - parameter specification;
>
> - operation/generator identity;
>
> - optional laws/constraints.
>
> Keep the representation backend-agnostic.
>
> Document formal composition and monoidal product carefully, including parameter-object ordering. Do not force the current executable implementation to use these operations yet.
>
> If composition of formal `ParamMap`s is implemented, test its purely structural behavior independently. It must not alter existing Sudoku execution.
>
> 4.2 Relationship to current `Network`
>
> Do not move or rewrite `Network` in Phase 1.
>
> Document it as a Torch realization of a parametric interaction map.
>
> If useful, add a side-effect-free helper such as
>
>     interaction_spec(network)
>
> or a read-only property returning a formal specification.
>
> The helper must:
>
> - not own parameters;
>
> - not wrap/re-register the module;
>
> - not affect equality/hash behavior unless deliberately tested;
>
> - not participate in `forward`;
>
> - not affect serialization.
>
> Do not make `Network` inherit from a new Torch or formal base class merely for aesthetics unless you prove this cannot change any behavior. Prefer composition-free metadata helpers.
>
> 4.3 `dynamics.py`
>
> This file formalizes the global dynamical interpretation.
>
> Introduce a formal notion of a transition
>
>     T_θ : S → S
>
> or, when external input/reinjection is semantically part of a round,
>
>     T_θ : I ⊗ S → S
>
> depending on the exact current implementation.
>
> The formal object must describe:
>
> - state object/width;
>
> - transition semantics;
>
> - optional external/reinjected input;
>
> - relationship to the interpreted CMap.
>
> Do not duplicate `CMap.forward`.
>
> If an executable adapter is introduced, it must delegate directly to the existing implementation. For example, an adapter may call the existing `CMap` with exactly the same arguments, but it must not reproduce the routing arithmetic independently.
>
> The current CMap remains the numerical implementation.
>
> State clearly that repeated rounds compute finite iteration:
>
>     T^n(s_0),
>
> not a fixed point.
>
> 4.4 `laws.py`
>
> Do not split or rewrite the current `Signature`, `Orbit`, or `Sym` data structures in Phase 1.
>
> Instead give their existing values a formal interpretation.
>
> Where appropriate, describe symmetry as a group action
>
>     ρ_X : G → Aut(X)
>
> and admissibility/equivariance as a commuting law
>
>     F ∘ ρ_X(g) = ρ_Y(g) ∘ F.
>
> Map the existing `Sym.NONE`, `Sym.CYCLIC`, and `Sym.PERM` semantics to formal action specifications without changing their current runtime behavior.
>
> Existing numerical equivariance checks remain the executable diagnostics.
>
> Be mathematically honest:
>
> - permutation equivariance is not automatically Frobenius algebra structure;
>
> - satisfying a symmetry constraint is not the same as preserving all equations of a source algebraic theory;
>
> - if a cell only approximately satisfies a law, call it approximate/lax/regularized rather than strict.
>
> 5. KEEP THE CURRENT RUNTIME CLASSES IN PLACE
>
> During Phase 1, do not reorganize the package into a `neural.torch` subpackage.
>
> Specifically, keep the current locations and public import paths of:
>
> - `Network`;
>
> - `CMap`;
>
> - `Engine`;
>
> - `RecursionEngine`;
>
> - `ACTEngine`;
>
> - current cells such as `Site`, `Relation`, `Gate`, `Cyclic`;
>
> - `Signature`, `Orbit`, `Sym`;
>
> - batching machinery;
>
> - current Functor/Interpretation machinery.
>
> The eventual architecture may move these. Phase 1 does not.
>
> The purpose of Phase 1 is to make the mathematical boundaries explicit before changing ownership boundaries.
>
> 6. FORMALIZE THE EXISTING SOLVER/TRAINING MACHINERY WITHOUT CHANGING IT
>
> Do not force TRM or ACT into the categorical core.
>
> Separate the following ideas conceptually:
>
> A. Denotational/local semantics:
>
>     local parametric interaction maps.
>
> B. Global state dynamics:
>
>     T_θ : S → S induced by CMap/GoI.
>
> C. Execution policy:
>
>     how many rounds/cycles/steps are evaluated.
>
> D. Differentiation policy:
>
>     which evaluations are part of the autograd graph.
>
> E. Observation/readout:
>
>     how logits/answers/halting values are extracted.
>
> F. Optimization/training:
>
>     losses, optimizer steps, clipping, scheduler, EMA, ACT slot refill.
>
> The current `Schedule` may combine C and D. That is acceptable in Phase 1. Do not replace it simply to achieve conceptual purity.
>
> Document its exact semantics.
>
> For ordinary finite execution:
>
>     s_{t+1} = T_θ(s_t).
>
> For a recursive/TRM cycle, derive the exact expression from the code. If the code computes
>
>     C_θ = R_θ ∘ T_θ^r,
>
> state that.
>
> For multiple cycles, distinguish the denotational value
>
>     C_θ^c(s)
>
> from truncated differentiation when earlier cycles are computed under `no_grad` or detached.
>
> State explicitly which cycle/step is differentiated.
>
> Do not move the `detach` boundary.
>
> Do not move `refresh`.
>
> Do not alter the number of calls to the transition.
>
> Do not alter supervision placement.
>
> 7. ACT
>
> Treat ACT as an execution/training policy over the same underlying state dynamics, not as a categorical primitive.
>
> Read the actual code and formalize:
>
> - the state passed to the halt head;
>
> - the answer/readout used for logits;
>
> - whether the halt head sees a detached or non-detached tensor under each option;
>
> - the exact meaning of `halt_detach`;
>
> - the exact order of undifferentiated and differentiated cycles;
>
> - the stopping criterion;
>
> - maximum compute;
>
> - slot refill;
>
> - loss terms;
>
> - optimizer, clipping, scheduler and EMA order.
>
> Do not simplify ACT into generic pseudocode if doing so hides behavior that affects gradients.
>
> 8. TRACE, FEEDBACK AND STATE TERMINOLOGY
>
> Do not change current self-wired/traced execution semantics.
>
> However, correct the documentation if it conflates:
>
> - categorical trace;
>
> - fixed-point feedback;
>
> - delayed feedback/state;
>
> - self-wired GoI interaction channels.
>
> A self-wired port that carries state across finite execution rounds should be described operationally as a persistent feedback/state channel unless the implementation actually computes a categorical trace according to a specified traced semantics.
>
> Do not introduce a fixed-point solver in Phase 1.
>
> Do not claim Banach convergence merely because finite iteration exists.
>
> If contractivity is supported or tested, describe it as an additional analytic condition, not something supplied automatically by category theory.
>
> 9. TESTING STRATEGY
>
> All existing tests must pass after Phase 1.
>
> Do not weaken tolerances or delete tests merely because the architecture changed.
>
> Where a test is too tightly coupled to an implementation detail, preserve it during Phase 1 unless the tested detail is genuinely removed. Since this phase should not change runtime ownership/layout, most such tests should remain unchanged.
>
> Add tests for the new formal layer that are orthogonal to numerical execution:
>
> - importing `parametric.py`, `dynamics.py`, and `laws.py` must not import Torch if practical under the package's lazy import design;
>
> - formal objects have correct domain/codomain/parameter metadata;
>
> - formal parameter ordering under composition/tensor is deterministic;
>
> - `interaction_spec(Network(...))` reproduces the exact formal boundary metadata without altering the module;
>
> - formal transition metadata agrees with CMap state width and routing metadata;
>
> - symmetry/action specifications agree with existing `Sym` semantics.
>
> 10. ADD A QUICK END-TO-END SUDOKU/TRM/ACT SMOKE TEST
>
> Add a CI-friendly end-to-end test that verifies the actual training machinery, not only isolated units.
>
> Target runtime: comfortably below five minutes on the expected CI/development machine. Prefer much less if possible.
>
> Do not make the test depend on solving full Sudoku reliably from random initialization. That would be statistically brittle and slow.
>
> Instead construct the smallest representative Sudoku training configuration that exercises the real production code paths.
>
> The smoke test must exercise, using the existing public construction/training machinery:
>
> - creation of a tiny Sudoku model;
>
> - a tiny dataset or deterministic subset;
>
> - forward execution;
>
> - loss construction;
>
> - backward;
>
> - optimizer step;
>
> - TRM segmented recursion;
>
> - at least one detached/non-differentiated cycle when TRM supports it;
>
> - refresh machinery;
>
> - ACT training path;
>
> - halt head;
>
> - ACT `halt_detach` behavior for at least one configuration, with a focused gradient test for the alternative if running both end-to-end is too expensive;
>
> - adaptive or bounded ACT execution;
>
> - state/slot refill if this is part of the actual ACT trainer;
>
> - fixed-compute non-ACT execution;
>
> - parameter finiteness after training;
>
> - loss finiteness;
>
> - expected gradient presence/absence at the critical detach boundaries.
>
> Prefer deterministic assertions such as:
>
> - loss is finite;
>
> - parameters change after an optimizer step;
>
> - selected parameters receive gradients;
>
> - selected tensors/parameters do not receive gradients across intentional detach boundaries;
>
> - ACT probabilities/logits are finite and correctly shaped;
>
> - state shapes are correct after every cycle;
>
> - no NaNs/Infs;
>
> - checkpoint round-trip reproduces outputs;
>
> - the same tiny batch can execute under the principal configurations.
>
> If a short deterministic overfit is feasible, add a tiny synthetic/very-small Sudoku subset on which loss decreases over a handful of steps. Treat this as an additional smoke assertion, not the sole correctness criterion.
>
> Seed all randomness.
>
> Keep the test small enough for regular use.
>
> 11. EQUIVALENCE GATES
>
> Before and after Phase 1, run and compare:
>
> - full existing test suite;
>
> - neural-specific test suite;
>
> - exact/equivalence/golden tests;
>
> - new formal-layer tests;
>
> - new Sudoku/TRM/ACT smoke test.
>
> Where existing golden tests compare:
>
> - routing fingerprints;
>
> - parameter names;
>
> - parameter shapes;
>
> - initialization values;
>
> - forward outputs;
>
> - gradients;
>
> - short optimization trajectories,
>
> they must continue to pass unchanged during this phase.
>
> If any fail, assume the refactor is wrong until you can demonstrate otherwise.
>
> Never “solve” a failure by loosening tolerances before understanding the arithmetic difference.
>
> 12. DOCUMENTATION DELIVERABLE
>
> Rewrite/add a concise `docs/neural` architecture document explaining the following pipeline:
>
>     source category / diagram
>
>             ↓
>
>     Skeleton / categorical combinatorics
>
>             ↓
>
>     parametric interaction interpretation
>
>             ↓
>
>     CMap wiring and GoI execution
>
>             ↓
>
>     global transition T_θ
>
>             ↓
>
>     Schedule / Engine / RecursionEngine / ACTEngine
>
>             ↓
>
>     task-specific Sudoku training
>
> Explain why the current `Network` is an interaction map on a box boundary rather than merely a feed-forward X → Y layer.
>
> Explain which parts are categorical, which are analytic/dynamical, which are Torch realization, and which are training policy.
>
> Use the language of category theory precisely but keep the document readable.
>
> Do not overclaim strict functorial preservation of Frobenius or other algebraic laws when the actual cells only satisfy symmetry constraints or approximate laws.
>
> 13. IMPLEMENTATION DISCIPLINE
>
> Work in small commits or logically isolated patches.
>
> For every proposed edit, ask:
>
> 1. Does this change a tensor value?
>
> 2. Does this change tensor order?
>
> 3. Does this change parameter ownership?
>
> 4. Does this change module registration order?
>
> 5. Does this change random-number consumption?
>
> 6. Does this change a `state_dict` key?
>
> 7. Does this change the autograd graph?
>
> 8. Does this change the number/order of forward calls?
>
> 9. Does this change `no_grad`/detach placement?
>
> 10. Does this change train/eval/device/compile behavior?
>
> 11. Does this change public imports?
>
> 12. Does this change checkpoint compatibility?
>
> 13. Does this change Sudoku construction or training?
>
> If the answer to any is “yes” or “possibly”, do not make the change as part of Phase 1 unless it is strictly necessary and you have a focused equivalence test proving the intended behavior.
>
> Do not perform cosmetic large-scale renames at the same time as semantic work.
>
> Do not introduce parallel implementations of the same numerical operation.
>
> Do not hide existing complexity behind abstractions that make the actual gradient semantics harder to inspect.
>
> 14. FINAL ACCEPTANCE CRITERIA
>
> Phase 1 is complete only when:
>
> - the categorical/parametric/GoI/execution boundaries are explicit in code and docs;
>
> - a Torch-free formal representation of parametric maps and interaction maps exists;
>
> - a formal representation of the global transition exists;
>
> - existing `Network`, `CMap`, `Signature`, cells, batching, `Schedule`, `Engine`, TRM and ACT numerical paths remain intact;
>
> - the existing public Sudoku training path still works;
>
> - all existing tests pass;
>
> - the new formal tests pass;
>
> - the new quick Sudoku/TRM/ACT smoke test passes;
>
> - no equivalence test has been weakened;
>
> - no numerical discrepancy has been dismissed without root-cause analysis;
>
> - documentation accuratelyistinguishes finite iteration, delayed/state feedback, categorical trace, and fixed-point semantics.
>
> 15. OUTPUT FROM YOU
>
> Before editing, provide a short refactor plan listing:
>
> - files to add;
>
> - existing files to touch;
>
> - files explicitly frozen;
>
> - invariants at risk;
>
> - tests guarding each risk.
>
> Then implement the plan incrementally.
>
> After implementation, report:
>
> - exact files changed;
>
> - why each change is semantics-only or behavior-preserving;
>
> - tests run and their results;
>
> - Sudoku/TRM/ACT smoke-test configuration and runtime;
>
> - any remaining architectural debt reserved for Phase 2.
>
> If any test or training path fails, stop and diagnose the root cause. Do not continue stacking refactors on top of an unexplained discrepancy.
>
> The governing principle is:
>
>     FORMALIZE FIRST; DO NOT REWRITE THE NUMERICS.
>
> Phase 1 should leave us with a codebase whose existing behavior is substantially easier to describe categorically, and whose later deep refactor can be performed against explicit semantic interfaces and strong end-to-end regression tests.

## The work

- [x] Read every file under `discopy/neural`, `docs/neural` and every test
      that depends on them; trace the forward, routing, schedule, refresh,
      detach, ACT and batching paths against the code rather than the names.
- [x] Add `discopy/neural/parametric.py`: `Parametric`, `ParamMap`,
      `InteractionMap`, `interaction_spec`. Torch-free, no forward pass.
- [x] Add `discopy/neural/dynamics.py`: `Transition`, `Iteration`,
      `from_map`. Torch-free, delegates to nothing and duplicates nothing.
- [x] Add `discopy/neural/laws.py`: `Strictness`, `Action`, `Law`, `action`,
      `symmetry`, reading `Sym`/`Orbit`/`Signature` without changing them.
- [x] Make `signature._generators` public as `leg_generators`, so the formal
      layer reads the group rather than reimplementing it.
- [x] Correct the docs that conflated categorical trace, delayed state and
      fixed points (`core.py` module docstring, `Signature.loops`).
- [x] Export the three modules from `discopy/neural/__init__.py`.
- [x] Add `test/neural/test_formal.py`.
- [x] Add `test/neural/test_sudoku_smoke.py`.
- [x] Add `docs/neural/ARCHITECTURE.md` and point `README.md` at it.
- [x] Record in `NOTES.md` what was noticed and deliberately not touched.
- [x] Equivalence gate: clean-HEAD baseline versus post-change run, on the
      full suite and on a fingerprint of what the four models actually
      compute (structure, parameters, logits, gradients, 20-step loss
      trajectories, float32 and float64).
- [ ] Phase 2, deliberately out of scope: moving the runtime classes under a
      `neural.torch` subpackage; splitting `Schedule` into an execution
      policy and a differentiation policy; making the executable `Network`
      and `CMap` consume the formal specs; re-recording `golden/` under a
      pinned torch. File as issues before deleting this file.
