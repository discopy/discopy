# TODO — Phase 2: rebuild `discopy.neural` around `MapNN`

## The prompt, verbatim

> Refactor only discopy/neural and docs/neural. First read both carefully and understand the current implementation, including Signature, Skeleton, Interpretation, CMaps/Geometry of Interaction, dynamics, engines, ACT, recursive execution, batching, and cells.
>
> The goal is to simplify the library around the user’s actual workflow:
>
> dataset of DisCoPy diagrams
>     -> MapNN
>     -> batching
>     -> solver
>     -> ordinary PyTorch training
>
> Preserve the existing capabilities and mathematical ideas, but hide most internal complexity behind a small, coherent public API.
>
> MapNN should be the central abstraction and behave as a normal torch.nn.Module. A dataset should naturally consist of (diagram, inputs, target) samples, potentially with different diagram structures but shared generators and therefore shared learned parameters.
>
> Internally, preserve the categorical interpretation:
>
> F_theta : C -> Para
>
> with diagrams compiling through CMap/Geometry of Interaction to a global transition
>
> T_{D,theta} : S_D -> S_D.
>
> This formal machinery should support the implementation without becoming mandatory user-facing API.
>
> Aim for a compact organization roughly like:
>
> neural/
>     model.py
>     map.py
>     solver.py
>     batch.py
>     cells.py
>     laws.py
>
>
> Keep additional files such as signature.py only where they provide a genuinely useful abstraction. Prefer fewer coherent concepts over a large formal hierarchy.
>
> model.py should center on MapNN; map.py should contain the formal parametric/interaction semantics and compilation machinery; solver.py should unify execution strategies such as finite iteration, fixed-point solving, segmented/TRM execution, and ACT; batch.py should handle heterogeneous diagram batching; cells.py should provide concrete neural interpretations; laws.py should handle structural/equivariance validation.
>
> Be mathematically careful about the distinction between categorical trace, feedback, finite iteration, recurrent dynamics, and fixed-point convergence. Likewise, do not claim categorical composition for InteractionMap if composition actually requires wiring and interaction semantics.
>
> Reconsider whether Skeleton and Signature need to be prominent public concepts. Since DisCoPy already has diagrams, the preferred user story is:
>
> DisCoPy diagram + MapNN interpretation
>
> rather than a long pipeline of intermediate abstractions.
>
> Keep training itself outside the library: MapNN should integrate naturally with ordinary PyTorch optimizers, losses, dataloaders, etc.
>
> Preserve backward compatibility where sensible, especially existing experiments, but do not retain conceptually incorrect abstractions just for compatibility.
>
> Update tests and docs/neural around the simplified public workflow. Documentation should explain usage first and the categorical/GoI semantics second.
>
> The target philosophy is:
>
> “discopy.neural trains neural interpretations of DisCoPy diagrams. MapNN compiles diagram structure and shared learnable generator maps into a global interaction, while a solver specifies how that interaction is executed.”
>
> Use this as architectural guidance rather than a rigid specification. Inspect the existing code, decide the cleanest minimal refactor consistent with these principles, and then implement it.
>
>
> Also, have in docs:
>
> docs/neural/
>
>     examples/
>
>         sudoku/
>
>             README.md
>
>             dataset.py
>
>             model.py
>
>             train.py
>
>             evaluate.py
>
>             config.py
>
> Inside the sudoku folder (inside examples), is where the code to reproduce the actual experiments lives. Here, is the code needed to reproduce the results we have performed so far. Keep everything clean, nice, simple, and mathematically rigorous.

## The work

- [x] Read every file under `discopy/neural`, `docs/neural` and every test
      that depends on them; trace the forward, routing, schedule, refresh,
      detach, ACT and batching paths against the code rather than the names.
- [x] `discopy/neural/map.py`: the interpretation and the compiled
      `Interaction`, together with `ParamMap` and `InteractionMap`.
      `InteractionMap` refuses `>>`: gluing two interactions is wiring plus
      iteration, not substitution. Absorbs `functor.py`, `parametric.py`
      and `dynamics.py`.
- [x] `discopy/neural/model.py`: `MapNN(ob, ar, solver)`, a
      `torch.nn.Module` that compiles a diagram (cached), builds an initial
      state from named port families, runs the solver and reads the result.
- [x] `discopy/neural/solver.py`: `Iterate`, `FixedPoint`, `Refresh`,
      `Recursion`, `HaltHead`, `ACT` — execution *and* differentiation
      policy, no training loop. Absorbs `engine.py`.
- [x] `discopy/neural/batch.py`: `Batch` over diagrams rather than over
      skeletons and an interpretation; per-member site counts and flat
      widths read off the source maps.
- [x] `discopy/neural/signature.py`: keep `Sym`/`Orbit`/`Signature` and the
      two wiring builders, which now return a source `CMap`; drop
      `Skeleton`, which was a `CMap` plus signatures nothing downstream
      needed. `check_equivariant` and `fusion_residual` move to `laws.py`.
- [x] `discopy/neural/core.py`: unchanged category, plus a module-level
      `box_ports` so that source and image are read the same way.
- [x] `docs/neural/examples/sudoku/{README,config,dataset,model,train,
      evaluate}.py`: the whole study in six files, replacing `core/`,
      `sudoku/`, `migration.py` and `eval_noise_trm_act.py`.
- [x] Equivalence gate: `test_equivalence.py` green bitwise against
      `golden/` — structure, parameters, forward, backward and 20-step loss
      trajectories, float32 and float64, all four models.
- [x] Rewrite `test_formal.py`, `test_general.py`, `test_sudoku_smoke.py`,
      `test_noise_eval.py` and `test_sudoku_act_e2e.py` around the new
      public API; add coverage for `FixedPoint` and for batching through
      `MapNN`.
- [x] Rewrite `docs/neural/README.md` (usage first, semantics second),
      `ARCHITECTURE.md` and `NOTES.md`.
- [x] `pflake8 discopy` and the full `pytest` run.
- [ ] Out of scope, to file as issues: `docs/optuna/*.py` and
      `docs/notebooks/neural-cells-lecture.ipynb` import the removed
      `sudoku`/`core` packages. The optuna scripts were already stale before
      this refactor (they call `zoo.RRNSolver` / `zoo.TRMSolver`, removed
      earlier), so they need a pass of their own.
- [ ] Out of scope, to file as an issue: the noise study's matplotlib
      figures. `evaluate.py --noise` reproduces the grid and its artifacts;
      the plotting that turned them into `figures/` was not ported.
