Refactor this neural net PR after Claude did a first version and I gave feedback then try to port this CatGPT benchmark to DisCoPy https://github.com/discopy/discopy/pull/399

Mathematical design: a neural box is a bidirectional process on the direct
sum of its boundary port spaces, optionally paired with a private memory
space. One synchronous execution round first routes boundary messages along
the combinatorial map's edge involution, then applies every box independently;
private memory is threaded between rounds but is not part of the categorical
wiring. Backend-specific tensor and module operations should be isolated from
this geometry-of-interaction execution.

- [x] @codex-pr399-2026-07-24 13:00 Refactor neural execution around an explicit backend boundary and make the geometry-of-interaction steps legible.
- [x] @codex-memory-2026-07-24 13:15 Add optional per-network memory without representing private state as public wiring.
- [x] @codex-catgpt-2026-07-24 13:00 Port the CatGPT benchmark to DisCoPy.
- [x] @codex-validation-2026-07-24 14:10 Add concise tests and documentation, then run lint and the full test suite.

## Backend split follow-up

ok split the work in three abc / torch / jax: first refactor the existing torch into abc and torch then add jax on top
you can push directly to this PR for abc+torch, open a fresh one for jax

- [x] Merge current main into the PR branch and resolve conflicts.
- [x] Turn `neural.Backend` into an explicit abstract interface and pass backend-owned modules into execution.
- [x] Adapt and bind the existing PyTorch implementation to the abstract interface.
- [x] Add concise backend contract and PyTorch regression tests, update the documentation, and run lint and tests.
- [x] Remove the accidentally tracked generated execution-plan API stub.

## Review feedback outstanding (toumix, 07-28 and 08-14)

Each point quotes USER's review comment verbatim; the thread links are on
[#399](https://github.com/discopy/discopy/pull/399).

- [x] "let's make neural into a folder, name this file neural/network.py and
      move the backend to neural/torch.py" — and "goes to
      discopy/neural/backend.py" for the backend interface
- [x] "This category doesn't really exist because neural networks aren't
      really a traced category: the fixed points are not guaranteed to be
      reached we only do a fixed number of iterations." — fix the module
      docstring's claim
- [x] "nobody says 'combinatorial maps of a category', just 'morphism'"
- [x] "this looks like pure boiler plate we shouldn't need it" — the
      factory-wiring block at neural.py:126
- [x] "This looks like extra bureaucracy on top of the CMap, not sure we need
      it" — the wrapper at neural.py:376
- [x] "It should be clear whether this is a map neural network or just a
      plain feedforward one" — neural.py:237
- [x] @evening-bk8zei-2026-08-15 01:05 Fix `neural.rdiff` for the layer
      representation of [#438](https://github.com/discopy/discopy/pull/438),
      which the last merge of `main` broke: `to_staircases` no longer yields
      unpackable layers, so read `Layer.boxes_and_types` instead.

## JAX backend follow-up

ok split the work in three abc / torch / jax: first refactor the existing torch into abc and torch then add jax on top
you can push directly to this PR for abc+torch, open a fresh one for jax

- [x] Add a lazy JAX implementation of the neural backend primitives.
- [x] Wrap compiled execution plans as callable JAX PyTrees with explicit runtime modules.
- [x] Cover eager execution, JIT, gradients, sharing, nesting, and private memory.
- [x] Document the JAX module protocol and run lint and tests.
- [x] @evening-2026-08-19T01:20Z Port the JAX backend onto the package layout
      of #399, which landed the abc/torch split this PR's prompt asked for:
      `discopy/neural_jax.py` becomes `discopy/neural/jax.py` with one
      `JAX(Backend)` class in place of a class plus a module of free
      functions, mirroring `discopy/neural/torch.py`; `BACKENDS` registers it
      by qualified name so it stays lazy; the PyTree holds the `CMap` rather
      than the `ExecutionPlan` that #399 removed. The tests are
      `test/neural/jax_backend.py`, since neither name the convention would
      pick is available: pytest puts a test module's own directory on
      `sys.path`, so `test/neural/jax.py` shadows the library it is testing —
      which is why there is no `test/neural/torch.py` either — and
      `test/neural/backend.py` collides on basename with
      `test/drawing/backend.py`, as test directories carry no `__init__.py`. `discopy/neural/jax.py` joins
      `test/plugin.py`'s `UNIMPORTABLE`, so `--skip-extra` skips it.
