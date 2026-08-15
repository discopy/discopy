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

- [WIP] @evening-bk8zei-2026-08-15 00:20 "let's make neural into a folder, name this file neural/network.py and
      move the backend to neural/torch.py" — and "goes to
      discopy/neural/backend.py" for the backend interface
- [WIP] @evening-bk8zei-2026-08-15 00:20 "This category doesn't really exist because neural networks aren't
      really a traced category: the fixed points are not guaranteed to be
      reached we only do a fixed number of iterations." — fix the module
      docstring's claim
- [WIP] @evening-bk8zei-2026-08-15 00:20 "nobody says 'combinatorial maps of a category', just 'morphism'"
- [WIP] @evening-bk8zei-2026-08-15 00:20 "this looks like pure boiler plate we shouldn't need it" — the
      factory-wiring block at neural.py:126
- [WIP] @evening-bk8zei-2026-08-15 00:20 "This looks like extra bureaucracy on top of the CMap, not sure we need
      it" — the wrapper at neural.py:376
- [WIP] @evening-bk8zei-2026-08-15 00:20 "It should be clear whether this is a map neural network or just a
      plain feedforward one" — neural.py:237
