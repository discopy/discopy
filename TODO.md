Refactor this neural net PR after Claude did a first version and I gave feedback then try to port this CatGPT benchmark to DisCoPy https://github.com/discopy/discopy/pull/399

Mathematical design: a neural box is a bidirectional process on the direct
sum of its boundary port spaces, optionally paired with a private memory
space. One synchronous execution round first routes boundary messages along
the combinatorial map's edge involution, then applies every box independently;
private memory is threaded between rounds but is not part of the categorical
wiring. Backend-specific tensor and module operations should be isolated from
this geometry-of-interaction execution.

- [WIP] @codex-pr399-2026-07-24 13:00 Refactor neural execution around an explicit backend boundary and make the geometry-of-interaction steps legible.
- [ ] Add optional per-network memory without representing private state as public wiring.
- [ ] Port the CatGPT benchmark to DisCoPy.
- [ ] Add concise tests and documentation, then run lint and the full test suite.
