# TODO

Merging `split/4-tensor` down — 38 commits, textually clean — leaves this leaf red four ways.
Each cell reproduced at the `pr` profile with `--hypothesis-seed=0`:

- [WIP] @session_01SJ5ZXJKxbX5AAjRxLg5ABY-2026-09-07 02:30 `serialisation` — `from_tree` asks `discopy.hopf` for an attribute named
      `Representation[Double(4)]`. A class subscripted by an *algebra instance* has no importable
      factory name, which is this leaf's own CHANGELOG prediction arriving as a cell. Declare the
      law inapplicable on the carrier, the way `matrix` and `hypergraph` already do.
- [WIP] @session_01SJ5ZXJKxbX5AAjRxLg5ABY-2026-09-07 02:30 `transparency` — `NameError: name 'complex128' is not defined`. A `tensor.Box` subscripted
      by numpy's `complex128` reprs as `tensor.Box[complex128](...)`, and no module in
      `Strategy.environment()` binds that name. Decide whether the environment is short a name or
      the repr is wrong, fix it or file it.
- [WIP] @session_01SJ5ZXJKxbX5AAjRxLg5ABY-2026-09-07 02:30 `reidemeister_1_cap` and `reidemeister_1_cup` — both recorded counterexamples now **fail
      rather than xfail**. The axiom still raises its `AxiomFailure`, but evaluating
      `failure.equation` dies in `Hypergraph.__init__` with a bare argument-less `ValueError`
      raised from `Hypergraph.rotate`, a wiring arity mismatch. The record can no longer be
      evaluated at all, which is a different failure from the one its `reason` describes.

- [WIP] @session_01SJ5ZXJKxbX5AAjRxLg5ABY-2026-09-07 02:30 `pflake8 discopy` and the suite green before the push.
