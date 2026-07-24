# TODO — Fix issue #468

User prompt, verbatim:

> Fix this issue

Context: <https://github.com/discopy/discopy/issues/468>

## Checklist

- [x] @codex-2026-07-24 19:29 Make `hypergraph.draw` reproducible by ensuring a fixed layout seed
      determines all random coordinates without mutating global random state;
      add a concise regression test and run focused and full verification.

## Mathematical description

A hypergraph drawing is a map from its finite set of nodes to points in the
plane. Given a seed, this map should be a deterministic function of the
hypergraph and layout parameters, independent of any ambient random state.
