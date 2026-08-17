# TODO

> Check out my comments in https://github.com/rel-int/optyx/pull/16 We need to simplify this
> implementation.

Simplifying rel-int/optyx#16 onto DisCoPy's own tensor contraction surfaced
two bugs here, reported as issues and fixed on this branch.

- [x] Fix [#581](https://github.com/discopy/discopy/issues/581): rename the
      hatching loop variable in `draw_discard` so it no longer shadows the
      layer index, with a regression test drawing a three-wire `Discard`.
- [x] Fix [#582](https://github.com/discopy/discopy/issues/582): return the
      array of `Tensor.spider_factory` on the active backend, with a
      regression test evaluating spiders under the PyTorch backend.
- [x] Changelog entries for both fixes.
- [ ] One green CI run.
