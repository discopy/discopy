# TODO

> We need to make a breakthrough with this neural lambda term experiment
> https://github.com/discopy/discopy/pull/401

- [WIP] @w0ybcx-2026-08-28 Build the exact two-stack token machine for
  almost-linear maps: delta nodes push/pop a second (exponential) stack,
  Böhm-tree readback verified exact on Church numerals and on `plus`/`times`
  compositions that beta-reduce through the delta boxes.
- [ ] Learn the arithmetic constants (`zero, one, two, three, plus, times,
  square`) as opaque boxes: oracle labelling by GoI compositionality, the
  two-stack watermark collapsing visits to innocent-strategy rules, one
  small MLP per constant trained in JAX.
- [ ] Scorecard: training tables exact, length generalization beyond
  training magnitudes, and the lambda term for `plus` itself read back out
  of the trained weights.
- [ ] Notebook, tests, CHANGELOG; `pflake8` and `pytest` green.
