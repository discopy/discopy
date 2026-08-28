# TODO

> We need to make a breakthrough with this neural lambda term experiment
> https://github.com/discopy/discopy/pull/401

- [x] Build the exact two-stack token machine for almost-linear maps:
  delta nodes push/pop a private exponential stack per delta, Böhm-tree
  readback verified exact on Church numerals, `plus`/`times`/`exponent`
  compositions, shared-variable polynomials (89 random cases), `pred` and
  `sub` through a new weakening node ε in `to_map`; the frontier is
  self-application of a shared variable, i.e. Lévy-optimal sharing's oracle.
- [x] Learn the arithmetic constants (`zero, one, two, three, plus, times,
  square`) as opaque boxes: oracle labelling by GoI compositionality, the
  two-stack watermark collapsing visits to innocent-strategy rules (2–16
  per constant), one small MLP per constant trained in JAX.
- [x] Scorecard: 91 training equations, 23 held-out compositions and sums
  and products up to 32 — numerals ten times larger than training — all
  exact, and the lambda term of `plus` itself read back out of the trained
  weights, along with every other constant.
- [x] Notebook (`docs/notebooks/neural-church.ipynb`, executed, 2m08s),
  tests for the ε node, `pflake8 discopy` and the affected pytest files
  green; no `CHANGELOG.md` on this branch's lineage to update.
