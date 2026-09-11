# TODO

> This came up while reviewing https://github.com/discopy/discopy/pull/489#discussion_r3896050565
> Search in the codebase for all `tensor` methods that look like `def tensor(self, other: T = None, *others: T) -> T:` or `def tensor(self, other: T) -> T` with no `@unbiased`, or in general any `tensor` implementation that does not conform to the contract defined in `abc.MonoidalCategory.tensor` which should be `def tensor(self, *others: T) -> T`. When it would help performance, avoid the use of `@unbiased` and implement simultaneous `n`-ary tensoring.

## Survey

`grep -n "def tensor" discopy/**/*.py`, classified against the contract:

Non-conforming, `(self, other=None, *others)`:

- [x] `monoidal.Diagram.tensor`
- [x] `monoidal.Sum.tensor`
- [x] `symmetric.Permutation.tensor`
- [x] `python.finset.Permutation.tensor`
- [x] `tensor.Tensor.tensor`
- [x] `matrix.Matrix.tensor`
- [x] `quantum.channel.Channel.tensor`

Non-conforming, binary with no `@unbiased`:

- [x] `monoidal.Layer.tensor`
- [x] `python.finset.Function.tensor`
- [x] `python.additive.Function.tensor`
- [x] `python.multiplicative.Function.tensor`

Conforming through `@unbiased`, to be measured for an `n`-ary rewrite:

- [x] `hypergraph.Hypergraph.tensor` and `cmap.CMap.tensor` rewritten
      `n`-ary, both quadratic folds before
- [x] `drawing.Drawing.tensor` left as a fold: nothing calls it with more
      than one argument, the pairwise fold being in `Functor.__call__`,
      so an `n`-ary rewrite would be dead code -- filed as #759
- [x] `para.Symmetric.tensor`, `stream.Ty.tensor`, `stream.Stream.tensor`,
      `interaction.Diagram.tensor` -- already conforming through
      `@unbiased`, left as folds: their bodies interleave swaps rather
      than concatenate, so an `n`-ary form is a different construction
      rather than the same one done once

Already conforming: `abc.Nat`, `monoidal.FreeMonoid`, `monoidal.Nat`,
`interaction.Ty`, `hopf.Representation`, `quantum.channel.CQ`.

## Wrap-up

- [x] benchmark the `n`-ary rewrites against the fold
- [x] tests, `CHANGELOG.md`, `pflake8 discopy`, `coverage run -m pytest`
- [x] filed #759 (`Functor.__call__` folds a layer's images pairwise,
      `Drawing.tensor` still a fold) and #760 (`then` has the same
      non-conforming signatures as `tensor` did)
