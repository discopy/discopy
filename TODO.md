# TODO

> This came up while reviewing https://github.com/discopy/discopy/pull/489#discussion_r3896050565
> Search in the codebase for all `tensor` methods that look like `def tensor(self, other: T = None, *others: T) -> T:` or `def tensor(self, other: T) -> T` with no `@unbiased`, or in general any `tensor` implementation that does not conform to the contract defined in `abc.MonoidalCategory.tensor` which should be `def tensor(self, *others: T) -> T`. When it would help performance, avoid the use of `@unbiased` and implement simultaneous `n`-ary tensoring.

## Survey

`grep -n "def tensor" discopy/**/*.py`, classified against the contract:

Non-conforming, `(self, other=None, *others)`:

- [WIP] @session_01Cx6uSFsQAKANhJLV1mQVpM-2026-09-11 08:07 `monoidal.Diagram.tensor`
- [WIP] @session_01Cx6uSFsQAKANhJLV1mQVpM-2026-09-11 08:07 `monoidal.Sum.tensor`
- [ ] `symmetric.Permutation.tensor`
- [ ] `python.finset.Permutation.tensor`
- [WIP] @session_01Cx6uSFsQAKANhJLV1mQVpM-2026-09-11 08:11 `tensor.Tensor.tensor`
- [WIP] @session_01Cx6uSFsQAKANhJLV1mQVpM-2026-09-11 08:11 `matrix.Matrix.tensor`
- [WIP] @session_01Cx6uSFsQAKANhJLV1mQVpM-2026-09-11 08:11 `quantum.channel.Channel.tensor`

Non-conforming, binary with no `@unbiased`:

- [WIP] @session_01Cx6uSFsQAKANhJLV1mQVpM-2026-09-11 08:07 `monoidal.Layer.tensor`
- [ ] `python.finset.Function.tensor`
- [ ] `python.additive.Function.tensor`
- [ ] `python.multiplicative.Function.tensor`

Conforming through `@unbiased`, to be measured for an `n`-ary rewrite:

- [ ] `hypergraph.Hypergraph.tensor`, `cmap.CMap.tensor`, `drawing.Drawing.tensor`
- [ ] `para.Symmetric.tensor`, `stream.Ty.tensor`, `stream.Stream.tensor`,
      `interaction.Diagram.tensor`

Already conforming: `abc.Nat`, `monoidal.FreeMonoid`, `monoidal.Nat`,
`interaction.Ty`, `hopf.Representation`, `quantum.channel.CQ`.

## Wrap-up

- [ ] benchmark the `n`-ary rewrites against the fold
- [ ] tests, `CHANGELOG.md`, `pflake8 discopy`, `coverage run -m pytest`
