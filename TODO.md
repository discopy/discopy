# TODO

> cut the `then` changes starting from main instead of stacking and open a new PR, again with the proptest label. again the main intent is to homogenise signatures, optimising is only a side quest.

`abc.Category.then` declares `(self, *others)`, "sequential composition of
`n >= 1` morphisms". Python does not check an override's signature, so nine
implementors narrow it to `(self, other)` or widen it to
`(self, other=None, *others)` and still satisfy the abstract method. The
survey of what each one actually does at `n = 0, 2, 3`:

| class | signature | n=0 | n=2 | n=3 |
| --- | --- | --- | --- | --- |
| `cat.Arrow` | `(self, *others)` | self | ok | ok |
| `monoidal.Diagram` | inherited | self | ok | ok |
| `matrix.Matrix` | `@unbiased` | self | ok | ok |
| `hypergraph.Hypergraph` | `@unbiased` | self | ok | ok |
| `tensor.Tensor` | `(self, other=None, *others)` | `TypeError` | ok | `ValueError` |
| `quantum.channel.Channel` | `(self, other=None, *others)` | `TypeError` | ok | `ValueError` |
| `cat.Functor` | `(self, other)` | `TypeError` | ok | `TypeError` |
| `cat.Transformation` | `(self, other)` | `TypeError` | ok | `TypeError` |
| `monoidal.Functor` | `(self, other)` | `TypeError` | ok | `TypeError` |
| `python.function.Function` | `(self, other)` | `TypeError` | ok | `TypeError` |
| `python.finset.Function` | `(self, other)` | `TypeError` | ok | `TypeError` |
| `python.finset.Permutation` | `(self, other)` | `TypeError` | ok | `TypeError` |

- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 12:06 `utils.unbiased` advertises `(self, *others)` to `inspect.signature`
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 12:06 `abc.Category` teaches the contract: the docstring example and the
      `Parameters:` of the abstract method both say `other`
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 12:06 `cat.Functor`, `cat.Transformation` and `monoidal.Functor` compose `n`
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 12:06 `python.function.Function` and `python.finset` compose `n`
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 12:06 `tensor.Tensor` and `quantum.channel.Channel` drop the `other=None`
      hack, which sends `n >= 3` to the wrong composition
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 12:06 One test of the contract at `n = 0, 1, 2, 3` over every implementor
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 12:06 `CHANGELOG.md`, `pflake8`, the test suite and the property matrix
