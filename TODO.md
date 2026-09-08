# TODO

> it is not so clear to me what is the serialization infrastructure in discopy.
> have a look at the codebase and map out a plan to extract the various serialization interfaces (pickle, repr, to_tree) to ensure the interface is stable across all implementors

- [x] `utils.Serialisable`, generic `to_tree`/`from_tree` driven by one `tree_keys` hook, with `BinaryBoxConstructor` as first client
- [x] rebase `cat.Ob`/`Arrow`/`Box`/`Sum`/`Bubble` on the hook, byte-identical trees, `DeprecationWarning` on the `Bubble` `'arg'` shim
- [x] `rigid.Box` serialises `z`, fixing the silent loss on rotated boxes
- [x] fix pickling and deepcopy of parameterised `NamedGeneric` classes losing their parameter
- [x] `from_tree` resolves parameterised factory names such as `tensor.Box[float]`
- [ ] file one umbrella issue with every remaining missing piece, property-based round-trip testing included
- [ ] regression tests for each fixed bug, `CHANGELOG.md` entry, `pflake8` and `pytest` green
