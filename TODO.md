# TODO

> FreeMonoid to List rename https://github.com/discopy/discopy/pull/747
> This will impact the future of the library so much we want to make sure the code is as clean as possible, review the current proposal and take over the ownership of the PR

- [ ] Review the proposal against `STYLE.md`: one interface for `List` and `Ty`, no duck-typing in `cat`, no tuple hack left in `python`
- [ ] Move the sequence interface of `Ty` (`__iter__`, `__pow__`, `__eq__`, `__hash__`, `__repr__`) up to `List`, so that `Ty` and `Dim` only add what they need on top
- [ ] Revert the `hasattr` special case in `cat.FreeCategory.__getitem__`: a `List` whose atoms carry no colour slices itself
- [ ] `python.Function.ob = List[type]` declared plainly, the import cycle broken once in `python/__init__.py` rather than by a lazy class property and three module `__getattr__`
- [ ] Cast tuples of types into `List[type]` at the boundaries of `python.Function`, the tuple special case of `stream.Ty` and `utils.is_tuple` go with it
- [ ] `CHANGELOG.md`, tests, `pflake8`, `pylint`, the full suite, the docs build and the notebooks
