# TODO

> addition goes away altogether, only tensor/matmul remains

toumix on https://github.com/discopy/discopy/pull/747#discussion_r3966654374

- [ ] Remove `List.__add__`/`__radd__` and the `__add__` aliases of `stream.Ty` and `interaction.Ty`
- [ ] Every `+` on objects becomes `@`/`tensor`: `python`, `para`, `stream`, `interaction`, `hypergraph`, `abc`
- [ ] `para.Symmetric` checks its objects are `category.ob`, so a raw tuple is refused (#750)
- [ ] Tests pin `TypeError` on `+`, `CHANGELOG.md`, `pflake8`, the full suite, the docs and the notebooks
