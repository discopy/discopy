# TODO

Human prompt, verbatim:

> fix this issue: https://github.com/discopy/discopy/issues/694

- [x] read #694 and locate the eager names on `main`
- [x] let `cat.Ob` / `cat.Box` take `name=None` for a lazily computed name
- [x] make `biclosed.Exp.name` lazy
- [x] make `biclosed.Curry.name` lazy
- [x] make `biclosed.Application.name` and `biclosed.Abstraction.name` lazy
- [x] add a test that the names are absent until read
- [x] changelog entry
- [ ] report the `monoidal.Bubble` bug that drops `data` and `is_dagger`
- [x] `pflake8 discopy` and `coverage run -m pytest`
