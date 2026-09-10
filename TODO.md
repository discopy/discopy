# TODO

Human prompt, verbatim:

> fix this issue: https://github.com/discopy/discopy/issues/694

- [ ] read #694 and locate the eager names on `main`
- [ ] let `cat.Ob` / `cat.Box` take `name=None` for a lazily computed name
- [ ] make `biclosed.Exp.name` lazy
- [ ] make `biclosed.Curry.name` lazy
- [ ] make `biclosed.Application.name` and `biclosed.Abstraction.name` lazy
- [ ] add a test that the names are absent until read
- [ ] changelog entry
- [ ] report the `monoidal.Bubble` bug that drops `data` and `is_dagger`
- [ ] `pflake8 discopy` and `coverage run -m pytest`
