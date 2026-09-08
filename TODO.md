# TODO

> open a draft PR for this branch
> by the way please rename the branch to refactor/serde
> also, make Serialisable the only ser/de interface for all three ways of serializing

- [x] rename the branch to `refactor/serde`
- [ ] open a draft pull request for the branch
- [ ] `Serialisable.__repr__`, generic from `tree_keys` and the class defaults, shared with `to_tree`
- [ ] delete the hand-written reprs it reproduces: `cat.Ob`, `BinaryBoxConstructor`, `rigid.Box`, all but the dagger case of `cat.Box`
- [ ] `Serialisable.__setstate__`, the terminal of every pickle migration chain, with `cat.Ob` and `cat.Arrow` chaining into it
- [ ] extend the `CHANGELOG.md` entry, `pflake8` and `pytest` green
