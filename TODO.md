# TODO

Prompt ([#490](https://github.com/discopy/discopy/issues/490#issuecomment-5109305444), verbatim):

> ok great let's fix this @toumix-agents please open a PR

---

- [x] `closed.Discard`, a subclass mirroring `markov.Discard`, replaces the
  `lambda X: Copy(X, 0)` that `markov.Copy.__new__` cannot dispatch on
- [x] Delete the second, identical `markov.Copy.__new__`
- [x] Regression test: `Diagram.discard` in a closed diagram, and its repr
- [x] Run `pflake8 discopy` and `pytest`
