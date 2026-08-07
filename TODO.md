# Add Matrix.permutation (#535)

USER on [#535](https://github.com/discopy/discopy/issues/535#issuecomment-5207139512), verbatim:

> omg why do LLMs make PR titles so long 😅
> another issue I just spotted: Matrix is a subclass of MonoidalCategory when it should be at least MarkovCategory until we have a BiproductCategory class
> just changing that subclassing will solve the issue but we should also have a native Matrix.permutation method which would be much much faster than decomposition in terms of binary swaps

- [ ] make `Matrix` a `MarkovCategory` instead of a `MonoidalCategory`
- [ ] add a native `Matrix.permutation` building the matrix in one pass
- [ ] check the native method agrees with the inherited swap decomposition
- [ ] `pflake8 discopy` and `coverage run -m pytest`
- [ ] `CHANGELOG.md` entry
