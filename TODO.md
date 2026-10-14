# TODO

> we need to take care of https://github.com/discopy/discopy/issues/803 - consider @~/src/python/metatheory-refresh/ IN DEPTH and how holes and pattern variables and the pattern language is done there. we need to make a PR that targets branch ale/table-carrier https://github.com/discopy/discopy/pull/730 - and base on top of it
>
> INVESTIGATE VARIOUS SOLUTIONS, OPTIONS AND THEIR TRADEOFFS AND PROS AND CONS AND ASK QUESTIONS BEFORE YOU MAKE A DECISION.
> READ THE REFERENCES IN FULL, DO NOT SKIP FOR TOKENS
>
> evaluate various options and their tradeoff and ask me questions about it

- [x] `cat.Hole`, `Arrow.holes`, `Arrow.fill`, `Filling` and the `fill_*` axioms, defined once in `discopy/cat.py`
- [x] `monoidal.Hole` drawn as `?name`, `Layer.holes`, `Diagram.fill_tensor`, `Diagram.substitute` inlined and `Match` deleted
- [x] `braided.Diagram.naturality` without `Match`
- [x] A `Hole` per level from `traced` to `frobenius`, mirroring `Box`
- [WIP] @omp-2026-10-14 17:30 `ehypergraph.Symmetric.holes`
- [ ] Tests in `test/cat.py`, `test/monoidal.py` and `test/ehypergraph.py`, changelog entry
