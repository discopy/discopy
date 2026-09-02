# TODO

> refactor the discopy.neural module to use para and optics making sure we get feature parity

Step two of the plan approved 2026-09-02, [#702](https://github.com/discopy/discopy/issues/702): one `discopy.neural` from #399, #495, #585 and #686, on `discopy.para` and `discopy.optics` (#701).

- [x] import the three stacks verbatim: #585's package (via #686's port to the post-#532 `CMap`), #399's `backend`/`torch`/`rdiff`, #495's `jax`, and their tests
- [x] `core.py`: `Network.mem` and the backend-generic `Execution` of #399 beside #585's fused torch path, one `forward` dispatching between them
- [x] `map.py`: `ParamMap` and `InteractionMap` as `para.Symmetric` maps over neural diagrams
- [x] `rdiff.py`: reverse rules as `optics.Optic` over neural diagrams, `differentiate` as their functorial fold
- [x] backends registered (`pytorch`, `jax`), `as_network` through `Backend.wrap`, `test/plugin.py`, docs index, changelog
- [ ] feature parity: #399's `test/neural/network.py` and `rdiff.py`, #495's JAX tests, #585's `test_formal`/`test_general` all pass on the merged package (torch on CI, JAX and the torch-free half locally)
- [x] the artefacts stay out: no `docs/optuna`, no golden files, no `.ipynb`, no checkpoints; the examples return as marimo notebooks in step three
