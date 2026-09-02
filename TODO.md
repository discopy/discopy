# TODO

> cubic, round 1 on #705: **26 issues found** across 23 files. `Recursion.run` retains every autograd
> graph and returns a detached state; `forward` dispatches through torch inside a `backend(...)`
> context and accepts negative `n_rounds`; `MapNN.compile` can serve a stale interaction; cells and
> laws have shape and dtype/device edge cases.

- [ ] `core.py`: dispatch on the ambient backend context; reject negative `n_rounds` and an input on a closed map on the fused path; `_prepare` falls back to buffers
- [ ] `execution.py`: a causal schedule rejects scalar loops
- [ ] `map.py`: the formulas read `X @ P` and `∂f @ P` as the code does; `sites` says what it counts
- [ ] `solver.py`: `Recursion.run` keeps one checkpoint unless `deep` and returns the last one with its graph; `HaltHead` leaves the RNG alone
- [ ] `cells.py`: a zero-width message orbit, a multi-role first orbit, a recurrent cell with the wrong number of states and `emit=False` with a state wider than a leg are rejected at construction
- [ ] `laws.py`: inputs on the module's dtype and device, non-finite residuals are failures, `rounds < 1` is an error
- [ ] `signature.py`, `batch.py`, `rdiff.py`, `jax.py`: empty incidence, diagram members, structural permutations, the docs link
- [ ] answered and declined: `backend` in the package namespace (parity with `discopy.matrix.backend`), `MapNN.compile` invalidation (documented instead), `FixedPoint` keeping graphs under `backward="full"` (by design)
