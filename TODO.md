# TODO

USER's review on #701, 2026-09-07:

> the swap is useless here, we can get rid of it by changing the formula for lenses

(`docs/_static/optics/lens.svg`)

> even worse here the double swap shouldn't be here from the start

(`docs/_static/optics/tensor.svg`)

- [WIP] @session_0124rh162JphJzjCUFHzdBtC-2026-09-07 17:15 The residual on the right of the forward leg, `dom.positive -> cod.positive @ residual`, the backward leg unchanged: `to_int` is planar and `Lens.to_optic` is `copy >> get @ id`, no swap.
- [WIP] @session_0124rh162JphJzjCUFHzdBtC-2026-09-07 17:15 `tensor` with one swap per leg and none undone by the next; `then` with the swap on the forward leg, as `para` composes coparameters.
- [WIP] @session_0124rh162JphJzjCUFHzdBtC-2026-09-07 17:15 Tests and the four `docs/_static/optics` baselines regenerated; both threads answered; #705's `rdiff` follows on its own branch.
