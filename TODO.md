# TODO

USER's review on #701, 2026-09-07, followed here since `rdiff` is built on its optics:

> the swap is useless here, we can get rid of it by changing the formula for lenses

> even worse here the double swap shouldn't be here from the start

- [WIP] @session_0124rh162JphJzjCUFHzdBtC-2026-09-07 17:35 `rdiff` on the new convention: a reverse rule's forward leg is `A -> B @ M`, `reverse_rule` reads the codomain off the end, `rdiff` discards the primal output on the left of the residual; tests and doctest to match.
