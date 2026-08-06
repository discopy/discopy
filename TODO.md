# TODO

> Check out the PR https://github.com/rel-int/optyx/pull/21 in optyx. We need to find a
> simpler way to do this, definitely I don't want to add a new module contract. Checkout
> also the PR https://github.com/discopy/discopy/issues/523 in discopy. The proposal here
> was to add a contract parameter in eval and tensor.Functor to decide which contractor to
> use between einsum, opt_einsum and quimb/cotengra. With einsum as default switching to
> opt_einsum when there are index problems. it's good to keep the quimb interface to be
> able to use the methods in that library. I would like to have minimal edits in optyx and
> handle all the contraction in DisCoPy, so that optyx remains syntactic (except for the
> PercevalBackend). What is the best solution? Propose a plan for standardising this
> between discopy and optyx

- [WIP] @f1h17k-2026-08-06 12:23 Extract the operand loop of `Functor.contract` as a public `Functor.operands`
- [WIP] @f1h17k-2026-08-06 12:23 Add a `contract` parameter to `tensor.Functor` and `Diagram.eval` choosing the
      engine between `einsum` (default, switching to `opt_einsum` past
      `config.MAX_EINSUM_INDICES`) and `quimb`
- [WIP] @f1h17k-2026-08-06 12:23 Build the quimb network from the operands: `Functor.to_quimb` and `CMap.to_quimb`,
      with `Diagram.to_quimb` delegating through `to_map()`
- [WIP] @f1h17k-2026-08-06 12:23 Thread `contract` through `quantum.circuit.Circuit.eval` and `Sum.eval`
- [WIP] @f1h17k-2026-08-06 12:23 Declare the contraction dependencies: a `tensor` extra with `opt_einsum`, `quimb`
      and `cotengra`; fix the README description of quimb
- [WIP] @f1h17k-2026-08-06 12:23 Tests: engines agree on the `CMap` diagram zoo, `CMap.to_quimb` vs `eval`,
      trailing-swap regression (#297), compressed contraction, autodiff through quimb,
      repr round-trip
- [WIP] @f1h17k-2026-08-06 12:23 `CHANGELOG.md` entry, `pflake8` and `coverage run -m pytest` green
