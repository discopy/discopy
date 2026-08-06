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

- [x] Extract the operand loop of `Functor.contract` as a public `Functor.operands`
- [x] Add a `contract` parameter to `tensor.Functor` and `Diagram.eval` choosing the
      engine between `einsum` (default, switching to `opt_einsum` past
      `config.MAX_EINSUM_INDICES`) and `quimb`
- [x] Build the quimb network from the operands: `Functor.to_quimb` and `CMap.to_quimb`,
      with `Diagram.to_quimb` delegating through `to_map()`
- [x] Thread `contract` through `quantum.circuit.Circuit.eval` and `Sum.eval`
- [x] Declare the contraction dependencies: a `tensor` extra with `opt_einsum`, `quimb`
      and `cotengra`; fix the README description of quimb
- [x] Tests: engines agree on the `CMap` diagram zoo, `CMap.to_quimb` vs `eval`,
      trailing-swap regression (#297), compressed contraction, autodiff through quimb,
      repr round-trip
- [x] `CHANGELOG.md` entry, `pflake8` and `coverage run -m pytest` green
