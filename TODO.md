# TODO

Review round from the session driving #659, reviewing this branch from
the stage above (`split/1-axiom-infra` merged there and green), quoted:

> `Strategy.strategy` declares `(cls, **params)`, but nothing holds an
> implementation to it and `Atomic.strategy` below already narrows it
> away. #659 pays for that three times in `Wire.strategy` [...] Please
> say here what an override owes: accept `**params` and forward what it
> does not consume. A base that takes `**params` and pops its own
> bounds also lets a subclass override them by passing them [...]
> [update] `frobenius.Wire.strategy`'s `.__func__` is fixed on #659 in
> 43f666c [...] The remark here still stands on its own terms.

> `Atomic.strategy` is the only one of the four subspace wrappers that
> takes no `**params`, and it and `NonEmpty` reach for
> `factory.strategy(...)` where `Small` and `BoundaryConnected` go
> through `resolve`. Between them those two differences mean only
> `Small` composes [...] Route all four through `resolve` and give all
> four `**params`, so a subspace annotation nests however it reads.

> `falsify` (l. 725) and `__call__` (l. 754) open with the same two
> lines, and `strategy` — which needs the carrier just as much, for
> `self.carrier.dom` and its `.ob`/`.ar` — has neither [...] Moving the
> guard here covers all three and deletes the duplicate.

Plus the maintainer's direction in session: reuse `NamedGeneric` in
`discopy.testing` for the wrapper parametrisation.

- [ ] F3: unbound-carrier guard moves into `Axiom.strategy`;
      `falsify`'s copy deleted; `__call__` keeps its own; the three
      unbound paths pinned in `test_axiom_binding`.
- [ ] F1: `Strategy.strategy` docstring states the override contract
      (delegating strategies accept `**params`, pop what they consume,
      forward the rest; terminal strategies reject unknown params
      loudly).
- [ ] F2: every factory-parameterised wrapper becomes a
      `NamedGeneric["factory"]` mixin with `strategy(cls, **params)`
      reading `cls.factory`; `resolve` slims to a check plus
      `annotation.strategy(**params)`; `substitute` re-substitutes
      through `__origin__`; call sites and tests updated; composition
      pinned end-to-end.
- [ ] CHANGELOG entry; replies on the three threads; resolve them.
- [ ] Delete this file.
