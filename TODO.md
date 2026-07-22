# TODO

Prompt ([#374](https://github.com/discopy/discopy/issues/374), verbatim):

> There should be `discopy.kleisli` submodule with the following modules:
>
> - `discopy.kleisli.monad` builds upon the existing `Transformation` to define monads as monoids in the category of Python endofunctors, the underlying functor should be of a new class `python.function.EndoFunctor`. This should come with examples for the maybe, powerset and subdistribution monads 
> - `discopy.kleisli.channel` defines the Kleisli category with `Channel[M]` as the `NamedGeneric` class based on a given `M: Monad`
> - `discopy.kleisli.additive` defines Kleisli as a traced cocartesian monoidal category with disjoint union as tensor, the trace is given by the execution formula (i.e. a while loop) and it should come with tests that show this converges whenever the monad is sub-additive e.g. the maybe, powerset and subdistribution monads
> - `discopy.kleisli.multiplicative` defines Kleisli as a premonoidal copy-discard category with tuple as tensor and the monoidal strength given pointwise, this should come with a test that the category is monoidal iff the monad is commutative.
>
> The last two modules should come with their own evaluation methods for the corresponding `Hypergraph` data structure, i.e. probabilistic token passing for `additive` (i.e. a monadic value over token state/positions gets updated until all the traced component goes to zero) and a monadic generalisation of message-passing/belief propagation for `multiplicative` (i.e. at each step every node updates its own state according to that of its neighbours until a fixed-point is reached).

---

- [WIP] @evening-2026-07-22T02:00 `discopy.kleisli.monad`: monads as monoids over a new `python.function.EndoFunctor`, with maybe, powerset and subdistribution examples
- [ ] `discopy.kleisli.channel`: `Channel[M]` as a `NamedGeneric` over `M: Monad`
- [ ] `discopy.kleisli.additive`: traced cocartesian Kleisli with the execution formula as trace; convergence tests for sub-additive monads
- [ ] `discopy.kleisli.multiplicative`: premonoidal copy-discard Kleisli with pointwise strength; test monoidal iff the monad is commutative
- [ ] `Hypergraph` evaluation methods: token passing for `additive`, message passing for `multiplicative` — coordinate with #366 and #363
- [ ] Run `pflake8 discopy` and `coverage run -m pytest`

## Guidance (🐦 birdsong, 2026-07-22)

- last of the six drafts by design (per Alexis's own wave order) — start `monad` +
  `channel` (self-contained) whenever, but hold off on `additive`/`multiplicative`'s
  `Hypergraph` evaluation methods until #366 (additive.Hypergraph) and #363
  (multiplicative.Hypergraph) land — both still draft, both build the base classes
  this module's token-passing/message-passing hooks into.
- `python.function.EndoFunctor` is new — check `python.function`'s existing
  `Function`/`Transformation` factory pattern before adding it, keep the same shape.
- the monoidal-iff-commutative test for `multiplicative` is the one non-obvious
  correctness property here — write it first, as a property-based test if #347
  (property-based testing PR) has landed by the time you start.
