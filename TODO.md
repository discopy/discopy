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

- [x] `discopy.kleisli.monad`: monads as monoids over a new `python.function.EndoFunctor`, with maybe, powerset and subdistribution examples
- [x] `discopy.kleisli.channel`: `Channel[M]` as a `NamedGeneric` over `M: Monad`
- [ ] `discopy.kleisli.additive`: traced cocartesian Kleisli with the execution formula as trace; convergence tests for sub-additive monads — claimed and released by @evening-2026-07-25T02:25, see note below
- [WIP] @evening-2026-07-27T06:34 `discopy.kleisli.multiplicative`: premonoidal copy-discard Kleisli with pointwise strength; test monoidal iff the monad is commutative
- [ ] `Hypergraph` evaluation methods: token passing for `additive`, message passing for `multiplicative` — coordinate with #366 and #363
- [ ] `multiplicative` stress test: compare results against tensor contraction on small enough models (per issue comment)
- [ ] Implement the state monad for seeded randomness; compare empirical distributions against the ones computed explicitly via sub-distribution dicts (value → nonzero weight)
- [ ] `additive` worked example: Dal Lago–Hoshino's token machines (*Geometry of Bayesian Programming*) — the best source found so far for a non-trivial case
- [ ] Write every example as a term in the effectful lambda calculus of #370, not as a diagram built with tensor/composition
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

## Guidance (🐦 birdsong, 2026-07-23)

- Folded in guidance from two issue comments on #374 (2026-07-07, `toumix`) that
  predate this checklist's drafting and were never incorporated: the
  tensor-contraction stress test + state-monad idea, and the "examples as #370
  terms, not diagrams" requirement — both now their own points above.

## CI fix (🌙 evening, 2026-07-25)

`test (3.14)` was genuinely red on this head (not stale-base): Python 3.14 unified
`X | Y` onto `typing.Union` (`type(int | None)` is now `typing.Union`, was
`types.UnionType` on 3.12/3.13), which broke `utils.get_origin`/`assert_isinstance`
for any `Function` with a union-typed `dom`/`cod` — exactly what `EndoFunctor`'s
`Maybe` monad constructs. Fixed in
[038cacd](https://github.com/discopy/discopy/commit/038cacd): `get_origin` leaves
union types untouched (so `isinstance` handles them natively) and `factory_name`
falls back to `str()` for non-`type` objects (the same union also crashed the
rejection-message path on 3.12/3.13 with `AttributeError`, latent but unreached
before). Verified on 3.12/3.13/3.14 directly; full suite diffed before/after
(75→70 failures, all 5 fixed were exactly the reported ones, remaining 70 are
pre-existing `ModuleNotFoundError`s for the unavailable quantum/tensor optional
deps in this sandbox, identical set before and after).

## Design gap on `additive` (🌙 evening, 2026-07-25) — releasing the claim

Claimed this point to start it (CI was the blocker Birdsong flagged), but backed
out before writing code: the guidance above pins the *shape* (disjoint union
tensor, execution-formula trace) but not how a monadic bind interacts with the
trace's tag-routing loop, and that's a real design choice, not a mechanical
translation of `python.additive.Function` (whose `trace` is a plain Python
`while` because it has no effect to bind). Concretely: `python.additive.Function`
tracks a single `(obj, tag)` pair and loops while `tag` says "still in the traced
part". Once the channel is monadic (`X -> M(Y)`), each step yields an `M`-valued
tagged result — for `Powerset`/`Subdistribution` that's a *set* of `(obj, tag)`
outcomes, some exiting the loop and some not in the same step. The generic
`(unit, mult, functor)` interface `Monad` currently exposes has no way to
partition or aggregate an `M(X)` value by tag, so a trace that's generic over
any monad needs either: (a) extra structure on `Monad` (e.g. a fold/partition
op), or (b) a bound on how far to unroll before giving up, with "sub-additive"
cashed out as a concrete numeric condition on that unrolling. Either is a design
call, not an implementation detail — exactly the kind of thing AGENTS.md asks to
pin down in mathematical terms with Alexis before coding (this is normally
Daylight's job). Flagging for the bridge chat / next Birdsong plan rather than
guessing at the semantics; happy to implement once the trace's exact recursion
is spelled out.

## Review follow-up (🌙 evening, 2026-07-25)

Alexis's live directive, verbatim:

> Go through the discopy PRs you own and follow the agents/EVENING.md prompt i.e. go through the reviews and implement them

- [x] @evening-2026-07-25T09:00Z-2026-07-25 09:05Z Address the review on
  [PR #443](https://github.com/discopy/discopy/pull/443): fix the `Iterable`
  annotation, align the subdistribution contract and implementation, expose
  monad APIs in autosummary, prefer `Monad.from_maps`, and file the pre-existing
  callable-repr problem separately if needed
  ([#474](https://github.com/discopy/discopy/issues/474)).
