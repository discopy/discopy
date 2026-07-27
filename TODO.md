# TODO

> Add let statements and not-strictly-associative products to DisCoPy terms https://github.com/discopy/discopy/issues/370
> use the term for the CatGPT benchmark as example https://github.com/discopy/discopy/issues/458

- [x] Add a `Product` type constructor to `discopy.closed` with `Ty.product`, `Ty.__mul__`, `Ty.is_product`, `Ty.factors` and strictification in `Functor`.
- [x] Add `Pack` and `Unpack` boxes for the canonical isomorphism between a product type and the tensor of its factors.
- [x] Add `Tuple`, `Projection` and `Let` terms with `eval` into any closed markov category, plus the `let` introspection helper of issue #370.
- [x] Extend `Substitution` to the new terms and fix the bugs found in it: constants substituted to `None`, the `left` flag dropped on applications, infinite recursion on abstractions.
- [x] Replace the non-deterministic `list(set(...))` in `Application.__check_dom__` with a deterministic deduplication.
- [x] Use a term for the CatGPT transformer block of issue #458 as the documentation example, with a drawing test.
- [x] Fix `closed.Diagram.discard` crashing: `discard_factory` was a lambda while `markov.Copy.__new__` needs a class, replaced by a `closed.Discard`; also removed the duplicated `__new__` in `markov.Copy`.
- [x] Fix drawing of closed diagrams containing `Copy`, `Swap` or `Twist`: the functor chain called `cod.copy`, `cod.swap`, `cod.braid`, `cod.twist` without checking the codomain has them, which `Drawing` does not.
- [x] Make `python.Function.tensor` variadic like `monoidal.Diagram.tensor`, so that `functor.cod.tensor(*terms)` works with any number of terms.
- [x] Run `pflake8` and the full test suite, commit any regenerated documentation images.

> i'm more interested in seeing diagrams printed as terms in a super compact way that reads almost like textbook effectful lambda calculus (same principle that guided the existing Term for lambda calculus)
> also use hypergraphs to simplify stuff

- [x] Print `closed.Constant` as its bare name so terms read like textbook effectful lambda calculus under the obvious variable naming convention of STYLE.md.
- [x] Add `Diagram.to_term` reading a causal diagram as one let statement per box, via `Hypergraph` so that copy, discard and swap simplify away into the spider structure.

> can you avoid producing pack then unpack in the transformer? for now i don't want to introduce rewriting just yet but it should be possible in that case
> open issues with the bugs you reported

- [WIP] @claude-8u19lz-2026-07-27 19:29 Avoid producing `Pack` then `Unpack` when a let statement binds a literal tuple: `TermBase.eval_unpacked` overridden by `Tuple`, no rewriting involved.
- [WIP] @claude-8u19lz-2026-07-27 19:29 Open issues for the bugs reported in the PR description and link them. (The quantum extras cannot be installed in this sandbox — torch download is blocked — so `test/tensor.py`, `test/quantum` and the jax/sympy/graphviz doctests were compared against `main` instead: the failure lists are byte-for-byte identical, everything else passes.)
