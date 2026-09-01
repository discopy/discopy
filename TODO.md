# TODO

Review round from a handoff file ("Handoff review — PR #658
'Property-testing infrastructure'", reviewed at `f0e54f0` against `main`
at `ce044a0`, nothing posted to GitHub), quoted findings:

> ### 1.1 `.failing` makes the promised "bug fixed" signal impossible
>
> `Axiom.failing` wraps the equation so it raises `AxiomError`
> *unconditionally* — it raises even on arguments where the law actually
> holds. [PROPTEST.md](PROPTEST.md) promises the opposite mechanic [...]
> for every record bound to a `.failing` axiom [...]
> `test_counterexample` only re-confirms that the wrapper raises. It
> would not catch a typo'd record either, since the arguments are never
> really used.

> ### 1.2 The deviation vocabulary needs a third value
>
> **Suggestion:** add a third classifier — `.free("holds only
> semantically")` or similar — that marks a law a free carrier
> deliberately does not quotient by.

> ### 2.1 `proptest.yml` cancels runs on `main`
>
> `build.yml` was changed in #645 to `cancel-in-progress: ${{
> github.event_name == 'pull_request' }}` [...] The new workflow
> reintroduces exactly that. [...] A cancelled `main` run is lost
> exploration, not just a lost checkmark.

> ### 2.2 Two annotation-resolution mechanisms where one suffices
>
> Running `resolve(substitute(annotation, scope))` over
> ordinarily-evaluated annotations removes the future-import
> requirement, the `eval_str` dance, and the paragraph of docstring
> explaining why they are needed.

> - **Combinator field propagation is asymmetric.** `modulo` preserves
>   `subspaces` and `broken`; `failing` drops `subspaces`; `weaken`
>   preserves both (and has a redundant `result = ...; return result`
>   at `testing.py:663-666`). [...] A single `replace(**changes)`-style
>   helper would make every combinator one line and uniform.
> - **The atomic-object idiom is repeated five times.**
> - **Silent axiom shadowing.**
> - **`Equation`'s `NamedGeneric["ar"]` parameter is decorative.**
> - **`Functor.strategy(dom=, cod=)`** [...] post-`filter` [...]
> - **Forward-looking configuration in `proptest/`.**
> - **Dagger laws sit on `Category`, which has no dagger.** [...] I'd
>   call it acceptable as-is [...]
> - **PR description drift.** The **How** section says `Strategy` is a
>   `Protocol` [...] In the code it is an `ABC` [...] and
>   `free_strategy` does not exist until stage 2. Also, the body says
>   the `NamedGeneric.__setstate__` fix lands in `split/4-tensor`; it
>   actually lands in `split/3-cmap-hypergraph-strategy`'s `abc.py`.

The work, as decided with the maintainer (keep the current API, KISS
over DRY, focus on the replay promise and verified correctness issues):

- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-01 12:40 1.1: `failing` raises a new `AxiomFailure(AxiomError)` naming its
      equation; `test_counterexample` reads the equation off the
      failure so a record earns its xfail and flips to XPASS when the
      bug is fixed; PROPTEST.md reworded to match.
- [WIP] @session_01Bwih1mV32usVtEFyNbDhq8-2026-09-01 12:40 Field drops: `Axiom` becomes a `@dataclass(eq=False)` and every
      combinator goes through `dataclasses.replace`, fixing `failing`
      dropping `subspaces` (and `modulo`/`weaken` dropping fields) and
      the redundant temporary in `weaken`; unit tests for the
      preserved subspaces and for `inapplicable`.
- [ ] 2.1: proptest.yml concurrency mirrors build.yml.
- [ ] 2.2: declined — PEP 695 type parameters live in a scope `eval`
      cannot see (verified `NameError`), so the `locals` rebinding is
      the only working mechanism; pin the reasoning in the `C0`/`C1`
      docstring.
- [ ] CHANGELOG entry; PR body drift (ABC not Protocol, no
      `free_strategy`, split/3 not split/4) and one round-summary
      comment on the PR, answering the declined points (1.2 declined:
      `.inapplicable` covers permanent deviations; atomic idiom kept
      repeated by choice; shadowing, NamedGeneric, Functor.strategy,
      forward-looking config, dagger laws: previously triaged or
      accepted as-is).
- [ ] Delete this file.
