# TODO

> make closed fully linear, no copy at all here.
> then, introduce markov Diagram and Term
> and then another cartesian module defining again Diagram and Term, and instantiates a new abc.CartesianCategory (for now it is just markov but once we have property testing it'd have another axiom to check for determinism)
> change the existing examples of closed diagrams with copies to the appropriate setting

- [x] Make `closed` fully linear: no `Copy`, `Merge`, `Discard`, no borrowed markov supply, no `abc.MarkovCategory`; its terms are the linear symmetric lambda calculus — an application sharing a free variable and an abstraction over an unused or repeated variable raise.
- [x] Introduce `markov.Term`: `Variable`, `Constant` and `Application`, the first-order terms in context of a Markov category, with `Context` moved from `closed` — a shared variable is copied, an unused one is discarded, no abstraction since there are no exponentials.
- [x] Introduce `discopy.cartesian`: `Diagram` and `Term` subclassing `markov`'s, instantiating a new `abc.CartesianCategory(MarkovCategory)` whose determinism axiom — the naturality of copy — waits for the property-testing matrix.
- [x] Move the copy examples of `closed` to their setting: the non-linear lambda terms become first-order markov terms, the copy drawings and discards move to `markov`, `para.Closed` wraps a symmetric base.
- [x] Changelog, `pflake8`, full test suite.
