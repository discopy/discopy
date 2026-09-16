# TODO

USER, 2026-09-16 11:22 UTC, three review comments on
https://github.com/discopy/discopy/pull/400 at `4668990f`:

On `closed.TermBase.then`:

> this cannot be called then because it clashes with TermBase being a subclass of Diagram
>
> let's call it "compose"

On the changelog entry for `grammar.abstract`:

> remove this slop keep only the important part in the docs of the abstract module itself, here it should only say "added categorial grammar module"

On the docstring of `closed.TermBase`, "since a closed category is markov":

> a closed category isn't necessarily markov, this is a design choice of discopy to implement closed markov categories

- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 11:50 `closed.TermBase.then` renamed `compose`, so that `>>` on terms is the composition of diagrams again; every string written with `>>` in the docs, the tests and the changelog follows.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 11:50 The changelog entry for `grammar.abstract` says that the module was added and nothing else; what it explained is in the module's docs.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 11:50 The docstrings say that DisCoPy's closed categories are markov by design, not that a closed category is.
