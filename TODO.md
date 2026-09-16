# TODO

USER, 2026-09-16, in the interactive session, on the Codex push `85185ab2` to
https://github.com/discopy/discopy/pull/400:

> please clean up the Codex mess, private methods are bad enough but private classes now?

- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 00:20 No private class: `_CategorialSource` goes, the retained categorial type and box carry their `source` themselves, transparent through `repr` and `to_tree`.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 00:20 No `isinstance` chains: `CategorialFunctor` reads every rule as a generator with the one case a closed functor has (`Curry`), `StringFunctor` interprets words and rules, `Diagram.from_categorial` retains every box the same way.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 00:20 `Grammar` names its atomic types like the paper's vocabulary instead of collecting them by a functor's side effect.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 00:20 `test/grammar/acg_examples.py` folded into `test/grammar/abstract.py`, one test file per module.
- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 00:20 The changelog entries and docstrings back in the house style, with what the code does rather than summaries.
