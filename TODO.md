# TODO

USER, 2026-09-16 14:04 UTC, a review comment on `Grammar` in `discopy/grammar/abstract.py` of
https://github.com/discopy/discopy/pull/400 at `90252cd9`:

> "all the grammar is in the lexicon", we don't really need the start type (can be the input of a "parse" method we implement in a later stage) and the other attributes could be derived from the lexicon

- [WIP] @session_01JvjjihGD4ybHGeqLXyLqUi-2026-09-16 14:20 `Grammar` goes: a lexicon is the grammar, its vocabulary the atoms and constants it is defined on; the docs, the tests and the changelog say so, the distinguished type being the input of a parser to come.
