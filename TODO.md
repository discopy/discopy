# TODO

Prompt (verbatim):

> Scan through all my vibe coded PRs (claude or codex) and review them, proposing refactors when you see fit

> add review comments for the next round please
> then refactor 375/376 to follow the guidelines (they didn't exist yet when the prs were opened)

Design call from Alexis: go alpha-canonical, i.e. drop the `varname` side-channel
and name bound variables from their de Bruijn level, rather than keeping the
round-trip faithful on the nose.

---

- [x] Review the 20 open `claude/`/`codex/` PRs and post review comments
- [x] Drop the global `_fresh_names` counter: `fresh_name(level, avoid)` is pure
      and deterministic (STYLE.md: DisCoPy is deterministic)
- [x] Drop `annotate` / `varname`, the attribute smuggled onto copied `Ob`s that
      `__eq__` ignores (STYLE.md: DisCoPy is transparent)
- [x] `Diagram.to_term(*freevars)` takes the free variables instead, recovering
      both their names and the grouping of the wires they stand for
- [x] Expose `_dom_to_variables` / `_split_scan` / `_box_to_term` /
      `_diagram_to_term` as `Diagram.to_term`, `Diagram.decompile`,
      `Diagram.split_scan` and `Box.to_terms` (STYLE.md: DisCoPy has no secrets)
- [x] Replace the `isinstance` chain in `_box_to_term` by overriding
      `Box.to_terms` in `Eval`, `Coeval`, `Curry`, `Sum`, `TermBase`, `Swap` and
      `Copy` (STYLE.md: never repeats itself, never nests)
- [x] Replace `getattr(type(diagram), "braid_factory", None) is not None`, thrice
      inlined, by `Diagram.is_braided` and `Diagram.term_left`
- [x] Split `TermBase.to_map`'s nested `go` into `to_map_and_freevars` on
      `Variable`, `Application` and `Abstraction`
- [x] Split `CMap.to_term`'s eight closures into `TermReader`, `Unifier` and the
      `Var` / `App` / `Abs` skeleton
- [x] Move the inline error strings to `messages.py`
- [x] Fix `closed.Application.__check_dom__`: `list(set(...))` made the order of
      the free variables depend on `PYTHONHASHSEED`
- [x] `uv run pflake8 discopy` and `uv run coverage run -m pytest`
