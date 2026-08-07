# TODO

> Hypergraph.to_diagram https://github.com/discopy/discopy/issues/539
>
> Hypergraph.to_diagram should use permutation instead of swap
>
> Now that we have symmetric.Layer we should make Hypergraph.to_diagram and
> CMap.to_diagram output Permutation boxes

- [WIP] @session_01Ek64A3Z8YhE3j52EhZDqWs-2026-08-07 09:58 Make `Hypergraph.to_diagram` route wires with a single permutation per
      episode, emitted with `Diagram.from_permutation` so that categories with
      a native `Permutation` factory output permutation boxes and the others
      keep their swap decomposition.
- [WIP] @session_01Ek64A3Z8YhE3j52EhZDqWs-2026-08-07 09:58 Same for `CMap.to_diagram`.
- [WIP] @session_01Ek64A3Z8YhE3j52EhZDqWs-2026-08-07 09:58 Update docs and doctests, add tests for the new outputs.
- [WIP] @session_01Ek64A3Z8YhE3j52EhZDqWs-2026-08-07 09:58 Add a `CHANGELOG.md` entry.
- [WIP] @session_01Ek64A3Z8YhE3j52EhZDqWs-2026-08-07 09:58 `uv run pflake8 discopy` and `uv run coverage run -m pytest` green.
