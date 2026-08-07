# TODO

> Hypergraph.to_diagram https://github.com/discopy/discopy/issues/539
>
> Hypergraph.to_diagram should use permutation instead of swap
>
> Now that we have symmetric.Layer we should make Hypergraph.to_diagram and
> CMap.to_diagram output Permutation boxes

- [ ] Make `Hypergraph.to_diagram` route wires with a single permutation per
      episode, emitted with `Diagram.from_permutation` so that categories with
      a native `Permutation` factory output permutation boxes and the others
      keep their swap decomposition.
- [ ] Same for `CMap.to_diagram`.
- [ ] Update docs and doctests, add tests for the new outputs.
- [ ] Add a `CHANGELOG.md` entry.
- [ ] `uv run pflake8 discopy` and `uv run coverage run -m pytest` green.
