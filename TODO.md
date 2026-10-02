# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

🌙 took [#784](https://github.com/discopy/discopy/issues/784), the candidate the board ranked
for tonight: `pickle.loads(pickle.dumps(x))` drops the subscript of a `NamedGeneric`, so a term
comes back a member of the bare origin and compares **unequal to itself** — on `Tensor[float]`,
`Matrix[float]`, `List[int]`, `List[type]` and `Hypergraph`, whose `__eq__` then raises.

🐦 measured the repair the issue describes — a `__setstate__` on the `Result` body — and it breaks
three committed 0.6-era back-compat cells. So the first point is to find where the edge actually
goes, not to write the hook the issue asked for.

- [ ] Measure why the `__setstate__` hook breaks the 0.6 cells, rather than work around it
- [ ] Restore the subscript on the reconstruction path, so no class gains a `__setstate__`
- [ ] All six carriers round-trip and compare equal, `Hypergraph` included
- [ ] The three 0.6/1.2 back-compat cells stay green, and say what pins them
- [ ] State the law as an ordinary test, since the matrix enrols none of these carriers
- [ ] `CHANGELOG.md` entry under `[Unreleased]`
- [ ] `pflake8 discopy`, `pylint discopy`, `coverage run -m pytest`, and the property matrix
- [ ] Report the bugs met on the way rather than fold them in
