# TODO

> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

[#660](https://github.com/discopy/discopy/pull/660) is the hinge of the search
stack — `main ← #659 ← #660 ← {#661 → #662 → {#663, #664, #665}, #666, #667,
#668, #669}` — and the one head of the ten that **conflicts with its own base**
rather than merely trailing it: `split/2-monoidal-strategy` moved to
`b446191d` on 2026-10-08 when #659's merge-down was pushed, and
`CHANGELOG.md`, `test/biclosed.py` and `test/closed.py` have collided with it
since. A dirty head has no `refs/pull/660/merge`, so no `pull_request` job can
run on it and sign-off condition 2 is unsatisfiable until the merge lands.

The content it conflicts with is #659's own, i.e. exactly what `main` will
carry once #659 is merged, so this resolution is the same either way: it is
not an arrear that #659's merge re-invalidates.

The pin forbids pushing to #660's branch, so the resolution is authored here
and offered as a pull request into it (desire#46).

- [ ] Resolve `origin/split/2-monoidal-strategy` into #660 — `CHANGELOG.md`,
      `test/biclosed.py`, `test/closed.py`, measured by merging rather than
      from a file-list intersection
- [ ] Keep both sides of each test conflict: this head's carrier list, which
      adds `CMap` resp. `Hypergraph, CMap` to `assert_axioms`, and the test
      functions #659's branch appended at the same spot
- [ ] Grep the whole tree for conflict markers, the suite reading no
      `CHANGELOG.md`
- [ ] Validate the merged tree: `pflake8`, `pylint` against this head's own
      score in the same venv, the unit suite, and the property matrix at CI's
      `pr` profile, the stack's axioms being what #660 adds
- [ ] Offer the resolution as a pull request into
      `split/3-cmap-hypergraph-strategy`, opened as a draft and undrafted by
      hand, `no-todo-on-main.yml` being gated `branches: [main]`
- [ ] Re-read the five heads based here — #661, #666, #667, #668, #669 — and
      the four through `split/4-matrix`, since a push into a base strands
      every head stacked on it (desire#32's proposal 3)
