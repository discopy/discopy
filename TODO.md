# TODO

> review this PR locally. analyze the diff with the base commit, merge with main and record conflict
> resolution with possible improvements/integrations to this PR for additional review, and make the
> code style, doctest/unit test style and documentation verbosity match the rest of the codebase.
> strive to minimize diff with main.

- [x] Analyze the diff with the base commit
- [x] Merge upstream/main and record conflict resolution (see the merge commit message)
- [x] Match code, doctest/test style and documentation verbosity to the codebase
- [x] Minimize the diff with main
- [x] Run pflake8 and the test suite

## For additional review

Points surfaced by the merge with main, left to a human to sign off or file:

- [ ] `tensor.py` sets `CMap.dtype` and `CMap.eval` on the shared alias
  `cmap.CMap[Diagram]` instead of subclassing; deliberate since the alias is
  keyed by `tensor.Diagram`, but any other module reaching for
  `cmap.CMap[tensor.Diagram]` sees the mutation.
- [ ] `Hypergraph.to_diagram` and `CMap.to_diagram` each define the same
  nested `swap` helper guarding on `SymmetricCategory`; they live on
  different classes so sharing it needs a common home.
- [ ] Integrations with main already covered: `Permutation` plumbing (#594)
  flows through `CMap.from_diagram` as wiring (`test_eliminate_swaps`),
  `CMap.trace(0)` is the identity matching the vanishing axiom (#588), and
  the `curry(n=0)` early return (#588) is subsumed by
  `abc.RigidCategory.curry` (`test_curry_uncurry`, `test_curry_zero`).
