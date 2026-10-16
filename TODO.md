# TODO

> Help daydream and I merge this property-based testing framework starting with a clean PR to
> refactor the architecture of DisCoPy's modules with the minimum amount of boilerplate, there
> must be a way to get there without a import cycle

- [ ] `cat.Arrow.strategy` searches composite paths, not only a single generator
- [ ] `monoidal`: `Ty`, `Wire`, `Colour`, `Nat`, `Dim`, `Layer` and `Diagram` generate themselves;
      the generators of a free category are its factories, read off `Diagram.generators`, and a
      box class listing its diagram among its bases is that category's `box_factory`
- [ ] every structural box of every level draws itself with its own `strategy`, so that no level
      overrides `Box.strategy`: braids, twists, traces, cups, caps, copies, spiders, evaluations,
      feedbacks, permutations
- [ ] every free category reads back from its `repr`, its pickle and its tree: fix the structural
      boxes that do not, declare the dagger laws inapplicable where there is no dagger
- [ ] enrol the free categories in `proptest/categories.py`; `assert_axioms` in each module's tests
- [ ] `CHANGELOG.md`, `CONTRIBUTING.md`; `pflake8`, `pylint` under the gate, `pytest`, the matrix
