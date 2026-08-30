# PR #688 style-review round

- [x] Address DisCoPy bot comment 3889114330: "The four `close`/`*_close`
  functions repeat the same `np.allclose` boilerplate with different
  evaluators. Consider a `make_close(evaluator)` helper."
- [x] Address DisCoPy bot comment 3889114332: "`high_spider_parity` and
  `high_red_parity` are near-identical apart from the box-type filter;
  parametrizing the predicate would remove the duplication."
- [x] Address DisCoPy bot comment 3889114333 by moving the exact-RootOf note
  from an inline comment into `main`'s docstring.
- [x] Run the exact checker, notebook export check, lint, and diff checks.
- [ ] Push, reply to all three comments, and resolve all three threads.
