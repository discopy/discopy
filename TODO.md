Go through the discopy PRs you own and follow the agents/EVENING.md prompt i.e. go through the reviews and implement them

Mathematical design: spider drawing preserves the graph's stable node order;
grouping by shape must preserve first occurrence rather than pass through sets.

- [x] Restore deterministic spider
  rendering and add a regression test.

- [x] Merge `main` into the branch, resolving the `savefig` conflict.

Mathematical design: saving a figure is a function of the format, which is
determined by the path when it is a file name and has to be given explicitly
when it is an in-memory buffer. There is one such function, `savefig`, shared
by the diagram and hypergraph drawing code.

Verification: `pflake8 discopy` and `pytest` pass, 751 passed with the four
`biclosed` and `cmap` doctest failures reproducing identically on `origin/main`
(the Graphviz `dot` executable is missing from this environment). The two
conflicted hypergraph SVGs regenerate byte-identically to the resolved version.
