Go through the discopy PRs you own and follow the agents/EVENING.md prompt i.e. go through the reviews and implement them

Mathematical design: spider drawing preserves the graph's stable node order;
grouping by shape must preserve first occurrence rather than pass through sets.

- [x] Restore deterministic spider
  rendering and add a regression test.

Verification: `pflake8 discopy` and the focused rich-display test pass. The
full suite is environment-blocked at collection by missing optional quantum,
tensor, JAX, SymPy and NLTK dependencies.
