# TODO

Human prompt, verbatim (USER, 2026-09-08):

> make a thorough review of the existing discopy.neural work (spawn subagents as needed) consolidate it in one stack of PRs with a clear plan for how to review it

This branch is one level of the stack that consolidates #705: mapnn, the torch module a diagram trains as, and batching over heterogeneous diagrams.

- [ ] cut the level's files from #705's tree onto its base
- [ ] `uv run pflake8 discopy` and the level's tests green locally
- [ ] pull request opened with its review cost and what to check
