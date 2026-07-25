Go through the discopy PRs you own and follow the agents/EVENING.md prompt i.e. go through the reviews and implement them

Mathematical design: spider drawing preserves the graph's stable node order;
grouping by shape must preserve first occurrence rather than pass through sets.

- [x] Restore deterministic spider
  rendering and add a regression test.

Verification: `pflake8 discopy` and the focused rich-display test pass. The
full suite is environment-blocked at collection by missing optional quantum,
tensor, JAX, SymPy and NLTK dependencies.

---

get the PR about rich display rendering and make another PR, targeting that
branch, not main, fixing alexis comment and conflicts to main that appear on
that branch

Mathematical design: an image is a function of the diagram alone, so the
regenerated files under `docs/_static` carry no information and belong to CI,
not to the branch. Saving a figure is one operation parameterised by a format,
whether it lands on a path or in a buffer, so `savefig` takes the format rather
than each caller re-deriving it.

- [x] Merge `origin/main` into the branch so it sits on top of the SVG work and
  the docs-generation job.
- [x] Resolve `discopy/drawing/backend.py` by extending main's `savefig` with
  `format` and `metadata` instead of duplicating it inside `Matplotlib.output`.
- [x] Drop the hand-committed `docs/_static` regeneration, the `docs-static` job
  on main commits it back on every pull request.
- [x] Remove the `TODO` comments and the redundant `svg_hashsalt` config entry
  reintroduced by the earlier merge.
