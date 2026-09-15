# TODO

> feedback.Diagram should inherit from symmetric, not markov
> drop symmetric_trace, just make trace inherit from feedback (which in turn inherits from symmetric)

- [x] Rename the planar traced module `traced.py` to `planar.py`, re-pointing `balanced`, `pivotal`, `interaction`, `cmap`, tests and docs, keeping the doctest image paths.
- [WIP] @session_01Ux5oPZt3xhLRBUZBAZTQ4f-2026-09-15 12:20 Make the diagram hierarchy follow `abc`: `symmetric.Diagram` inherits from `braided.Diagram`, dropping the trace and twist it inherited through `balanced`; `compact` keeps the identity twist; `markov` loses its `Trace`.
- [ ] Move `feedback.Diagram` down from `markov.Diagram` to `symmetric.Diagram`, keeping `Copy` and `Merge` as generators borrowed from `markov`.
- [ ] Rebuild `traced.py` above `feedback`: `Ty(feedback.Ty)` and `Diagram(feedback.Diagram, TracedCategory)` with the delay trivial and the feedback given by the trace, `Trace` from `planar.Trace`, the symmetric trace axioms and tests moved in, `para.Traced` re-pointed.
- [ ] Changelog, `pflake8`, full test suite.
