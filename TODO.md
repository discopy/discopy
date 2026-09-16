# TODO

> no discard in closed, closed diagrams & terms should be linear not affine. just make it follow the theory exactly. move all the contraction and weakening logic in markov and refine it in cartesian.
> also, give me a list of all examples that failed and that needed to be ported or any kind of adjustment.

- [WIP] @session_01Ux5oPZt3xhLRBUZBAZTQ4f-2026-09-16 11:00 Audit `closed` for any affine residue: no discard, no weakening, terms linear on the nose, with the contraction and weakening logic living in `markov` only.
- [ ] Refine the contraction in `cartesian`: a repeated subterm is evaluated once and its result copied, by naturality of copy, where `markov` evaluates it once per occurrence.
- [ ] Changelog, `pflake8`, full test suite; list every ported example in the session report.
