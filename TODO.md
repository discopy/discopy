# Alpha-equivalence of lambda terms

> let's open an alpha equivalence PR with a new method alpha_eq that takes an arbitrary number of terms with a helper method that takes a substitution for each so the whole check is linear time, use the property testing scheme to add axioms for these terms e.g. alpha equivalence is a congruence wrt application

— toumix, 2026-09-17, live.

- [ ] `biclosed.TermBase.alpha_eq(*others)` and its helper `alpha_eq_under(substitutions, *others, depth)`, one substitution of the bound variables per term, extended and restored in place so the whole check is one pass over the terms; a case per term class, inherited by `closed`
- [ ] `TermBase.strategy` generating well-typed linear planar terms (any closed term is one too), with `Renamed[C]` generating the same term under several namings of its bound variables
- [ ] the laws as axioms of `TermBase`: reflexivity, renaming, symmetry, congruence with respect to application and to abstraction, soundness for evaluation; `serialisation` declared failing for #692
- [ ] `biclosed.TermBase` and `closed.TermBase` enrolled in `proptest/categories.py`, with a dry run in the unit tests
- [ ] unit tests: shadowing, free variables by name, `left` flags, non-linear closed terms, the substitutions read the same after the call
- [ ] changelog entry
- [ ] `pflake8`, the full suite and the `TermBase` cells of the property matrix green
