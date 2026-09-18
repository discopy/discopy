> you're missing transitivity!
>
> there should be a way of defining these properties for an abstract Equivalence class

— toumix, 2026-09-18 13:38, review comment on `alpha_symmetry` in `discopy/biclosed.py` of #767.

- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-18 13:45 transitivity as a law of alpha-equivalence, a cell of the matrix at both levels, non-vacuous: quantified over a term, a renaming of it and a third term of the same type that may or may not be alpha-equivalent, so that both directions run
- [WIP] @session_01NTmDuo6GWCjsJ4uV7FBmjQ-2026-09-18 13:45 `axioms.Equivalence`, the abstract class of a type with an equivalence relation, stating reflexivity, symmetry and transitivity once for any such relation; `TermBase` inherits them with `AlphaEquation` as the relation and its generator of related terms, `alpha_reflexivity` and `alpha_symmetry` gone; tests, changelog and description follow
