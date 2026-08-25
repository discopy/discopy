# TODO

> Take this paper, reproduce it with DisCoPy and then find some way to generalise or improve on it https://arxiv.org/html/2608.14872v2

The paper is Stoltz & Vilmart, *Minimality of the Pure Qubit ZX Calculus*
(arXiv:2608.14872): the ruleset `ZXopt = ZXV \ {(Ir)}` is complete — (Ir) is
derivable — and minimal — (B) is broken by a countermodel over the dual
numbers `C[ε]/(ε²)` and (Ig) by a relational countermodel — and likewise for
a second ruleset `ZX'opt`.

- [WIP] @qc5c7o-2026-08-25 21:00 Encode the rulesets ZXopt and ZX'opt of Figures 2 and 3 as `zx.Diagram` equations, including the (EU)/(EU') side conditions, and check their soundness under the standard interpretation
- [ ] Reproduce Theorem 3.2: the dual-number countermodel breaking (B) while satisfying every other rule, with the exact `-√2/4 ε` witness coefficient
- [ ] Reproduce Theorem 4.2: the Boolean relational countermodel breaking (Ig) while satisfying every other rule
- [ ] Verify the derivations of (Ir): every lemma of Appendices A and C is sound, so the two rulesets are complete
- [ ] Generalise: characterise which nilpotent perturbations `O = I + εN` yield a countermodel for (B), and show the paper's three-qubit `N` is the smallest possible one
- [ ] Docs, tests, CHANGELOG entry, `pflake8` and full test suite green
