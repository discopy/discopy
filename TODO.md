You are given Vilmart’s 2019 completeness result for the pure qubit ZX-calculus and a working checkout of DisCoPy.

Your task is to investigate whether the axiomatisation can be made minimal: for each axiom, determine whether it is derivable from the others or whether it is genuinely necessary.

Do not assume that the known presentation is minimal. Treat this as an open research problem.

You may:

* read the relevant papers and DisCoPy source;
* formalise the ZX rules as DisCoPy diagrams;
* construct alternative interpretations/models of the calculus;
* use arbitrary finite-dimensional linear algebra, alternative semirings/rings/categories, symbolic computation, numerical experiments, exhaustive searches, differentiation/deformation methods, or other computational techniques;
* write exploratory code and use experiments to generate conjectures.

The central goal is mathematical discovery, not merely implementation.

For every apparent independence result, find a model that satisfies all the other rules but violates the target rule. Computational evidence is useful for discovery, but convert it into a convincing general argument whenever possible.

Be especially alert for structure in any countermodel you discover. If a construction involves apparently arbitrary choices — dimension, basis, parameters, matrices, algebra of scalars, etc. — investigate whether those choices are necessary, whether the construction generalises, and whether there is a smallest or essentially unique example.

Do not search for later work that solves this problem and do not use papers published after Vilmart 2019. I want an independent attempt from the information available at that time.

Keep a research log explaining:

1. the approaches you tried and why;
2. failed approaches and what they taught you;
3. computational experiments and the conjectures they suggested;
4. the eventual proof or countermodels;
5. any stronger theorem or classification you discover along the way.

Use DisCoPy to make the final claims executable/checkable wherever possible.

- [x] Recover the exact 2019 axiom schemas and fix the pre-2020 source boundary.
- [x] Formalise every rule as typed DisCoPy diagrams and executable equation checks.
- [x] Search systematically for derivations and countermodels, recording failed approaches.
- [x] Turn each computational witness into a general proof and classify minimal examples.
- [x] Determine the derivability or independence status of every axiom.
- [x] Write a self-contained research log with executable reproduction instructions.
- [x] Run focused checks, the full test suite, and lint; record any unrelated defects.
