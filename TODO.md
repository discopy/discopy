ok let's do 572 then, stacked on #559

[#572](https://github.com/discopy/discopy/issues/572) folds Copara and the
two-sided construction into `discopy.para`, USER's ruling verbatim: "I would
just add it to the existing para module, same way we already folded markov
and comarkov in the same module".

Mathematical design: a coparametric map places its hidden object on the
codomain, `inside : dom -> cod @ copar`; a stateful map carries both,
`inside : dom @ param -> cod @ copar`, i.e. `Para(Copara(C))` with the
symmetry as distributive law — the type of one time step of `Stream`, whose
`then` and `tensor` are verbatim its composition. Parametric and
coparametric maps embed by trivialising one side; the diagonal
`param == copar` is closed under composition, the free category with
feedback of Katis, Sabadini & Walters.

- [WIP] @daylight-j11voy-2026-08-15 12:48 Add `Copara` and `Stateful` to
      `para.py` at the symmetric level, with composition and tensor
      accumulating the hidden objects in forward order, `coreparam` as the
      covariant 2-cell, doctests for the axioms and the embeddings, tests,
      bibliography and `CHANGELOG.md`. The class name `Stateful` is a
      placeholder for USER to rule on.
- [ ] Refactor `neural.rdiff.ReverseRule.then`/`tensor` onto `Copara` —
      lives on #399's branch where `neural` exists, not here; follow-up on
      [#571](https://github.com/discopy/discopy/pull/571) once this merges.
