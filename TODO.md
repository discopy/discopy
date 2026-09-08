# TODO

> summarize all the changes suggested by toumix-agents into a plan

Decisions taken by the prompter on the plan: drop `cat.Functor` from
`CARRIERS`; move every stage-2 piece nothing on this head calls to #659;
take the four config cuts; on CI and the API surface follow the
recommendation — PR runs never upload the database, CI's database is an
explicit `shared` profile, the export and docs page stay.

- [ ] `Functor` leaves `CARRIERS`: its strategy, classifiers, `Relabelling`, the `Functor.ob`/`__call__`/`is_tuple` changes go; #648 becomes a regression test in `test/cat.py`, the `functor_factory` pin moves to `test/rigid.py`
- [ ] Stage-2 vocabulary moves to #659: the argument shapes nothing here calls, `Natural`, `CMap.is_boundary_connected`, `proptest/test_drawing.py`, `proptest/test_normal_form.py`, the toy carriers of `test/axioms.py`
- [ ] Config cuts: the conftest database-key hook, `--import-mode=importlib`, the `--axioms` flag; `Axiom` checks its module defers annotations
- [ ] CI: only non-PR runs upload the database; CI's database is the `shared` profile, registered on demand; docs and changelog follow
- [ ] Revert the unrelated `permutation` rewrite, run the checks, refresh the PR body, comment on #658 and #659
