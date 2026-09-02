# TODO

> review the discopy.neural workstream and propose a refactor, in particular I would like to include a new module for cartesian lenses and the category of optics in general monoidal categories

> let's go with your proposed plan, then give me a handoff for a fresh session to give a first short at ARC-AGI 1

- [x] `discopy/optics.py`: `Ty` pairs, `Optic` over a symmetric category, `Lens` over a Markov category, `to_int` and `to_lens`
- [x] `test/optics.py`, docs images, `docs/api/semantics.rst`, bibliography, `CHANGELOG.md`
- [x] `pflake8` clean and `pytest --skip-extra` green
- [ ] file the two follow-ups: one `discopy.neural` from #399/#585/#686 on `para` and `optics`, and the ARC-AGI-1 hand-off
