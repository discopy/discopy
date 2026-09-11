# TODO

Review feedback from @toumix on `discopy/abc.py`, quoted verbatim:

> I'd rather keep `Category` as the first class of the `abc` module so let's move `Serialisable` to the end of the `axioms` module?

> Let's add an issue to add `str_transparency` as an axiom too (it requires a bit more work to define the environment in that case)
>
> ```suggestion
>     def repr_transparency(cls, term: Self) -> Equation:
> ```

- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 13:54 Move `Serialisable` to the end of `discopy/axioms.py`
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 13:54 Rename `transparency` to `repr_transparency`
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 13:54 File the issue for `str_transparency` and link it
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 13:54 `CHANGELOG.md` and the docs that name either
- [WIP] @session_01PoVD21cMmz9uw2UqreoNwM-2026-09-11 13:54 `pflake8`, the test suite and the property matrix
