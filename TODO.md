# TODO

Review feedback from @toumix on `discopy/abc.py`, quoted verbatim:

> I'd rather keep `Category` as the first class of the `abc` module so let's move `Serialisable` to the end of the `axioms` module?

> Let's add an issue to add `str_transparency` as an axiom too (it requires a bit more work to define the environment in that case)
>
> ```suggestion
>     def repr_transparency(cls, term: Self) -> Equation:
> ```

- [ ] Move `Serialisable` to the end of `discopy/axioms.py`
- [ ] Rename `transparency` to `repr_transparency`
- [ ] File the issue for `str_transparency` and link it
- [ ] `CHANGELOG.md` and the docs that name either
- [ ] `pflake8`, the test suite and the property matrix
