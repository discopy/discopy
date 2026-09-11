# TODO

> remove @proptest/carriers.py and put everything into @proptest/test_axioms.py. don't filter based on `issubclass(Testable) and "strategy" in cls.__dict__`, instead you should manually override `def axioms` for every class that does not implement `Testable` yet, raising `NotImplementedError(f"No search strategy implemented for {cls.__name__}")`

- [ ] Declare the opt-out once in `discopy.axioms`, next to `Theory.axioms`
- [ ] Override `axioms` on every class that does not generate its own terms
- [ ] Fold `proptest/carriers.py` into `proptest/test_axioms.py` and delete it
- [ ] Update `CHANGELOG.md`, `CONTRIBUTING.md` and `AGENTS.md`
- [ ] `pflake8`, the test suite and the property matrix
